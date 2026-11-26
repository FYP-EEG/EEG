"""
Author: Anson
Created on: 12/11/2026
Purpose: one command to localize BCI failure across hardware, raw data,
         filtering, class separability, offline classifiers and live logs.

Edited by Anson
Date: 18/11/2026
Purpose: include the first explicit NPZ screening step that rejected data.

Offline existing recordings:
    python tools/debug_bci_pipeline.py \
      --input "dataset/motor_intent/calib_brian_*.npz" \
      --out reports/brian_pipeline_debug

Hardware + recordings:
    python tools/debug_bci_pipeline.py --port COM8 --montage cyton8_motor \
      --input "dataset/motor_intent/calib_brian_*.npz" \
      --out reports/brian_pipeline_debug

The script stops interpreting downstream layers after an upstream failure unless
``--force-downstream`` is supplied. A smooth filtered waveform or high training
score never overrides a failed raw-signal layer.
"""

import argparse
import csv
import glob
import html
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import f_oneway

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_sdk.performance import TrainingDataPerformance
from EEG.signal_preflight import analyse_preflight
from ML.montage import MONTAGES, get as get_montage


def _expand(patterns):
    files, seen = [], set()
    for value in patterns or []:
        path = Path(value)
        matches = sorted(path.glob("calib_*.npz")) if path.is_dir() else [
            Path(item) for item in sorted(glob.glob(str(value)))]
        if path.is_file() and path not in matches:
            matches.append(path)
        for match in matches:
            resolved = str(match.resolve())
            if resolved not in seen:
                files.append(match); seen.add(resolved)
    return files


def load_reference_csv(path, columns, time_column="session_time", fs=None,
                       time_unit="auto", value_unit="uv", scale_to_uv=None):
    """Load a continuous comparison CSV and convert channel values to µV."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        lookup = {name.strip().lower(): name for name in reader.fieldnames}
        wanted = [value.strip() for value in columns]
        missing = [name for name in [time_column, *wanted]
                   if name.strip().lower() not in lookup]
        if missing:
            raise ValueError(
                f"CSV missing columns {missing}; found {reader.fieldnames}")
        time_key = lookup[time_column.strip().lower()]
        channel_keys = [lookup[name.lower()] for name in wanted]
        times, rows = [], []
        for row_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row[time_key]))
                rows.append([float(row[key]) for key in channel_keys])
            except (TypeError, ValueError):
                # Ignore blank/footer rows but fail if nothing numeric remains.
                continue
    if len(rows) < 10:
        raise ValueError("CSV contains fewer than 10 numeric EEG rows")
    times = np.asarray(times, dtype=float)
    data = np.asarray(rows, dtype=float).T
    if fs is None:
        diffs = np.diff(times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if not len(diffs):
            raise ValueError("cannot infer sample rate from session_time")
        dt = float(np.median(diffs))
        if time_unit == "s":
            fs = 1.0 / dt
        elif time_unit == "ms":
            fs = 1000.0 / dt
        else:
            candidates = [1.0 / dt, 1000.0 / dt]
            plausible = [value for value in candidates if 20 <= value <= 5000]
            if not plausible:
                raise ValueError(
                    f"cannot infer plausible Hz from median time step {dt}; "
                    "pass --reference-fs or --reference-time-unit")
            fs = min(plausible, key=lambda value: abs(value - 250.0))
    factor = (float(scale_to_uv) if scale_to_uv is not None else
              {"uv": 1.0, "mv": 1000.0, "v": 1_000_000.0}[value_unit])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("scale-to-µV factor must be finite and greater than zero")
    data *= factor
    return {"path": str(path), "data": data, "fs": float(fs),
            "labels": wanted, "value_unit_input": value_unit,
            "scale_to_uv": factor, "time_column": time_column,
            "rows": int(data.shape[1])}


def reference_layer(reference, thresholds, window_s=3.0):
    """Apply the same amplitude/flat/jump rules to continuous CSV windows."""
    data, fs = reference["data"], reference["fs"]
    result = analyse_preflight(
        data, fs=fs, elapsed=data.shape[1] / fs,
        channel_labels=reference["labels"], thresholds=thresholds)
    window_n = max(2, int(round(window_s * fs)))
    windows = [data[:, start:start + window_n]
               for start in range(0, data.shape[1] - window_n + 1, window_n)]
    reason_counts = {"amplitude": 0, "flat": 0, "jump": 0}
    kept = 0
    for window in windows:
        window_result = analyse_preflight(
            window, fs=fs, elapsed=window.shape[1] / fs,
            channel_labels=reference["labels"], thresholds=thresholds)
        failed = {
            reason: any(reason in row["reasons"]
                        for row in window_result["channels"])
            for reason in reason_counts
        }
        for reason, value in failed.items():
            reason_counts[reason] += int(value)
        kept += int(not any(failed.values()))
    window_drop_rate = (1.0 - kept / len(windows)) if windows else 1.0
    passed = bool(result["connection_ok"] and windows and window_drop_rate <= .33)
    details = {**result, "source": reference["path"],
               "input_unit": reference["value_unit_input"],
               "scale_to_uv": reference["scale_to_uv"],
               "window_s": window_s, "windows": len(windows), "kept": kept,
               "drop_rate": window_drop_rate,
               "drop_reason_counts": reason_counts}
    # This is comparison evidence, not part of the target pipeline's first-fail
    # chain. WARN rather than FAIL prevents a poor external reference from
    # hiding the target recording's own first failure.
    status = "PASS" if passed else "WARN"
    summary = (f"{kept}/{len(windows)} non-overlapping {window_s:g}-s windows retained; "
               f"drop reasons={reason_counts}")
    return _stage("REF external CSV raw quality", status, summary, details)


def _source_trace_svg(data, fs, labels, title, seconds=3.0):
    """Draw every node with its own labelled physical µV scale."""
    n = min(data.shape[-1], max(2, int(round(fs * seconds))))
    values = np.asarray(data[:, :n], dtype=float)
    width, row_h = 1000, 78
    height = row_h * len(labels) + 35
    pieces = [f"<h3>{html.escape(title)}</h3>",
              "<p>Each row is median-centred and has its own labelled µV scale. "
              "Node names are kept source-specific; unlike names are not treated "
              "as if they were the same electrode.</p>",
              f'<svg viewBox="0 0 {width} {height}">']
    for row, (label, channel) in enumerate(zip(labels, values)):
        channel = channel - np.median(channel)
        scale = max(float(np.percentile(np.abs(channel), 98)), 1e-9)
        centre = 20 + row * row_h + row_h / 2
        take = np.linspace(0, len(channel) - 1, min(700, len(channel))).astype(int)
        points = []
        for pos, value in enumerate(channel[take]):
            px = 68 + pos * 920 / max(1, len(take) - 1)
            py = centre - float(np.clip(value / scale, -1, 1)) * row_h * .38
            points.append(f"{px:.1f},{py:.1f}")
        pieces.extend([
            f'<text x="4" y="{centre + 4:.1f}" fill="#dce8eb" '
            f'font-size="12">{html.escape(str(label))}</text>',
            f'<line x1="68" y1="{centre:.1f}" x2="988" y2="{centre:.1f}" '
            f'stroke="#294653"/>',
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="#66c7e8" stroke-width="1.2"/>',
            f'<text x="795" y="{centre - row_h * .31:.1f}" fill="#9fb7bf" '
            f'font-size="10">98% ±{scale:.2f} µV</text>',
        ])
    pieces.append("</svg>")
    return "".join(pieces)


def _common_channel_overlay(reference, monitor, title):
    """SVG overlay on common nodes using one shared physical µV scale."""
    ref_labels = {label.strip().upper(): index
                  for index, label in enumerate(reference["labels"])}
    target_labels = {label.strip().upper(): index
                     for index, label in enumerate(monitor.channel_labels)}
    common = [name for name in ref_labels if name in target_labels][:5]
    if not common:
        return (f"<h3>{html.escape(title)}</h3><p>No common channel names between "
                "the CSV and NPZ metadata; compare per-source tables instead.</p>")
    ref_n = min(reference["data"].shape[1], int(round(reference["fs"] * 3)))
    target = monitor.X[0]
    width, row_h = 1000, 100
    height = row_h * len(common) + 35
    pieces = [f"<h3>{html.escape(title)}</h3>",
              "<p>Shared µV scale per node: green=external CSV, red=target NPZ. "
              "A nearly flat green line beside a large red trace indicates the "
              "target amplitude is orders of magnitude larger.</p>",
              f'<svg viewBox="0 0 {width} {height}">']
    for row, name in enumerate(common):
        a = reference["data"][ref_labels[name], :ref_n]
        b = target[target_labels[name]]
        a = a - np.median(a); b = b - np.median(b)
        scale = max(float(np.percentile(np.abs(a), 98)),
                    float(np.percentile(np.abs(b), 98)), 1e-9)
        centre = 25 + row * row_h + row_h / 2
        pieces.append(f'<text x="4" y="{centre + 4:.1f}" fill="#dce8eb" '
                      f'font-size="12">{name}</text>')
        pieces.append(f'<line x1="55" y1="{centre:.1f}" x2="990" '
                      f'y2="{centre:.1f}" stroke="#294653"/>')
        for values, color in ((a, "#47d7a0"), (b, "#ff806f")):
            take = np.linspace(0, len(values) - 1,
                               min(500, len(values))).astype(int)
            points = []
            for pos, value in enumerate(values[take]):
                px = 55 + pos * 935 / max(1, len(take) - 1)
                py = centre - float(np.clip(value / scale, -1, 1)) * row_h * .4
                points.append(f"{px:.1f},{py:.1f}")
            pieces.append(f'<polyline points="{" ".join(points)}" fill="none" '
                          f'stroke="{color}" stroke-width="1.2"/>')
        pieces.append(f'<text x="820" y="{centre - row_h * .32:.1f}" '
                      f'fill="#9fb7bf" font-size="10">±{scale:.1f} µV</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def spectral_comparison_layer(reference, monitor):
    """Compare Welch band power only for nodes common to both montages."""
    ref_lookup = {label.strip().upper(): index
                  for index, label in enumerate(reference["labels"])}
    target_lookup = {label.strip().upper(): index
                     for index, label in enumerate(monitor.channel_labels)}
    common = sorted(set(ref_lookup) & set(target_lookup))
    if not common:
        return _stage(
            "REF common-node spectral comparison", "WARN",
            "no identically named nodes; no cross-montage spectral claim made",
            {"reference_labels": reference["labels"],
             "target_labels": monitor.channel_labels})
    ref_nperseg = min(reference["data"].shape[-1],
                      max(16, int(round(reference["fs"] * 2))))
    target_nperseg = min(monitor.X.shape[-1], max(16, int(round(monitor.fs * 2))))
    ref_f, ref_psd = welch(reference["data"], fs=reference["fs"],
                           nperseg=ref_nperseg, axis=-1)
    target_f, target_psd = welch(monitor.X, fs=monitor.fs,
                                 nperseg=target_nperseg, axis=-1)
    target_psd = np.median(target_psd, axis=0)
    bands = (("delta", 1, 4), ("theta", 4, 8), ("mu_alpha", 8, 13),
             ("beta", 13, 30), ("broad_eeg", 1, 40))

    def integrate(freq, psd, low, high):
        mask = (freq >= low) & (freq < high)
        return float(np.trapezoid(psd[mask], freq[mask])) if mask.sum() >= 2 else None

    rows = []
    for name in common:
        row = {"channel": name, "reference_uv2": {}, "target_npz_uv2": {}}
        for band, low, high in bands:
            row["reference_uv2"][band] = integrate(
                ref_f, ref_psd[ref_lookup[name]], low, high)
            row["target_npz_uv2"][band] = integrate(
                target_f, target_psd[target_lookup[name]], low, high)
        broad_ref = row["reference_uv2"]["broad_eeg"]
        broad_target = row["target_npz_uv2"]["broad_eeg"]
        row["reference_broad_rms_uv"] = (
            float(np.sqrt(broad_ref)) if broad_ref is not None else None)
        row["target_npz_broad_rms_uv"] = (
            float(np.sqrt(broad_target)) if broad_target is not None else None)
        row["target_to_reference_broad_power_ratio"] = (
            broad_target / broad_ref if broad_ref and broad_target is not None else None)
        rows.append(row)
    ratios = [row["target_to_reference_broad_power_ratio"] for row in rows
              if row["target_to_reference_broad_power_ratio"] is not None]
    ref_rms_values = [row["reference_broad_rms_uv"] for row in rows
                      if row["reference_broad_rms_uv"] is not None]
    ratio = float(np.median(ratios)) if ratios else None
    ref_rms = float(np.median(ref_rms_values)) if ref_rms_values else None
    low_reference_band = bool(ref_rms is not None and ref_rms < .1)
    summary = f"common nodes={', '.join(common)}"
    if ratio is not None:
        summary += f"; median target/reference 1-40 Hz power ratio={ratio:.3g}"
    if ref_rms is not None:
        summary += f"; reference median 1-40 Hz RMS={ref_rms:.3g} µV"
    if low_reference_band:
        summary += " (very low: verify conversion and acquisition filtering)"
    warnings = ["spectral differences do not establish MI usefulness or accuracy"]
    if low_reference_band:
        warnings.append(
            "Reference common-node 1-40 Hz power is near zero, so power ratios "
            "are denominator-sensitive and do not by themselves prove useful EEG.")
    return _stage(
        "REF common-node spectral comparison", "WARN", summary,
        {"common_channels": common, "channels": rows,
         "reference_median_broad_rms_uv": ref_rms,
         "reference_band_power_very_low": low_reference_band,
         "warnings": warnings})


def _stage(name, status, summary, details=None):
    return {"name": name, "status": status, "summary": summary,
            "details": details or {}}


def hardware_layers(port, montage, seconds=6.0, electrode_mode="thinkpulse",
                    cyton_gain=None):
    """Known ground, square and mode-specific live-electrode checks."""
    from EEG.cyton_commands import (live_channel_commands,
                                    normalize_electrode_mode)
    from tools.test_signal_check import capture, classify_waveform
    labels = list(get_montage(montage).get("labels", []))
    stages = []
    try:
        ground, fs, rows = capture("cyton", port, "0", seconds, 750)
        eeg = np.asarray(ground[rows, :], dtype=float)
        rms = np.std(eeg, axis=1)
        transfer_available = bool(
            np.isfinite(eeg).all() and eeg.shape[1] >= 750)
        ground_ok = bool(transfer_available and np.median(rms) < 100)
        stages.append(_stage(
            "L0A known internal ground", "PASS" if ground_ok else "FAIL",
            "Cyton/ADC noise-floor transfer" if ground_ok else
            "internal-ground signal is missing, nonfinite or excessively noisy",
            {"rms_uv": rms.tolist(), "samples": int(eeg.shape[1]), "fs": fs,
             "transfer_available": transfer_available}))
        # No sample flow means the board/radio/port path is unavailable. Do not
        # spend two more captures producing derivative square/Normal failures.
        if not transfer_available:
            return stages
    except Exception as exc:
        stages.append(_stage("L0A known internal ground", "FAIL",
                             f"{type(exc).__name__}: {exc}"))
        return stages

    try:
        square, fs, rows = capture("cyton", port, "=", seconds, 750)
        eeg = np.asarray(square[rows, :], dtype=float)
        channel_rows = []
        for index in range(min(8, len(eeg))):
            result = classify_waveform(
                eeg[index:index + 1, -750:], sample_freq=fs,
                band=(.5, 45.0))
            channel_rows.append({
                "channel": index, "kind": result["kind"],
                "fundamental": result["fundamental"],
                "h3_ratio": result["h3_ratio"], "snr": result["snr"],
            })
        matched = sum(
            row["kind"] == "square" and row["fundamental"] is not None and
            abs(row["fundamental"] - 2.0) <= .6 for row in channel_rows)
        square_ok = matched >= 7
        stages.append(_stage(
            "L0B known square-wave transfer", "PASS" if square_ok else "FAIL",
            f"{matched}/8 channels decoded as approximately 2-Hz square waves",
            {"channels": channel_rows, "samples": int(eeg.shape[1]), "fs": fs}))
    except Exception as exc:
        stages.append(_stage("L0B known square-wave transfer", "FAIL",
                             f"{type(exc).__name__}: {exc}"))
        return stages

    try:
        electrode_mode = normalize_electrode_mode(electrode_mode)
        resolved_gain = int(cyton_gain if cyton_gain is not None else
                            (8 if electrode_mode == "thinkpulse" else 24))
        commands = live_channel_commands(electrode_mode, gain=resolved_gain)
        normal, fs, rows = capture(
            "cyton", port, None, seconds, 750,
            channel_commands=commands)
        eeg = np.asarray(normal[rows[:8], -int(seconds * fs):], dtype=float)
        result = analyse_preflight(
            eeg, fs=fs, elapsed=eeg.shape[1] / fs, channel_labels=labels)
        stages.append(_stage(
            "L1 live normal-electrode signal",
            "PASS" if result["acceptable"] else "FAIL",
            "normal electrode input passes gross checks" if result["acceptable"]
            else "; ".join(result["debug"]),
            {**result, "electrode_mode": electrode_mode,
             "cyton_gain": resolved_gain}))
    except Exception as exc:
        stages.append(_stage("L1 live normal-electrode signal", "FAIL",
                             f"{type(exc).__name__}: {exc}"))
    return stages


def raw_layer(files, screening_kw, channel_override=None):
    stages, monitors = [], []
    for path in files:
        try:
            monitor = TrainingDataPerformance([path], screening_kw=screening_kw)
            report = monitor.report()
            if channel_override:
                monitor.channel_labels = list(channel_override)
                report["channel_labels"] = list(channel_override)
                for row, label in zip(report["channel_diagnostics"], channel_override):
                    row["label"] = label
            passed = report["kept"] > 0 and report["drop_rate"] <= .33
            status = "PASS" if passed else "FAIL"
            offenders = {
                reason: count for reason, count
                in report["drop_reason_counts"].items() if count}
            first_rejector = report.get("recording_decision", {}).get(
                "first_rejecting_step")
            summary = (f"{report['kept']}/{report['trials']} retained; "
                       f"drop reasons={offenders or 'none'}; "
                       f"first rejecting step={first_rejector or 'none'}")
            stages.append(_stage(f"L1 saved raw NPZ · {path.name}", status,
                                 summary, report))
            monitors.append((path, monitor, report, passed))
        except Exception as exc:
            stages.append(_stage(f"L1 saved raw NPZ · {path.name}", "FAIL",
                                 f"{type(exc).__name__}: {exc}"))
    return stages, monitors


def filtered_layer(path, monitor, raw_pass, force=False):
    if not raw_pass and not force:
        return _stage(f"L2 8-30 Hz filter · {path.name}", "SKIP",
                      "upstream raw-signal layer failed")
    try:
        sos = butter(4, [8, 30], btype="band", fs=monitor.fs, output="sos")
        filtered = sosfiltfilt(sos, monitor.X, axis=-1)
        finite = bool(np.isfinite(filtered).all())
        std = np.std(filtered, axis=2)
        dead = [monitor.channel_labels[index] for index in range(std.shape[1])
                if np.median(std[:, index]) < 1e-6]
        # Covariance positive-definiteness after the same regularization used
        # by the actual Riemannian pipeline.
        min_eigs = []
        for trial in filtered:
            centred = trial - trial.mean(axis=1, keepdims=True)
            cov = centred @ centred.T / max(1, trial.shape[1] - 1)
            trace = np.trace(cov)
            if trace > 0:
                cov /= trace
            cov = .95 * cov + .05 * np.eye(cov.shape[0]) / cov.shape[0]
            min_eigs.append(float(np.linalg.eigvalsh(cov).min()))
        passed = finite and not dead and min(min_eigs, default=-1) > 0
        return _stage(
            f"L2 8-30 Hz filter · {path.name}", "PASS" if passed else "FAIL",
            "filtered arrays finite/non-flat with valid covariance" if passed else
            f"finite={finite}, dead_after_filter={dead}, min_eigen={min(min_eigs, default=float('nan')):.3g}",
            {"filtered_std_median_uv": np.median(std, axis=0).tolist(),
             "dead_channels": dead, "finite": finite,
             "min_regularized_cov_eigenvalue": min(min_eigs, default=None)})
    except Exception as exc:
        return _stage(f"L2 8-30 Hz filter · {path.name}", "FAIL",
                      f"{type(exc).__name__}: {exc}")


def separability_layer(path, monitor, raw_pass, force=False):
    if not raw_pass and not force:
        return _stage(f"L3 class separability · {path.name}", "SKIP",
                      "upstream raw-signal layer failed")
    try:
        sos = butter(4, [8, 30], btype="band", fs=monitor.fs, output="sos")
        filtered = sosfiltfilt(sos, monitor.X, axis=-1)
        features = np.log(np.var(filtered, axis=2) + 1e-12)
        labels = monitor.labels
        classes = sorted(set(labels.tolist()))
        if len(classes) < 2 or min((labels == name).sum() for name in classes) < 4:
            return _stage(f"L3 class separability · {path.name}", "FAIL",
                          "need at least two classes and four trials per class")
        rows = []
        for channel in range(features.shape[1]):
            groups = [features[labels == name, channel] for name in classes]
            stat, pvalue = f_oneway(*groups)
            grand = features[:, channel].mean()
            between = sum(len(group) * (group.mean() - grand) ** 2 for group in groups)
            total = float(np.sum((features[:, channel] - grand) ** 2)) + 1e-12
            eta2 = float(between / total)
            rows.append({"channel": monitor.channel_labels[channel],
                         "anova_f": float(stat), "p": float(pvalue),
                         "eta_squared": eta2})
        best = max(rows, key=lambda row: row["eta_squared"])
        status = "PASS" if best["p"] < .05 and best["eta_squared"] >= .06 else "WARN"
        return _stage(
            f"L3 class separability · {path.name}", status,
            f"best channel {best['channel']}: eta²={best['eta_squared']:.3f}, p={best['p']:.3g}",
            {"classes": classes, "channels": rows,
             "warning": "within-session separability is not generalization"})
    except Exception as exc:
        return _stage(f"L3 class separability · {path.name}", "FAIL",
                      f"{type(exc).__name__}: {exc}")


def classifier_layer(monitors, skip=False, force=False):
    usable = [(path, monitor) for path, monitor, _, passed in monitors
              if passed or force]
    if skip:
        return _stage("L4 held-session classifiers", "SKIP",
                      "disabled by --skip-model")
    if len(usable) < 2:
        return _stage("L4 held-session classifiers", "SKIP",
                      "need at least two upstream-valid sessions")
    try:
        common = set(usable[0][1].labels.tolist())
        for _, monitor in usable[1:]:
            common &= set(monitor.labels.tolist())
        common = sorted(common)
        if len(common) < 2:
            return _stage("L4 held-session classifiers", "FAIL",
                          "sessions have fewer than two common classes")
        Xs, ys, groups = [], [], []
        to_y = {name: index for index, name in enumerate(common)}
        fs_values = {monitor.fs for _, monitor in usable}
        montages = {monitor.montage for _, monitor in usable}
        if len(fs_values) != 1 or len(montages) != 1:
            return _stage("L4 held-session classifiers", "FAIL",
                          f"incompatible fs={fs_values} or montages={montages}")
        for group, (_, monitor) in enumerate(usable):
            keep = np.isin(monitor.labels, common)
            Xs.append(monitor.X[keep])
            ys.append(np.array([to_y[value] for value in monitor.labels[keep]]))
            groups.append(np.full(int(keep.sum()), group))
        X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import GroupKFold
        from ML.mi_advanced import make_riemannian, make_riemannian_lr
        scores = {}
        splitter = GroupKFold(n_splits=len(usable))
        for name, factory in (("riemann_svm", make_riemannian),
                              ("riemann_lr", make_riemannian_lr)):
            fold_scores = []
            for train, test in splitter.split(X, y, g):
                if len(np.unique(y[train])) != len(common):
                    continue
                model = factory(sample_freq=next(iter(fs_values)))
                model.fit(X[train], y[train])
                fold_scores.append(float(balanced_accuracy_score(
                    y[test], model.predict(X[test]))))
            if fold_scores:
                scores[name] = fold_scores
        if not scores:
            return _stage("L4 held-session classifiers", "FAIL",
                          "no valid grouped folds")
        means = {name: float(np.mean(values)) for name, values in scores.items()}
        chance = 1.0 / len(common)
        best_name = max(means, key=means.get)
        best = means[best_name]
        status = "PASS" if len(usable) >= 3 and best > chance + .10 else "WARN"
        return _stage(
            "L4 held-session classifiers", status,
            f"best={best_name} {best:.1%}; chance={chance:.1%}; sessions={len(usable)}",
            {"classes": common, "scores": scores, "means": means,
             "chance": chance,
             "warning": "three sessions minimum for a reliable score"})
    except Exception as exc:
        return _stage("L4 held-session classifiers", "FAIL",
                      f"{type(exc).__name__}: {exc}")


def runtime_layer(session_dir):
    if not session_dir:
        return _stage("L5 live gameplay", "SKIP",
                      "provide --runtime <session directory> to inspect live logs")
    try:
        status = json.loads((Path(session_dir) / "status.json").read_text())
        stats = status.get("stats", {})
        accuracy = status.get("manual_accuracy")
        return _stage(
            "L5 live gameplay", "WARN" if accuracy is None else "PASS",
            "no manual feedback yet" if accuracy is None else
            f"manual accuracy={accuracy:.1%}, feedback={stats.get('feedback_total', 0)}",
            status)
    except Exception as exc:
        return _stage("L5 live gameplay", "FAIL",
                      f"{type(exc).__name__}: {exc}")


def _first_failure(stages):
    for stage in stages:
        if stage["status"] == "FAIL":
            return stage["name"]
    return None


def write_report(stages, out_base, files, comparison_html=""):
    base = Path(out_base)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    payload = {"files": [str(path) for path in files],
               "first_failure": _first_failure(stages), "stages": stages}
    json_path = base.with_suffix(".json")
    html_path = base.with_suffix(".html")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cards = []
    for stage in stages:
        status = stage["status"]
        css = {"PASS": "good", "WARN": "warn", "FAIL": "bad",
               "SKIP": "skip"}[status]
        details = html.escape(json.dumps(stage["details"], indent=2))
        cards.append(
            f'<section class="card"><h2>{html.escape(stage["name"])}</h2>'
            f'<p class="{css}">{status}: {html.escape(stage["summary"])}</p>'
            f'<details><summary>details</summary><pre>{details}</pre></details></section>')
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>BCI pipeline debug</title>
<style>body{{font-family:Segoe UI,Arial;background:#071018;color:#dce8eb;margin:24px}}.card{{background:#0d1e29;border:1px solid #31505d;border-radius:12px;padding:15px;margin:13px 0}}p{{padding:9px;border-radius:7px}}.good{{background:#174b3a}}.warn{{background:#57461d}}.bad{{background:#5c2929}}.skip{{background:#27343b}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head><body>
<h1>BCI Layered Debug Report</h1><p>First failure: <b>{html.escape(str(payload['first_failure'] or 'none'))}</b>. Downstream success never overrides an upstream failure.</p>{''.join(cards)}{comparison_html}</body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return html_path, json_path


def main():
    parser = argparse.ArgumentParser(
        description="One-command layered BCI hardware/data/model debugger")
    parser.add_argument("--input", nargs="+", default=None,
                        help="NPZ files, directories or glob patterns")
    parser.add_argument("--port", default=None,
                        help="run live Cyton ground/square/normal tests on this COM port")
    parser.add_argument("--montage", choices=sorted(MONTAGES),
                        default="cyton8_motor")
    parser.add_argument("--channel-labels", default=None,
                        help="comma-separated physical NPZ channel order for reports/plots")
    parser.add_argument("--hardware-seconds", type=float, default=6.0)
    parser.add_argument("--electrode-mode", choices=("passive", "thinkpulse"),
                        default="thinkpulse")
    parser.add_argument("--cyton-gain", type=int,
                        choices=(1, 2, 4, 6, 8, 12, 24), default=None,
                        help="default 8 for ThinkPulse, 24 for passive electrodes")
    parser.add_argument("--runtime", default=None,
                        help="runtime_logs/<app>/<timestamp> directory")
    parser.add_argument("--reference-csv", default=None,
                        help="continuous comparison EEG CSV")
    parser.add_argument(
        "--reference-columns", default="Fp1,Fp2,Fz,Cz,FC1,FC2,C3,C4",
        help="comma-separated EEG columns in physical order")
    parser.add_argument("--reference-time-column", default="session_time")
    parser.add_argument("--reference-fs", type=float, default=None,
                        help="CSV sample rate; otherwise infer from session_time")
    parser.add_argument("--reference-time-unit", choices=("auto", "s", "ms"),
                        default="auto")
    parser.add_argument("--reference-unit", choices=("uv", "mv", "v"),
                        default="uv", help="physical unit stored in CSV EEG columns")
    parser.add_argument(
        "--reference-scale-to-uv", type=float, default=None,
        help="custom multiplier from stored values to µV; overrides --reference-unit")
    parser.add_argument("--out", default=str(ROOT / "reports" / "bci_pipeline_debug"))
    parser.add_argument("--force-downstream", action="store_true",
                        help="diagnose later layers even after raw failure; results are not validity evidence")
    parser.add_argument("--skip-model", action="store_true",
                        help="skip potentially slow held-session classifier comparison")
    parser.add_argument("--amp-uv", type=float, default=None,
                        help="optional explicit cap; default uses adaptive scaling")
    parser.add_argument("--jump-uv", type=float, default=None,
                        help="optional explicit cap; default uses adaptive scaling")
    parser.add_argument("--flat-uv", type=float, default=.05)
    parser.add_argument("--z-max", type=float, default=4.0)
    args = parser.parse_args()
    if (args.electrode_mode == "thinkpulse" and
            args.cyton_gain is not None and args.cyton_gain > 8):
        parser.error("ThinkPulse requires --cyton-gain 8 or lower")

    files = _expand(args.input or [])
    if not args.port and not files and not args.runtime and not args.reference_csv:
        parser.error("provide --port, --input, --reference-csv and/or --runtime")
    labels = ([value.strip() for value in args.channel_labels.split(",")]
              if args.channel_labels else None)
    if labels and len(labels) != 8:
        parser.error("--channel-labels requires exactly 8 comma-separated labels")
    screening_kw = {"amp_uv": args.amp_uv, "jump_uv": args.jump_uv,
                    "flat_uv": args.flat_uv, "z_max": args.z_max}
    stages = []
    if args.port:
        print(f"Running hardware transfer and normal-input checks on {args.port}...")
        stages.extend(hardware_layers(
            args.port, args.montage, seconds=args.hardware_seconds,
            electrode_mode=args.electrode_mode, cyton_gain=args.cyton_gain))
    monitors = []
    if files:
        raw_stages, monitors = raw_layer(files, screening_kw, labels)
        stages.extend(raw_stages)
        for path, monitor, _, passed in monitors:
            stages.append(filtered_layer(path, monitor, passed,
                                         force=args.force_downstream))
            stages.append(separability_layer(path, monitor, passed,
                                             force=args.force_downstream))
        stages.append(classifier_layer(monitors, skip=args.skip_model,
                                       force=args.force_downstream))

    comparison_html = ""
    if args.reference_csv:
        try:
            reference = load_reference_csv(
                args.reference_csv,
                [value.strip() for value in args.reference_columns.split(",")
                 if value.strip()],
                time_column=args.reference_time_column,
                fs=args.reference_fs, time_unit=args.reference_time_unit,
                value_unit=args.reference_unit,
                scale_to_uv=args.reference_scale_to_uv)
            window_s = (monitors[0][1].X.shape[-1] / monitors[0][1].fs
                        if monitors else 3.0)
            stages.append(reference_layer(reference, screening_kw,
                                          window_s=window_s))
            comparison_html += (
                '<section class="card"><h2>Cross-recording traces</h2>'
                + _source_trace_svg(
                    reference["data"], reference["fs"], reference["labels"],
                    f"External CSV: {Path(reference['path']).name}") )
            for path, monitor, _, _ in monitors:
                comparison = spectral_comparison_layer(reference, monitor)
                comparison["name"] += f" · {path.name}"
                stages.append(comparison)
                comparison_html += _source_trace_svg(
                    monitor.X[0], monitor.fs, monitor.channel_labels,
                    f"Target NPZ first epoch: {path.name}")
                comparison_html += _common_channel_overlay(
                    reference, monitor,
                    f"Common-node overlay: CSV versus {path.name}")
            comparison_html += "</section>"
        except Exception as exc:
            stages.append(_stage(
                "REF external CSV import", "FAIL",
                f"{type(exc).__name__}: {exc}",
                {"source": args.reference_csv}))
    stages.append(runtime_layer(args.runtime))

    print("\n" + "=" * 86)
    print("BCI LAYERED DEBUG SUMMARY")
    print("=" * 86)
    for stage in stages:
        print(f"  [{stage['status']:<4}] {stage['name']}: {stage['summary']}")
    first = _first_failure(stages)
    print(f"\n  FIRST FAILURE: {first or 'none detected'}")
    if first:
        print("  Fix this layer before interpreting downstream layers.")
    html_path, json_path = write_report(
        stages, args.out, files, comparison_html=comparison_html)
    print(f"\nHTML report -> {html_path}")
    print(f"JSON report -> {json_path}")
    return 1 if first else 0


if __name__ == "__main__":
    raise SystemExit(main())
