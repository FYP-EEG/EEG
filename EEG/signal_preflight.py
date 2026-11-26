"""
Author: Anson
Created on: 12/11/2026
Purpose: connection and raw-signal go/no-go gate before a timed EEG recording.

This module never filters or modifies samples. It checks that samples arrive at
approximately the declared rate, remain finite, are not flat, and do not contain
gross amplitude/jump faults. Reports explain likely causes and corrective steps.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _longest_true_run(mask):
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


DEFAULT_THRESHOLDS = {
    # None means robust, data-adaptive limits. A caller may still supply an
    # absolute cap for a controlled experiment, but startup does not assume one
    # fixed microvolt scale for passive and amplified active electrodes.
    "amp_uv": None,
    "jump_uv": None,
    "flat_uv": .05,
    "amplitude_z": 12.0,
    "jump_z": 12.0,
    "peer_scale_factor": 8.0,
    "clip_fraction": .02,
    "rate_tolerance": .15,
}


def collect_preflight(backend, seconds=6.0):
    """Collect a fresh continuous baseline from an already-started backend."""
    seconds = max(2.0, float(seconds))
    info = backend.info()
    fs = int(info["fs"])
    n_channels = int(info["n_channels"])
    # Discard priming data collected while the board opened/reset.
    try:
        backend.read_available()
    except Exception:
        pass
    chunks = []
    started = time.monotonic()
    deadline = started + seconds
    while time.monotonic() < deadline:
        chunk = backend.read_available()
        if chunk is not None and np.asarray(chunk).size:
            chunk = np.asarray(chunk, dtype=np.float64)
            if chunk.ndim != 2 or chunk.shape[0] < n_channels:
                raise ValueError(
                    f"backend returned {chunk.shape}, expected at least "
                    f"({n_channels}, samples)")
            chunks.append(chunk[:n_channels])
        time.sleep(.01)
    elapsed = time.monotonic() - started
    data = (np.concatenate(chunks, axis=1) if chunks else
            np.empty((n_channels, 0), dtype=np.float64))
    return data, fs, elapsed


def analyse_preflight(data, fs=250, elapsed=None, channel_labels=None,
                      thresholds=None):
    """Return connection, per-channel signal, verdict and debugging details."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (channels,samples), got {x.shape}")
    n_channels, n_samples = x.shape
    elapsed = float(elapsed if elapsed is not None else
                    n_samples / max(1, int(fs)))
    limits = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    labels = list(channel_labels or [])
    if len(labels) != n_channels:
        labels = [f"CH{index + 1}" for index in range(n_channels)]

    expected_samples = max(1, float(fs) * max(elapsed, 0.0))
    effective_fs = n_samples / max(elapsed, 1e-9)
    rate_error = abs(effective_fs - fs) / max(1.0, float(fs))
    finite = bool(np.isfinite(x).all())
    connection_ok = bool(
        n_channels == 8 and n_samples >= expected_samples * .70 and
        finite and rate_error <= limits["rate_tolerance"])

    channels = []
    if n_samples:
        median = np.median(x, axis=1)
        centred = x - median[:, np.newaxis]
        peak = np.max(np.abs(centred), axis=1)
        std = np.std(x, axis=1)
        robust_scale = 1.4826 * np.median(
            np.abs(centred - np.median(centred, axis=1)[:, np.newaxis]),
            axis=1)
        if n_samples > 1:
            differences = np.diff(x, axis=1)
            jump = np.max(np.abs(differences), axis=1)
            diff_median = np.median(differences, axis=1)
            robust_step = 1.4826 * np.median(
                np.abs(differences - diff_median[:, np.newaxis]), axis=1)
        else:
            jump = robust_step = np.full(n_channels, np.inf)
        span = np.ptp(x, axis=1)
        plateau = []
        plateau_run = []
        for row, row_span in zip(x, span):
            tolerance = max(float(row_span) * 1e-7, 1e-9)
            low_mask = np.abs(row - np.min(row)) <= tolerance
            high_mask = np.abs(row - np.max(row)) <= tolerance
            plateau.append(max(float(np.mean(low_mask)),
                                float(np.mean(high_mask))))
            plateau_run.append(max(_longest_true_run(low_mask),
                                   _longest_true_run(high_mask)))
        plateau = np.asarray(plateau)
        plateau_run = np.asarray(plateau_run)
    else:
        median = peak = std = jump = robust_scale = robust_step = span = plateau = \
            np.full(n_channels, np.nan)
        plateau_run = np.zeros(n_channels, dtype=int)

    valid_scales = robust_scale[np.isfinite(robust_scale) &
                                (robust_scale >= limits["flat_uv"])]
    valid_steps = robust_step[np.isfinite(robust_step) &
                              (robust_step >= limits["flat_uv"])]
    peer_scale = float(np.median(valid_scales)) if valid_scales.size else 0.0
    peer_step = float(np.median(valid_steps)) if valid_steps.size else 0.0

    for index in range(n_channels):
        reasons = []
        metrics = [median[index], peak[index], std[index], jump[index],
                   robust_scale[index], robust_step[index], plateau[index]]
        adaptive_amp_limit = max(
            limits["amplitude_z"] * max(robust_scale[index], limits["flat_uv"]),
            limits["peer_scale_factor"] * max(peer_scale, limits["flat_uv"]))
        adaptive_jump_limit = max(
            limits["jump_z"] * max(robust_step[index], limits["flat_uv"]),
            limits["peer_scale_factor"] * max(peer_step, limits["flat_uv"]))
        if not np.isfinite(metrics).all():
            reasons.append("nonfinite")
        else:
            # Absolute DC offset is reported but is not itself rejected: active
            # electrodes and references can use very different common offsets.
            # True ADC saturation is scale-independent evidence: many samples
            # repeat at an extreme despite a non-trivial overall span.
            rail = bool(
                plateau[index] >= limits["clip_fraction"] and
                plateau_run[index] >= 3 and
                span[index] > limits["flat_uv"] * 20.0)
            if rail:
                reasons.append("rail")
            elif std[index] < limits["flat_uv"]:
                reasons.append("flat")
            else:
                absolute_amp = (limits["amp_uv"] is not None and
                                peak[index] > float(limits["amp_uv"]))
                relative_amp = bool(
                    peak[index] > adaptive_amp_limit or
                    (peer_scale > 0 and robust_scale[index] >
                     limits["peer_scale_factor"] * peer_scale))
                if absolute_amp or relative_amp:
                    reasons.append("amplitude")
                absolute_jump = (limits["jump_uv"] is not None and
                                 jump[index] > float(limits["jump_uv"]))
                relative_jump = bool(
                    jump[index] > adaptive_jump_limit or
                    (peer_step > 0 and robust_step[index] >
                     limits["peer_scale_factor"] * peer_step))
                if absolute_jump or relative_jump:
                    reasons.append("jump")
        channels.append({
            "index": index, "label": labels[index],
            "median_uv": float(median[index]),
            "peak_uv": float(peak[index]), "std_uv": float(std[index]),
            "robust_scale_uv": float(robust_scale[index]),
            "jump_uv": float(jump[index]),
            "clip_fraction": float(plateau[index]),
            "longest_extreme_run": int(plateau_run[index]),
            "amplitude_limit_uv": float(adaptive_amp_limit),
            "jump_limit_uv": float(adaptive_jump_limit),
            "reasons": reasons, "ok": not reasons,
        })

    bad = [row for row in channels if not row["ok"]]
    any_flat = any("flat" in row["reasons"] for row in channels)
    any_rail = any("rail" in row["reasons"] for row in channels)
    stable_offset = bool(channels and any(
        abs(row["median_uv"]) > limits["amplitude_z"] *
        max(row["robust_scale_uv"], limits["flat_uv"])
        for row in channels if np.isfinite(row["median_uv"])))
    boardwide_range = bool(channels and all(
        {"rail", "amplitude"}.intersection(row["reasons"])
        for row in channels))
    possible_test_signal = bool(
        len(channels) == 8 and
        np.all(plateau >= limits["clip_fraction"]) and
        np.nanmedian(std) > 500)
    signal_ok = bool(connection_ok and not bad)

    debug = []
    if n_samples == 0:
        debug.append("No samples arrived: verify COM port, board power, radio and board mode.")
    if rate_error > limits["rate_tolerance"]:
        debug.append(
            f"Effective rate {effective_fs:.1f} Hz differs from nominal {fs} Hz: "
            "check dropped packets, streaming source and board selection.")
    if any_flat:
        debug.append(
            "Flat channel(s): near-zero variation was measured around its raw "
            "median; verify power, Normal input and electrode contact.")
    if any_rail:
        debug.append(
            "Rail/clipping detected from repeated extrema. After median "
            "centring or high-pass filtering, a saturated raw trace can look "
            "flat even though the unprocessed level was out of range.")
    elif stable_offset:
        debug.append(
            "A large stable raw offset was retained: offset alone is not an "
            "amplitude fault. This code measures variation around each channel's "
            "median so active-electrode scaling is handled from the signal.")
    if boardwide_range:
        debug.append(
            "All channels are over-range/railed. ThinkPulse startup will lower "
            "Cyton gain stepwise (8 -> 4 -> 2 -> 1); also verify sensor power "
            "and SRB2 reference.")
    elif any("amplitude" in row["reasons"] or "jump" in row["reasons"]
             for row in channels):
        debug.append(
            "Adaptive amplitude/jump outlier: inspect the listed contact and "
            "cable movement. Stable common active-electrode scaling alone is "
            "not rejected.")
    if possible_test_signal:
        debug.append(
            "Pattern resembles a board-wide internal square/test signal rather "
            "than human EEG. Use Reset Channels or send Cyton command 'd'.")
    if not debug and signal_ok:
        debug.append("Connection and gross raw-signal checks passed.")

    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "shape": [int(n_channels), int(n_samples)], "fs": int(fs),
        "elapsed_s": elapsed, "effective_fs": effective_fs,
        "rate_error_pct": rate_error * 100.0, "finite": finite,
        "connection_ok": connection_ok, "signal_ok": signal_ok,
        "acceptable": signal_ok, "thresholds": limits,
        "adaptive_amplitude": True,
        "boardwide_range_failure": boardwide_range,
        "possible_test_signal": possible_test_signal,
        "channels": channels, "debug": debug,
    }


def _trace_svg(data, labels, width=920, height=310):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or not x.shape[1]:
        return "<p>No samples were available to draw.</p>"
    rows, samples = x.shape
    left, right = 62, width - 10
    row_h = (height - 20) / max(1, rows)
    take = np.arange(0, samples, max(1, int(np.ceil(samples / 500))))
    out = [f'<svg viewBox="0 0 {width} {height}">']
    for channel in range(rows):
        centre = 10 + (channel + .5) * row_h
        signal = x[channel] - np.nanmedian(x[channel])
        scale = max(float(np.nanpercentile(np.abs(signal), 98)), 1e-9)
        values = np.clip(signal[take] / scale, -1, 1)
        points = []
        for index, value in enumerate(values):
            px = left + index * (right - left) / max(1, len(values) - 1)
            py = centre - float(value) * row_h * .38
            points.append(f"{px:.1f},{py:.1f}")
        out.append(f'<line x1="{left}" y1="{centre:.1f}" x2="{right}" '
                   f'y2="{centre:.1f}" stroke="#263d49"/>')
        out.append(f'<text x="4" y="{centre + 4:.1f}" fill="#bfd3d9" '
                   f'font-size="11">{labels[channel]}</text>')
        out.append(f'<polyline points="{" ".join(points)}" fill="none" '
                   f'stroke="#47d7d0" stroke-width="1.2"/>')
    out.append("</svg>")
    return "".join(out)


def save_preflight_report(result, data, out_base):
    """Write JSON and self-contained HTML; return ``(html_path,json_path)``."""
    base = Path(out_base)
    if base.suffix.lower() in (".html", ".json"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path, html_path = base.with_suffix(".json"), base.with_suffix(".html")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = "".join(
        f"<tr><td>{row['index'] + 1}</td><td>{row['label']}</td>"
        f"<td>{row['median_uv']:.3f}</td><td>{row['peak_uv']:.3f}</td>"
        f"<td>{row['amplitude_limit_uv']:.3f}</td>"
        f"<td>{row['std_uv']:.3f}</td><td>{row['jump_uv']:.3f}</td>"
        f"<td>{row['clip_fraction']:.3%}</td>"
        f"<td>{', '.join(row['reasons']) or 'PASS'}</td></tr>"
        for row in result["channels"])
    debug = "".join(f"<li>{line}</li>" for line in result["debug"])
    color = "#7ddb9b" if result["acceptable"] else "#ff806f"
    labels = [row["label"] for row in result["channels"]]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>EEG preflight</title>
<style>body{{font-family:Segoe UI,Arial;background:#071018;color:#dce8eb;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border-bottom:1px solid #294653;text-align:left}}svg{{width:100%;background:#0a1720;border-radius:10px}}.verdict{{font-size:22px;font-weight:700;color:{color}}}</style></head><body>
<h1>EEG Recording Preflight</h1><p class="verdict">{'PASS' if result['acceptable'] else 'FAIL — recording blocked'}</p>
<p>Shape {result['shape']} · effective rate {result['effective_fs']:.1f} Hz · nominal {result['fs']} Hz</p>
<ul>{debug}</ul><table><thead><tr><th>#</th><th>Channel</th><th>Raw median µV</th><th>Peak about median µV</th><th>Adaptive limit µV</th><th>SD µV</th><th>Max jump µV/sample</th><th>Clipped/plateau fraction</th><th>Result</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Preflight baseline (auto-scaled per channel)</h2>{_trace_svg(data, labels)}</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return html_path, json_path


def print_preflight(result):
    print("\nEEG PREFLIGHT")
    print("-" * 78)
    print(f"  connection : {'PASS' if result['connection_ok'] else 'FAIL'}")
    print(f"  signal     : {'PASS' if result['signal_ok'] else 'FAIL'}")
    print(f"  samples    : {result['shape'][1]} at {result['effective_fs']:.1f} Hz")
    print(f"  {'ch':>2} {'node':<5} {'median':>10} {'peak':>10} "
          f"{'auto lim':>10} {'std':>10} {'clip%':>8}  result")
    for row in result["channels"]:
        status = "PASS" if row["ok"] else ",".join(row["reasons"])
        print(f"  {row['index'] + 1:>2} {row['label']:<5} "
              f"{row['median_uv']:>10.2f} {row['peak_uv']:>10.2f} "
              f"{row['amplitude_limit_uv']:>10.2f} {row['std_uv']:>10.2f} "
              f"{row['clip_fraction'] * 100:>7.2f}%  {status}")
    print("\n  Debug guidance:")
    for line in result["debug"]:
        print(f"   - {line}")
    print("-" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive all-channel raw-signal preflight on NPZ or "
                    "OpenBCI GUI CSV input")
    parser.add_argument("npz", nargs="?", help="NPZ with data or X and optional fs")
    parser.add_argument("--csv", dest="openbci_csv", default=None,
                        help="OpenBCI/electrode-labelled CSV with metadata or timestamps")
    parser.add_argument("--fs", type=float, default=None,
                        help="CSV sample-rate override when timestamps are unavailable")
    parser.add_argument("--out", default="reports/signal_preflight")
    args = parser.parse_args()
    if bool(args.npz) == bool(args.openbci_csv):
        parser.error("provide exactly one NPZ path or --csv OPENBCI_GUI.csv")
    if args.openbci_csv:
        from EEG.stream_bridge import load_openbci_gui_csv
        loaded = load_openbci_gui_csv(
            args.openbci_csv, n_channels=8, fs=args.fs)
        data = loaded["data"]
        fs = loaded["fs"]
        labels = loaded["channel_labels"]
    else:
        with np.load(args.npz, allow_pickle=True) as saved:
            if "data" in saved:
                data = saved["data"]
            elif "X" in saved:
                array = saved["X"]
                data = array[0] if array.ndim == 3 else array
            else:
                raise SystemExit("NPZ must contain data or X")
            fs = (float(np.asarray(saved["fs"]).item())
                  if "fs" in saved else 250.0)
            labels = (np.asarray(saved["channel_labels"]).astype(str).tolist()
                      if "channel_labels" in saved else None)
    result = analyse_preflight(
        data, fs=fs, elapsed=np.asarray(data).shape[-1] / fs,
        channel_labels=labels)
    print_preflight(result)
    html_path, json_path = save_preflight_report(result, data, args.out)
    print(f"  HTML -> {html_path}\n  JSON -> {json_path}")
    return 0 if result["acceptable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
