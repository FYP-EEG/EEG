"""
Author: Anson
Created on: 12/11/2026
Purpose: visualize multiple recorded NPZ sessions and assess training readiness.

Example:
    python tools/inspect_training_recordings.py \
      --input "dataset/motor_intent/calib_brian_*.npz" \
      --out reports/training_data_brian.html

The verdict is a software/signal-quality readiness check, not proof of motor
imagery. Biological usefulness still requires held-session and, for a general
model, held-subject evaluation.
"""

import argparse
import glob
import html
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_sdk.performance import TrainingDataPerformance
from ML.motor_intent import IDLE, UNIVERSAL_INTENTS, make_intent_label

EXPECTED_UNIVERSAL = tuple(make_intent_label(value)
                           for value in (*UNIVERSAL_INTENTS, IDLE))
MOTOR_CORE = {"C3", "CZ", "C4"}


def _expand(patterns):
    files, seen = [], set()
    for value in patterns:
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


def _trace_svg(window, labels, width=900, height=300):
    x = np.asarray(window, dtype=float)
    if x.ndim != 2 or not x.size:
        return "<p>No valid EEG epoch to draw.</p>"
    channels, samples = x.shape
    left, right = 58, width - 10
    row_h = (height - 20) / max(1, channels)
    step = max(1, int(np.ceil(samples / 450)))
    take = np.arange(0, samples, step)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" '
              'aria-label="representative EEG epoch">']
    for channel in range(channels):
        centre = 10 + (channel + .5) * row_h
        signal = x[channel] - np.nanmedian(x[channel])
        scale = max(float(np.nanpercentile(np.abs(signal), 98)), 1e-6)
        values = np.clip(signal[take] / scale, -1, 1)
        points = []
        for index, value in enumerate(values):
            px = left + index * (right - left) / max(1, len(values) - 1)
            py = centre - float(value) * row_h * .38
            points.append(f"{px:.1f},{py:.1f}")
        name = labels[channel] if channel < len(labels) else f"CH{channel + 1}"
        pieces.append(
            f'<line x1="{left}" y1="{centre:.1f}" x2="{right}" '
            f'y2="{centre:.1f}" stroke="#263d49" stroke-width="1"/>')
        pieces.append(
            f'<text x="4" y="{centre + 4:.1f}" fill="#bfd3d9" '
            f'font-size="11">{html.escape(str(name))}</text>')
        pieces.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="#47d7d0" stroke-width="1.2"/>')
    pieces.append("</svg>")
    return "".join(pieces)


def _assess(report, min_per_class=8):
    labels = {str(value).upper() for value in report["channel_labels"]}
    has_motor_core = MOTOR_CORE.issubset(labels)
    retained = report["retained_counts"]
    present_min = min(retained.values()) if retained else 0
    missing = [label for label in EXPECTED_UNIVERSAL
               if report["class_counts"].get(label, 0) == 0]

    if report["kept"] == 0:
        verdict = "NOT READY: all trials rejected"
        level = "bad"
    elif report["drop_rate"] > .33:
        verdict = "NOT READY: more than one-third rejected"
        level = "bad"
    elif not has_motor_core:
        verdict = "LIMITED: signal may be usable, but montage lacks C3/Cz/C4"
        level = "warn"
    elif present_min < int(min_per_class):
        verdict = f"INCOMPLETE: fewer than {min_per_class} retained trials in a class"
        level = "warn"
    else:
        verdict = "CANDIDATE: passes software quality/count checks"
        level = "good"
    return {
        "verdict": verdict, "level": level,
        "has_motor_core": has_motor_core,
        "missing_universal_labels": missing,
        "partial_universal": report["trials"] < 119 or bool(missing),
    }


def _bar_rows(counts, retained):
    maximum = max(counts.values(), default=1)
    rows = []
    for label, total in counts.items():
        kept = retained.get(label, 0)
        width = 100 * total / maximum
        rows.append(
            '<tr><td>' + html.escape(label.replace("intent::", "")) +
            f'</td><td>{kept}/{total}</td><td><div class="bar" '
            f'style="width:{width:.1f}%"></div></td></tr>')
    return "".join(rows)


def _diagnostic_rows(rows):
    output = []
    for row in rows:
        output.append(
            f"<tr><td>{row['index'] + 1}</td><td>{html.escape(str(row['label']))}</td>"
            f"<td>{row['peak_median_uv']:.2f} / {row['peak_p95_uv']:.2f} / {row['peak_max_uv']:.2f}</td>"
            f"<td>{row['amplitude_fail_trials']}</td>"
            f"<td>{row['std_median_uv']:.3f} / {row['std_min_uv']:.3f}</td>"
            f"<td>{row['flat_fail_trials']}</td>"
            f"<td>{row['jump_median_uv']:.2f} / {row['jump_p95_uv']:.2f} / {row['jump_max_uv']:.2f}</td>"
            f"<td>{row['jump_fail_trials']}</td></tr>")
    return "".join(output)


def inspect_file(path, screening_kw, labels_override=None, min_per_class=8):
    try:
        monitor = TrainingDataPerformance([path], screening_kw=screening_kw)
    except TypeError as exc:
        # Backward compatibility when somebody copied this tools/ script but
        # still has the previous bci_sdk/performance.py. Re-run screening here
        # so command-line thresholds remain effective, then enrich the older
        # report with the fields this HTML generator needs.
        if "screening_kw" not in str(exc):
            raise
        monitor = TrainingDataPerformance([path])
        from ML.quality import screen_trials
        monitor.screening = screen_trials(
            monitor.X, monitor.labels, fs=monitor.fs, verbose=False,
            **screening_kw)
        monitor.keep = np.asarray(monitor.screening["keep"], dtype=bool)
        monitor.drop_reasons = [
            [name for name, mask in monitor.screening["reasons"].items()
             if bool(mask[index])]
            for index in range(len(monitor.X))]
        monitor.retained_counts = {
            name: int(((monitor.labels == name) & monitor.keep).sum())
            for name in monitor.class_names}

    report = monitor.report()
    if "channel_labels" not in report:
        with np.load(path, allow_pickle=True) as data:
            saved = (np.asarray(data["channel_labels"]).astype(str).tolist()
                     if "channel_labels" in data else [])
        if saved:
            report["channel_labels"] = saved
        else:
            try:
                from ML.montage import get
                report["channel_labels"] = list(
                    get(report["montage"]).get("labels", []))
            except Exception:
                report["channel_labels"] = [
                    f"CH{index + 1}" for index in range(report["shape"][1])]
    if "drop_reason_counts" not in report:
        report["drop_reason_counts"] = {
            name: int(np.asarray(mask, dtype=bool).sum())
            for name, mask in monitor.screening["reasons"].items()}
    report.setdefault("screening_thresholds", dict(screening_kw))
    if "channel_diagnostics" not in report:
        centred = monitor.X - monitor.X.mean(axis=2, keepdims=True)
        peaks = np.max(np.abs(centred), axis=2)
        stds = np.std(monitor.X, axis=2)
        jumps = np.max(np.abs(np.diff(monitor.X, axis=2)), axis=2)
        report["channel_diagnostics"] = []
        for channel, label in enumerate(report["channel_labels"]):
            peak, std, jump = peaks[:, channel], stds[:, channel], jumps[:, channel]
            report["channel_diagnostics"].append({
                "index": channel, "label": label,
                "peak_median_uv": float(np.median(peak)),
                "peak_p95_uv": float(np.percentile(peak, 95)),
                "peak_max_uv": float(np.max(peak)),
                "amplitude_fail_trials": int((peak > screening_kw["amp_uv"]).sum()),
                "std_median_uv": float(np.median(std)),
                "std_min_uv": float(np.min(std)),
                "std_max_uv": float(np.max(std)),
                "flat_fail_trials": int((std < screening_kw["flat_uv"]).sum()),
                "jump_median_uv": float(np.median(jump)),
                "jump_p95_uv": float(np.percentile(jump, 95)),
                "jump_max_uv": float(np.max(jump)),
                "jump_fail_trials": int((jump > screening_kw["jump_uv"]).sum()),
            })
        report["trial_signal_metrics"] = {
            "peak_median_uv": float(np.median(np.max(peaks, axis=1))),
            "peak_max_uv": float(np.max(peaks)),
            "jump_median_uv": float(np.median(np.max(jumps, axis=1))),
            "jump_max_uv": float(np.max(jumps)),
        }
    if labels_override:
        if len(labels_override) != report["shape"][1]:
            raise ValueError("--channel-labels must contain exactly 8 labels")
        report["channel_labels"] = list(labels_override)
        for diagnostic, label in zip(report["channel_diagnostics"], labels_override):
            diagnostic["label"] = label
    assessment = _assess(report, min_per_class=min_per_class)
    representative_index = int(np.flatnonzero(monitor.keep)[0]) \
        if monitor.keep.any() else 0
    trial = monitor.trial(representative_index)
    return monitor, report, assessment, trial


def build_html(results, output):
    cards = []
    for path, monitor, report, assessment, trial in results:
        reason_text = ", ".join(
            f"{name}={count}" for name, count
            in report["drop_reason_counts"].items() if count) or "none"
        missing = assessment["missing_universal_labels"]
        missing_text = (", ".join(value.replace("intent::", "") for value in missing)
                        if missing else "none")
        cards.append(f"""
<section class="card">
  <h2>{html.escape(path.name)}</h2>
  <p class="verdict {assessment['level']}">{html.escape(assessment['verdict'])}</p>
  <div class="grid">
    <div><b>Shape</b><br>{html.escape(str(report['shape']))}</div>
    <div><b>Trials</b><br>{report['trials']} (expected universal: 119)</div>
    <div><b>Kept / dropped</b><br>{report['kept']} / {report['dropped']} ({report['drop_rate']:.1%})</div>
    <div><b>Subject</b><br>{html.escape(', '.join(report['subjects']))}</div>
    <div><b>Montage</b><br>{html.escape(report['montage'])}</div>
    <div><b>Channels</b><br>{html.escape(', '.join(report['channel_labels']))}</div>
  </div>
  <p><b>Drop reasons:</b> {html.escape(reason_text)}</p>
  <p><b>Missing universal labels:</b> {html.escape(missing_text)}</p>
  <table><thead><tr><th>Class</th><th>Retained/recorded</th><th>Recorded count</th></tr></thead>
  <tbody>{_bar_rows(report['class_counts'], report['retained_counts'])}</tbody></table>
  <h3>Per-channel rejection attribution</h3>
  <p>Peak threshold {report['screening_thresholds'].get('amp_uv', 400)} µV · flat threshold {report['screening_thresholds'].get('flat_uv', .05)} µV SD · jump threshold {report['screening_thresholds'].get('jump_uv', 75)} µV/sample</p>
  <table><thead><tr><th>#</th><th>Channel</th><th>Peak median / p95 / max µV</th><th>Amp fails</th><th>SD median / min µV</th><th>Flat fails</th><th>Jump median / p95 / max µV</th><th>Jump fails</th></tr></thead>
  <tbody>{_diagnostic_rows(report['channel_diagnostics'])}</tbody></table>
  <h3>Representative {'retained' if report['kept'] else 'rejected'} epoch</h3>
  {_trace_svg(trial['window'], report['channel_labels'])}
</section>""")

    payload = [{"file": str(path), **report, **assessment}
               for path, _, report, assessment, _ in results]
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>BCI Training Data Inspection</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#071018;color:#dce8eb;margin:24px}}
h1{{margin-bottom:4px}} .note{{color:#efaa8b;max-width:1000px}}
.card{{background:#0d1e29;border:1px solid #31505d;border-radius:14px;padding:18px;margin:18px 0}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:10px}}
.grid div{{background:#122b38;border-radius:8px;padding:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} th,td{{padding:6px;border-bottom:1px solid #294653;text-align:left}}
.bar{{height:12px;background:#47d7d0;border-radius:6px;min-width:2px;max-width:100%}}
.verdict{{font-weight:700;padding:9px;border-radius:7px}} .good{{background:#174b3a}} .warn{{background:#57461d}} .bad{{background:#5c2929}}
svg{{width:100%;background:#081720;border-radius:8px}}
</style></head><body>
<h1>BCI Training Data Inspection</h1>
<p class="note">This report checks file completeness, generic signal screening and montage suitability. It cannot prove that the participant performed MI or that classes generalize. Use held-session/held-subject evaluation for that claim.</p>
{''.join(cards)}
<script type="application/json" id="report-data">{html.escape(json.dumps(payload))}</script>
</body></html>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output, json_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NPZ sessions and assess training readiness")
    parser.add_argument("--input", nargs="+",
                        default=[str(ROOT / "dataset" / "motor_intent" / "calib_*.npz")])
    parser.add_argument("--out", default=str(ROOT / "reports" / "training_data_report.html"))
    parser.add_argument("--channel-labels", default=None,
                        help="visual/report override, comma-separated; does not edit NPZ")
    parser.add_argument("--min-per-class", type=int, default=8)
    parser.add_argument("--amp-uv", type=float, default=400.0)
    parser.add_argument("--jump-uv", type=float, default=75.0)
    parser.add_argument("--flat-uv", type=float, default=.05)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--open", action="store_true", help="open HTML in browser")
    args = parser.parse_args()

    files = _expand(args.input)
    if not files:
        print("No NPZ files matched.")
        return 1
    labels_override = ([part.strip() for part in args.channel_labels.split(",")]
                       if args.channel_labels else None)
    screening_kw = {"amp_uv": args.amp_uv, "jump_uv": args.jump_uv,
                    "flat_uv": args.flat_uv, "z_max": args.z_max}
    results = []
    print(f"Inspecting {len(files)} recording(s)\n")
    for path in files:
        try:
            monitor, report, assessment, trial = inspect_file(
                path, screening_kw, labels_override=labels_override,
                min_per_class=args.min_per_class)
            results.append((path, monitor, report, assessment, trial))
            reasons = {k: v for k, v in report["drop_reason_counts"].items() if v}
            print(f"{path.name}")
            print(f"  trials={report['trials']} kept={report['kept']} "
                  f"dropped={report['dropped']} reasons={reasons or 'none'}")
            print(f"  montage={report['montage']} "
                  f"channels={','.join(report['channel_labels'])}")
            amp_bad = [f"{row['label']}:{row['amplitude_fail_trials']}"
                       for row in report["channel_diagnostics"]
                       if row["amplitude_fail_trials"]]
            flat_bad = [f"{row['label']}:{row['flat_fail_trials']}"
                        for row in report["channel_diagnostics"]
                        if row["flat_fail_trials"]]
            jump_bad = [f"{row['label']}:{row['jump_fail_trials']}"
                        for row in report["channel_diagnostics"]
                        if row["jump_fail_trials"]]
            print(f"  amplitude offenders={amp_bad or 'none'}")
            print(f"  flat offenders={flat_bad or 'none'}")
            print(f"  jump offenders={jump_bad or 'none'}")
            print(f"  {assessment['verdict']}\n")
        except Exception as exc:
            print(f"{path}: ERROR {type(exc).__name__}: {exc}")
    if not results:
        return 1
    html_path, json_path = build_html(results, args.out)
    print(f"HTML report -> {html_path}")
    print(f"JSON report -> {json_path}")
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
