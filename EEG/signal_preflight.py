"""
Author: Anson
Created on: 12/11/2026
Purpose: connection and raw-signal go/no-go gate before a timed EEG recording.

This module never filters or modifies samples. It checks that samples arrive at
approximately the declared rate, remain finite, are not flat, and do not contain
gross amplitude/jump faults. Reports explain likely causes and corrective steps.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLDS = {
    "amp_uv": 400.0,
    "jump_uv": 75.0,
    "flat_uv": .05,
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
        centred = x - x.mean(axis=1, keepdims=True)
        peak = np.max(np.abs(centred), axis=1)
        std = np.std(x, axis=1)
        jump = (np.max(np.abs(np.diff(x, axis=1)), axis=1)
                if n_samples > 1 else np.full(n_channels, np.inf))
    else:
        peak = std = jump = np.full(n_channels, np.nan)

    for index in range(n_channels):
        reasons = []
        if not np.isfinite([peak[index], std[index], jump[index]]).all():
            reasons.append("nonfinite")
        else:
            if peak[index] > limits["amp_uv"]:
                reasons.append("amplitude")
            if std[index] < limits["flat_uv"]:
                reasons.append("flat")
            if jump[index] > limits["jump_uv"]:
                reasons.append("jump")
        channels.append({
            "index": index, "label": labels[index],
            "peak_uv": float(peak[index]), "std_uv": float(std[index]),
            "jump_uv": float(jump[index]), "reasons": reasons,
            "ok": not reasons,
        })

    bad = [row for row in channels if not row["ok"]]
    all_large = bool(channels and all(
        row["peak_uv"] > limits["amp_uv"] and
        row["jump_uv"] > limits["jump_uv"] for row in channels))
    any_flat = any("flat" in row["reasons"] for row in channels)
    possible_test_signal = bool(all_large and np.nanmedian(std) > 500)
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
            "Flat channel(s): verify channels are powered on, Input Type is Normal, "
            "and electrodes/connectors are not shorted or disconnected.")
    if all_large:
        debug.append(
            "All channels have gross peaks/jumps: reset Cyton channels to default, "
            "disable internal test/impedance signals, and check BIAS/SRB reference.")
    elif any("amplitude" in row["reasons"] or "jump" in row["reasons"]
             for row in channels):
        debug.append(
            "Local amplitude/jump faults: inspect the listed electrode contact and "
            "cable movement in OpenBCI GUI before recording.")
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
        f"<td>{row['peak_uv']:.3f}</td><td>{row['std_uv']:.3f}</td>"
        f"<td>{row['jump_uv']:.3f}</td><td>{', '.join(row['reasons']) or 'PASS'}</td></tr>"
        for row in result["channels"])
    debug = "".join(f"<li>{line}</li>" for line in result["debug"])
    color = "#7ddb9b" if result["acceptable"] else "#ff806f"
    labels = [row["label"] for row in result["channels"]]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>EEG preflight</title>
<style>body{{font-family:Segoe UI,Arial;background:#071018;color:#dce8eb;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border-bottom:1px solid #294653;text-align:left}}svg{{width:100%;background:#0a1720;border-radius:10px}}.verdict{{font-size:22px;font-weight:700;color:{color}}}</style></head><body>
<h1>EEG Recording Preflight</h1><p class="verdict">{'PASS' if result['acceptable'] else 'FAIL — recording blocked'}</p>
<p>Shape {result['shape']} · effective rate {result['effective_fs']:.1f} Hz · nominal {result['fs']} Hz</p>
<ul>{debug}</ul><table><thead><tr><th>#</th><th>Channel</th><th>Peak µV</th><th>SD µV</th><th>Max jump µV/sample</th><th>Result</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Preflight baseline (auto-scaled per channel)</h2>{_trace_svg(data, labels)}</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return html_path, json_path


def print_preflight(result):
    print("\nEEG PREFLIGHT")
    print("-" * 78)
    print(f"  connection : {'PASS' if result['connection_ok'] else 'FAIL'}")
    print(f"  signal     : {'PASS' if result['signal_ok'] else 'FAIL'}")
    print(f"  samples    : {result['shape'][1]} at {result['effective_fs']:.1f} Hz")
    print(f"  {'ch':>2} {'node':<5} {'peak uV':>11} {'std uV':>11} "
          f"{'jump uV':>11}  result")
    for row in result["channels"]:
        status = "PASS" if row["ok"] else ",".join(row["reasons"])
        print(f"  {row['index'] + 1:>2} {row['label']:<5} "
              f"{row['peak_uv']:>11.2f} {row['std_uv']:>11.2f} "
              f"{row['jump_uv']:>11.2f}  {status}")
    print("\n  Debug guidance:")
    for line in result["debug"]:
        print(f"   - {line}")
    print("-" * 78)
