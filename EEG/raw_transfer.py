"""Raw acquisition evidence utilities used before any EEG filtering/model.

The functions in this module intentionally do not band-pass, centre, resample,
clip, smooth, or otherwise modify samples.  They preserve physical microvolt
values and attach backend transport diagnostics so a flat or railed source is
localized before a classifier can hide it.
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from EEG.signal_preflight import analyse_preflight


def collect_raw_transfer(backend, seconds=10.0, *, already_started=False,
                         poll_interval=.005):
    """Capture channel-major raw samples and backend transport diagnostics."""
    duration = max(2.0, float(seconds))
    started_here = not already_started
    if started_here:
        backend.start()
    chunks = []
    try:
        # Drop only samples buffered while the connection/profile was opening.
        # This creates a well-defined capture interval without filtering data.
        backend.read_available()
        if hasattr(backend, "reset_raw_diagnostics"):
            backend.reset_raw_diagnostics()
        capture_started = time.monotonic()
        deadline = capture_started + duration
        while time.monotonic() < deadline:
            chunk = backend.read_available()
            if chunk is not None and np.asarray(chunk).size:
                chunk = np.asarray(chunk, dtype=np.float64)
                if chunk.ndim != 2:
                    raise ValueError(
                        f"backend returned {chunk.shape}; expected channels x samples")
                chunks.append(chunk)
            time.sleep(max(0.0, float(poll_interval)))
        elapsed = time.monotonic() - capture_started
    finally:
        if started_here:
            backend.stop()
    info = dict(backend.info())
    n_channels = int(info["n_channels"])
    data = (np.concatenate(chunks, axis=1) if chunks else
            np.empty((n_channels, 0), dtype=np.float64))
    if data.shape[0] != n_channels:
        raise ValueError(
            f"captured {data.shape[0]} channels; backend declares {n_channels}")
    diagnostics = (backend.raw_diagnostics()
                   if hasattr(backend, "raw_diagnostics") else {})
    return data, float(info["fs"]), elapsed, info, diagnostics


def _longest_constant_run(values):
    values = np.asarray(values)
    if values.size == 0:
        return 0
    longest = current = 1
    for index in range(1, values.size):
        if values[index] == values[index - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return int(longest)


def analyse_raw_transfer(data, fs, elapsed, *, source_info=None,
                         transport=None, channel_labels=None):
    """Summarize units, range, variance, rails, rate and packet continuity."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected channels x samples, got {x.shape}")
    source_info = dict(source_info or {})
    labels = list(channel_labels or source_info.get("channel_labels") or [])
    if len(labels) != x.shape[0]:
        labels = [f"CH{index + 1}" for index in range(x.shape[0])]
    preflight = analyse_preflight(
        x, fs=fs, elapsed=elapsed, channel_labels=labels)
    channel_rows = []
    for index, row in enumerate(x):
        finite = row[np.isfinite(row)]
        if finite.size:
            differences = np.diff(finite)
            unique = np.unique(finite)
            zero_delta_fraction = (float(np.mean(differences == 0.0))
                                   if differences.size else 1.0)
            metrics = {
                "min_uv": float(np.min(finite)),
                "max_uv": float(np.max(finite)),
                "span_uv": float(np.ptp(finite)),
                "mean_uv": float(np.mean(finite)),
                "median_uv": float(np.median(finite)),
                "std_uv": float(np.std(finite)),
                "mad_uv": float(np.median(
                    np.abs(finite - np.median(finite)))),
                "unique_values": int(unique.size),
                "zero_delta_fraction": zero_delta_fraction,
                "longest_identical_run": _longest_constant_run(finite),
            }
        else:
            metrics = {
                "min_uv": None, "max_uv": None, "span_uv": None,
                "mean_uv": None, "median_uv": None, "std_uv": None,
                "mad_uv": None, "unique_values": 0,
                "zero_delta_fraction": None, "longest_identical_run": 0,
            }
        quality = preflight["channels"][index]
        channel_rows.append({
            "index": index,
            "board_row": ((source_info.get("eeg_channel_rows") or [None] * x.shape[0])[index]
                          if len(source_info.get("eeg_channel_rows") or []) == x.shape[0]
                          else None),
            "label": labels[index],
            **metrics,
            "quality_ok": quality["ok"],
            "quality_reasons": list(quality["reasons"]),
            "clip_fraction": quality["clip_fraction"],
        })

    covariance = None
    if x.shape[1] >= 2 and np.isfinite(x).all():
        centered = x - np.mean(x, axis=1, keepdims=True)
        covariance = centered @ centered.T / max(1, x.shape[1] - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        covariance_summary = {
            "rank": int(np.linalg.matrix_rank(covariance)),
            "eigenvalues_uv2": eigenvalues.tolist(),
            "condition_number": (
                float(np.max(eigenvalues) / max(np.min(eigenvalues), 1e-30))
                if eigenvalues.size else None),
            "diagonal_variance_uv2": np.diag(covariance).tolist(),
        }
    else:
        covariance_summary = {
            "rank": 0, "eigenvalues_uv2": [], "condition_number": None,
            "diagonal_variance_uv2": [],
        }

    transport = dict(transport or {})
    continuity = {
        "exact": bool(transport.get("packet_continuity_exact", False)),
        "missing_packets_estimate": transport.get("missing_packets_estimate"),
        "duplicate_packages": transport.get("duplicate_packages"),
        "out_of_order_packages": transport.get("out_of_order_packages"),
        "nonmonotonic_timestamps": transport.get("nonmonotonic_timestamps"),
        "note": transport.get("packet_continuity_note"),
    }
    return {
        "kind": "CytonRawTransferEvidence",
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "raw_only": True,
        "filtering_applied": False,
        "units": source_info.get("units", "uV"),
        "shape": [int(value) for value in x.shape],
        "fs": float(fs),
        "elapsed_s": float(elapsed),
        "effective_fs": float(x.shape[1] / max(float(elapsed), 1e-9)),
        "source": source_info,
        "transport": transport,
        "packet_continuity": continuity,
        "preflight": preflight,
        "channels": channel_rows,
        "covariance_summary": covariance_summary,
        "decision": "RAW_TRANSFER_PASS" if preflight["acceptable"] else
                    "RAW_TRANSFER_FAIL",
        "limitations": [
            "A raw-transfer pass proves sample flow and gross signal variation, not EEG origin or motor-imagery accuracy.",
            "OpenBCI GUI UDP JSON has no packet sequence number; its exact UDP loss cannot be reconstructed.",
        ],
    }


def save_raw_transfer(data, report, out_base):
    """Write NPZ evidence plus JSON and a self-contained HTML summary."""
    base = Path(out_base)
    if base.suffix.lower() in {".npz", ".json", ".html"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    html_path = base.with_suffix(".html")
    labels = [row["label"] for row in report["channels"]]
    np.savez_compressed(
        npz_path,
        data=np.asarray(data, dtype=np.float32),
        fs=np.asarray(float(report["fs"])),
        channel_labels=np.asarray(labels),
        units=np.asarray(report["units"]),
        report_json=np.asarray(json.dumps(report, sort_keys=True)),
    )
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    channel_html = "".join(
        "<tr>"
        f"<td>{row['index'] + 1}</td><td>{html.escape(str(row['board_row']))}</td>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td>{row['min_uv'] if row['min_uv'] is not None else 'n/a'}</td>"
        f"<td>{row['max_uv'] if row['max_uv'] is not None else 'n/a'}</td>"
        f"<td>{row['std_uv'] if row['std_uv'] is not None else 'n/a'}</td>"
        f"<td>{row['unique_values']}</td>"
        f"<td>{row['zero_delta_fraction'] if row['zero_delta_fraction'] is not None else 'n/a'}</td>"
        f"<td>{html.escape(', '.join(row['quality_reasons']) or 'PASS')}</td>"
        "</tr>" for row in report["channels"])
    packet = report["packet_continuity"]
    source = report["source"]
    status_class = "good" if report["decision"] == "RAW_TRANSFER_PASS" else "bad"
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cyton raw transfer evidence</title><style>
body{{font-family:Segoe UI,Arial;background:#071018;color:#dce8eb;margin:24px}}
.card{{background:#0d1e29;border:1px solid #31505d;border-radius:12px;padding:15px;margin:13px 0}}
.good{{color:#7ddb9b}}.bad{{color:#ff806f}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:7px;border-bottom:1px solid #294653;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style></head><body><h1>Cyton raw transfer evidence</h1>
<h2 class="{status_class}">{html.escape(report['decision'])}</h2>
<section class="card"><h2>Acquisition contract</h2>
<p>Raw only: yes · filtering: none · units: {html.escape(report['units'])} · shape: {report['shape']} · nominal/effective rate: {report['fs']:.3g}/{report['effective_fs']:.3g} Hz</p>
<pre>{html.escape(json.dumps(source, indent=2))}</pre></section>
<section class="card"><h2>Packet continuity</h2>
<pre>{html.escape(json.dumps(packet, indent=2))}</pre></section>
<section class="card"><h2>Per-channel physical values</h2><table><thead><tr>
<th>CH</th><th>board row</th><th>label</th><th>min µV</th><th>max µV</th><th>SD µV</th><th>unique</th><th>zero Δ</th><th>quality</th>
</tr></thead><tbody>{channel_html}</tbody></table></section>
<section class="card"><h2>Complete transport metadata</h2><pre>{html.escape(json.dumps(report['transport'], indent=2))}</pre></section>
<p>No filter, smoothing, clipping, classifier, or amplitude normalization was used.</p>
</body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return npz_path, json_path, html_path


def load_raw_transfer(path):
    """Load an evidence NPZ created by :func:`save_raw_transfer`."""
    with np.load(path, allow_pickle=False) as saved:
        report = json.loads(str(np.asarray(saved["report_json"]).item()))
        return {
            "path": str(Path(path)),
            "data": np.asarray(saved["data"], dtype=np.float64),
            "fs": float(np.asarray(saved["fs"]).item()),
            "channel_labels": np.asarray(saved["channel_labels"]).astype(str).tolist(),
            "units": str(np.asarray(saved["units"]).item()),
            "report": report,
        }


def compare_raw_transfers(first, second):
    """Compare sequential captures without pretending they are time-aligned."""
    a = load_raw_transfer(first) if not isinstance(first, dict) else first
    b = load_raw_transfer(second) if not isinstance(second, dict) else second
    if a["units"] != "uV" or b["units"] != "uV":
        raise ValueError("raw-transfer comparison requires both captures in uV")
    if a["data"].shape[0] != b["data"].shape[0]:
        raise ValueError("captures have different channel counts")
    rows = []
    for index in range(a["data"].shape[0]):
        std_a = float(np.std(a["data"][index]))
        std_b = float(np.std(b["data"][index]))
        rows.append({
            "index": index,
            "first_label": a["channel_labels"][index],
            "second_label": b["channel_labels"][index],
            "first_std_uv": std_a,
            "second_std_uv": std_b,
            "second_to_first_std_ratio": std_b / max(std_a, 1e-30),
            "first_span_uv": float(np.ptp(a["data"][index])),
            "second_span_uv": float(np.ptp(b["data"][index])),
        })
    return {
        "kind": "CytonRawTransferComparison",
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "first": {"path": a.get("path"), "source": a["report"].get("source"),
                  "decision": a["report"].get("decision")},
        "second": {"path": b.get("path"), "source": b["report"].get("source"),
                   "decision": b["report"].get("decision")},
        "channels": rows,
        "interpretation": (
            "Captures are sequential, not sample-aligned. Compare scale, variance, "
            "rails, units, channel order and rate; do not compute waveform agreement."),
    }
