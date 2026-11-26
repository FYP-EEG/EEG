"""Capture/compare raw Cyton transfer before filtering or classification.

Direct route (close OpenBCI GUI; Python owns COM):

    python tools/cyton_transfer_check.py capture --route direct --port COM8 \
      --profile thinkpulse_active --gain 4 --seconds 12

GUI parity route (GUI owns COM; set Networking to UDP / Time Series Raw /
127.0.0.1 / 12345, then start its network stream):

    python tools/cyton_transfer_check.py capture --route gui-udp \
      --udp-ip 127.0.0.1 --udp-port 12345 --seconds 12

Compare two sequential captures made with unchanged wiring:

    python tools/cyton_transfer_check.py compare \
      reports/cyton_transfer/direct.npz reports/cyton_transfer/gui_udp.npz

Every capture is raw microvolts.  This tool deliberately applies no filters,
smoothing, centring, clipping, resampling, or model.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from EEG.cyton_commands import CYTON_PROFILE_NAMES, cyton_profile_settings
from EEG.raw_transfer import (
    analyse_raw_transfer,
    collect_raw_transfer,
    compare_raw_transfers,
    save_raw_transfer,
)
from ML.montage import get as get_montage


def _default_base(route):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "reports" / "cyton_transfer" / f"{route}_{stamp}"


def _capture(args):
    labels = list(get_montage(args.montage).get("labels", []))
    if args.route == "direct":
        if not args.port:
            raise SystemExit("direct route requires --port COM8")
        from EEG.stream_bridge import BrainFlowBackend
        settings = cyton_profile_settings(args.profile, gain=args.gain)
        backend = BrainFlowBackend(
            board_id=0, serial_port=args.port, n_channels=8,
            cyton_profile=settings["name"], cyton_gain=settings["gain"],
            electrode_mode=settings["electrode_mode"],
            reset_channels=not args.no_reset)
        print("DIRECT ROUTE: close OpenBCI GUI and every other program using "
              f"{args.port}. Applying profile {settings['name']} at gain "
              f"x{settings['gain']}.")
    else:
        from EEG.stream_bridge import OpenBCIGUIUDPBackend
        backend = OpenBCIGUIUDPBackend(
            bind_ip=args.udp_ip, port=args.udp_port, fs=args.fs,
            n_channels=8, startup_timeout=args.startup_timeout)
        print("GUI UDP ROUTE: OpenBCI GUI must own the Cyton COM port.")
        print("  Networking settings: UDP · Time Series Raw · "
              f"{args.udp_ip}:{args.udp_port}")
        print("  Start both the GUI data stream and Networking stream. Do not "
              "select Time Series Filtered.")

    print(f"Capturing {args.seconds:g}s of unfiltered physical microvolts...")
    data, fs, elapsed, info, transport = collect_raw_transfer(
        backend, seconds=args.seconds)
    report = analyse_raw_transfer(
        data, fs, elapsed, source_info=info, transport=transport,
        channel_labels=labels)
    out = Path(args.out) if args.out else _default_base(args.route)
    npz_path, json_path, html_path = save_raw_transfer(data, report, out)

    print("\nRAW TRANSFER SUMMARY")
    print(f"  decision       : {report['decision']}")
    print(f"  shape          : {report['shape']}")
    print(f"  units          : {report['units']}")
    print(f"  nominal/effective fs: {report['fs']:.1f} / "
          f"{report['effective_fs']:.1f} Hz")
    print(f"  EEG board rows : {info.get('eeg_channel_rows')}")
    for row in report["channels"]:
        status = "PASS" if row["quality_ok"] else ",".join(row["quality_reasons"])
        print(f"  CH{row['index'] + 1} {row['label']:<3} row={row['board_row']!s:<4} "
              f"std={row['std_uv']!s:<12} span={row['span_uv']!s:<12} "
              f"unique={row['unique_values']:<6} {status}")
    continuity = report["packet_continuity"]
    print("  packet continuity: " + json.dumps(continuity, sort_keys=True))
    print(f"\n  NPZ  -> {npz_path}\n  JSON -> {json_path}\n  HTML -> {html_path}")
    return 0 if report["decision"] == "RAW_TRANSFER_PASS" else 2


def _write_comparison(payload, out_base):
    base = Path(out_base)
    if base.suffix.lower() in {".json", ".html"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    html_path = base.with_suffix(".html")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = "".join(
        "<tr>"
        f"<td>{row['index'] + 1}</td>"
        f"<td>{html.escape(row['first_label'])}</td>"
        f"<td>{row['first_std_uv']:.6g}</td>"
        f"<td>{row['second_std_uv']:.6g}</td>"
        f"<td>{row['second_to_first_std_ratio']:.6g}</td>"
        f"<td>{row['first_span_uv']:.6g}</td>"
        f"<td>{row['second_span_uv']:.6g}</td>"
        "</tr>" for row in payload["channels"])
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cyton raw-route comparison</title><style>
body{{font-family:Segoe UI,Arial;background:#071018;color:#dce8eb;margin:24px}}
.card{{background:#0d1e29;border:1px solid #31505d;border-radius:12px;padding:15px;margin:13px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border-bottom:1px solid #294653;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head><body>
<h1>Cyton raw-route comparison</h1>
<section class="card"><h2>First</h2><pre>{html.escape(json.dumps(payload['first'], indent=2))}</pre></section>
<section class="card"><h2>Second</h2><pre>{html.escape(json.dumps(payload['second'], indent=2))}</pre></section>
<section class="card"><h2>Physical-scale comparison</h2><table><thead><tr>
<th>CH</th><th>label</th><th>first SD µV</th><th>second SD µV</th><th>SD ratio</th><th>first span µV</th><th>second span µV</th>
</tr></thead><tbody>{rows}</tbody></table></section>
<p>{html.escape(payload['interpretation'])}</p></body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return json_path, html_path


def _compare(args):
    payload = compare_raw_transfers(args.first, args.second)
    out = (Path(args.out) if args.out else
           ROOT / "reports" / "cyton_transfer" /
           f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    json_path, html_path = _write_comparison(payload, out)
    print("RAW ROUTE COMPARISON")
    for row in payload["channels"]:
        print(f"  CH{row['index'] + 1}: first SD={row['first_std_uv']:.6g} µV · "
              f"second SD={row['second_std_uv']:.6g} µV · "
              f"ratio={row['second_to_first_std_ratio']:.4g}")
    print(f"\n  JSON -> {json_path}\n  HTML -> {html_path}")
    return 0


def make_parser():
    parser = argparse.ArgumentParser(
        description="Raw Cyton direct-vs-GUI transfer capture and comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="capture one raw acquisition route")
    capture.add_argument("--route", choices=("direct", "gui-udp"), required=True)
    capture.add_argument("--port", default=None, help="Cyton COM port for direct route")
    capture.add_argument("--profile", choices=CYTON_PROFILE_NAMES,
                         default="thinkpulse_active")
    capture.add_argument("--gain", type=int, choices=(1, 2, 4, 6, 8, 12, 24),
                         default=None,
                         help="gain override; ThinkPulse profile permits x8 or lower")
    capture.add_argument("--no-reset", action="store_true",
                         help="do not reset/reapply direct Cyton channels; diagnostic only")
    capture.add_argument("--udp-ip", default="127.0.0.1")
    capture.add_argument("--udp-port", type=int, default=12345)
    capture.add_argument("--fs", type=float, default=250.0)
    capture.add_argument("--startup-timeout", type=float, default=15.0)
    capture.add_argument("--seconds", type=float, default=12.0)
    capture.add_argument("--montage", default="cyton8_motor")
    capture.add_argument("--out", default=None,
                         help="output base; writes .npz, .json and .html")

    compare = sub.add_parser("compare", help="compare two evidence NPZ files")
    compare.add_argument("first")
    compare.add_argument("second")
    compare.add_argument("--out", default=None)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        return _capture(args) if args.command == "capture" else _compare(args)
    except KeyboardInterrupt:
        print("\nAborted by operator.")
        return 130
    except Exception as exc:  # noqa: BLE001 - expose hardware root in CLI
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        if getattr(args, "command", None) == "capture":
            if getattr(args, "route", None) == "direct":
                print("Close OpenBCI GUI, verify the COM port, Cyton power/radio, "
                      "SRB2 reference, sensor power, and selected gain.")
            else:
                print("In OpenBCI GUI select Networking -> UDP -> Time Series Raw, "
                      "match IP/port, and start the network stream.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
