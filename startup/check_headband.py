"""
Author: Anson
Created on: 18/11/2026
Purpose: mandatory pre-use/pre-record Cyton headset startup gate.

Standalone use:
    python startup/check_headband.py --port COM8 --montage cyton8_motor \
      --confirm-srb-bias --out reports/startup/headband_check

The software can reset Cyton channels to their default Normal input and prove
that every channel transfers known internal-ground and square-wave signals. It
cannot see a physical connector, so SRB/reference and BIAS attachment requires
an explicit user confirmation. A final normal-electrode capture then checks
that all eight channels are finite, non-flat and free of gross peaks/jumps.
"""

import argparse
import copy
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stage(name, status, summary, details=None):
    return {"name": name, "status": status, "summary": summary,
            "details": details or {}}


def evaluate_hardware_stages(hardware_stages, physical_confirmed):
    """Convert unified hardware stages into strict all-eight-channel gates.

    ``tools.debug_bci_pipeline.hardware_layers`` accepts seven square-wave
    channels for general diagnosis. Recording startup is intentionally stricter:
    every one of the eight channels must transfer both known signals, followed
    by an acceptable normal-input baseline.
    """
    source = {stage.get("name"): copy.deepcopy(stage)
              for stage in hardware_stages}
    stages = [_stage(
        "S0 physical SRB/reference and BIAS confirmation",
        "PASS" if physical_confirmed else "FAIL",
        ("user confirmed SRB/reference and BIAS are firmly attached"
         if physical_confirmed else
         "physical connectors were not confirmed; software cannot inspect them"),
        {"manual_confirmation": bool(physical_confirmed)})]

    ground = source.get("L0A known internal ground")
    if ground is None:
        stages.append(_stage(
            "S1 all-channel internal-ground transfer", "FAIL",
            "internal-ground test did not complete"))
    else:
        rms = np.asarray(ground.get("details", {}).get("rms_uv", []), dtype=float)
        passed = bool(
            ground.get("status") == "PASS" and len(rms) == 8 and
            np.isfinite(rms).all() and np.all(rms < 100.0))
        stages.append(_stage(
            "S1 all-channel internal-ground transfer",
            "PASS" if passed else "FAIL",
            ("8/8 channels transferred a finite low-noise internal ground"
             if passed else
             "all 8 channels must transfer internal ground below 100 µV RMS"),
            ground.get("details", {})))

    square = source.get("L0B known square-wave transfer")
    if square is None:
        stages.append(_stage(
            "S2 all-channel square-wave transfer", "FAIL",
            "known square-wave test did not complete"))
    else:
        channels = square.get("details", {}).get("channels", [])
        matched = sum(
            row.get("kind") == "square" and
            row.get("fundamental") is not None and
            abs(float(row["fundamental"]) - 2.0) <= .6
            for row in channels)
        passed = bool(square.get("status") == "PASS" and
                      len(channels) == 8 and matched == 8)
        stages.append(_stage(
            "S2 all-channel square-wave transfer",
            "PASS" if passed else "FAIL",
            ("8/8 channels transferred the known approximately 2-Hz square wave"
             if passed else f"only {matched}/8 channels transferred the known square wave"),
            {**square.get("details", {}), "matched_channels": matched,
             "required_channels": 8}))

    normal = source.get("L1 live normal-electrode signal")
    if normal is None:
        stages.append(_stage(
            "S3 default Normal input and live electrodes", "FAIL",
            "normal-electrode test did not complete after Cyton default reset"))
    else:
        rows = normal.get("details", {}).get("channels", [])
        passed = bool(normal.get("status") == "PASS" and len(rows) == 8 and
                      all(row.get("ok") for row in rows))
        stages.append(_stage(
            "S3 default Normal input and live electrodes",
            "PASS" if passed else "FAIL",
            ("Cyton default reset completed and 8/8 normal-input channels passed"
             if passed else
             "one or more normal-input channels are flat, excessive, discontinuous, or missing"),
            normal.get("details", {})))

    return stages


def save_startup_report(stages, out_base, port, montage):
    base = Path(out_base)
    if base.suffix.lower() in (".html", ".json"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    passed = bool(stages and all(stage["status"] == "PASS" for stage in stages))
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "port": port, "montage": montage, "passed": passed,
        "decision": "READY" if passed else "BLOCKED",
        "stages": stages,
        "limitations": [
            "Physical SRB/reference and BIAS attachment is user-confirmed, not machine-visible.",
            "A startup PASS proves transfer and gross raw quality, not motor-imagery accuracy.",
        ],
    }
    json_path, html_path = base.with_suffix(".json"), base.with_suffix(".html")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cards = []
    for stage in stages:
        css = "good" if stage["status"] == "PASS" else "bad"
        details = html.escape(json.dumps(stage["details"], indent=2))
        cards.append(
            f'<section class="card"><h2>{html.escape(stage["name"])}</h2>'
            f'<p class="{css}">{stage["status"]}: '
            f'{html.escape(stage["summary"])}</p>'
            f'<details><summary>measurements</summary><pre>{details}</pre></details>'
            f'</section>')
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cyton headset startup check</title><style>
body{{font-family:Segoe UI,Arial;background:#071018;color:#dce8eb;margin:24px}}
.card{{background:#0d1e29;border:1px solid #31505d;border-radius:12px;padding:15px;margin:13px 0}}
p{{padding:9px;border-radius:7px}}.good{{background:#174b3a}}.bad{{background:#5c2929}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.decision{{font-size:22px;font-weight:700}}
</style></head><body><h1>Cyton Headset Startup Check</h1>
<p class="decision">{'READY — recording/use allowed' if passed else 'BLOCKED — do not record/use'}</p>
<p>Port {html.escape(str(port))} · montage {html.escape(str(montage))}</p>
{''.join(cards)}
<p>Physical connectors cannot be inspected by software. A PASS is a transfer/raw-quality gate, not biological MI validation.</p>
</body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return html_path, json_path, payload


def run_headband_check(port, montage="cyton8_motor", seconds=6.0,
                       physical_confirmed=False, out_base=None):
    """Run mandatory Cyton startup stages and save one HTML/JSON report."""
    if not port:
        raise ValueError("direct Cyton startup check requires --port COM8")
    from tools.debug_bci_pipeline import hardware_layers

    print("\nCYTON HEADSET STARTUP GATE")
    print("  Reset/test sequence: internal ground -> square wave -> default Normal input")
    hardware = hardware_layers(port, montage, seconds=max(2.0, float(seconds)))
    stages = evaluate_hardware_stages(hardware, physical_confirmed)
    if out_base is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_base = ROOT / "reports" / "startup" / f"headband_{stamp}"
    html_path, json_path, payload = save_startup_report(
        stages, out_base, port, montage)
    print("\nSTARTUP CHECK SUMMARY")
    print("-" * 86)
    for stage in stages:
        print(f"  [{stage['status']:<4}] {stage['name']}: {stage['summary']}")
    print(f"\n  DECISION: {payload['decision']}")
    print(f"  HTML -> {html_path}")
    print(f"  JSON -> {json_path}")
    return payload


def _confirm_physical_connections():
    print("\nPHYSICAL CHECK (software cannot inspect these connectors)")
    print("  1. SRB/reference electrode is firmly attached.")
    print("  2. BIAS electrode is firmly attached.")
    print("  3. All eight electrode leads are connected to the intended nodes.")
    try:
        answer = input("Type YES to confirm these three checks: ").strip().upper()
    except EOFError:
        return False
    return answer == "YES"


def main():
    parser = argparse.ArgumentParser(
        description="Mandatory pre-use/pre-record Cyton headset startup gate")
    parser.add_argument("--port", required=True, help="Cyton dongle COM port")
    parser.add_argument("--montage", default="cyton8_motor")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--confirm-srb-bias", action="store_true",
                        help="confirm SRB/reference, BIAS and all eight leads are attached")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    confirmed = args.confirm_srb_bias or _confirm_physical_connections()
    payload = run_headband_check(
        args.port, montage=args.montage, seconds=args.seconds,
        physical_confirmed=confirmed, out_base=args.out)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
