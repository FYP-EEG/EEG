"""
Author: Anson
Created on: 18/11/2026
Purpose: mandatory pre-use/pre-record Cyton headset startup gate that blocks
         use but keeps the caller alive and retries until the hardware passes.

Standalone use:
    python startup/check_headband.py --port COM8 --montage cyton8_motor \
      --out reports/startup/headband_check

The software can reset Cyton channels to their selected Normal-input profile and
prove that every channel transfers known internal-ground and square-wave signals.
It cannot see a physical connector, so it states the required SRB/reference and
BIAS wiring without demanding a typed confirmation. A final live-electrode
capture uses adaptive raw-signal checks across all eight channels.
"""

import argparse
import copy
import html
import io
import json
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from EEG.cyton_commands import normalize_electrode_mode  # noqa: E402


def _stage(name, status, summary, details=None):
    return {"name": name, "status": status, "summary": summary,
            "details": details or {}}


def evaluate_hardware_stages(hardware_stages, physical_confirmed,
                             electrode_mode="thinkpulse"):
    """Convert unified hardware stages into strict all-eight-channel gates.

    ``tools.debug_bci_pipeline.hardware_layers`` accepts seven square-wave
    channels for general diagnosis. Recording startup is intentionally stricter:
    every one of the eight channels must transfer both known signals, followed
    by an acceptable normal-input baseline.
    """
    electrode_mode = normalize_electrode_mode(electrode_mode)
    source = {stage.get("name"): copy.deepcopy(stage)
              for stage in hardware_stages}
    thinkpulse = electrode_mode == "thinkpulse"
    physical_name = (
        "S0 ThinkPulse wiring contract (not software-verifiable)"
        if thinkpulse else
        "S0 passive-electrode wiring contract (not software-verifiable)")
    physical_summary = (
        "required setup: powered ThinkPulse electrodes, one earlobe reference "
        "on SRB2, and no physical BIAS connection" if thinkpulse else
        "required setup: passive electrodes with firm SRB/reference and BIAS")
    # ``physical_confirmed`` remains in the API for old callers only. Requiring
    # a typed YES claimed more certainty than software has and blocked unattended
    # checks, so S0 now records the wiring contract rather than a manual gate.
    stages = [_stage(
        physical_name, "PASS", physical_summary,
        {"manual_confirmation_required": False,
         "legacy_confirmation_value": bool(physical_confirmed),
         "software_verified": False,
         "electrode_mode": electrode_mode,
         "bias_expected_attached": not thinkpulse})]

    ground = source.get("L0A known internal ground")
    ground_details = (ground or {}).get("details", {}) or {}
    source_error = str((ground or {}).get(
        "summary", "internal-ground connection test did not complete"))
    decode_failure = "unicodedecodeerror" in source_error.lower()
    rms = np.asarray(ground_details.get("rms_uv", []), dtype=float)
    samples = int(ground_details.get("samples", 0) or 0)
    transfer_missing = bool(
        ground is None or
        (ground.get("status") != "PASS" and
         (len(rms) != 8 or not np.isfinite(rms).all() or samples <= 0)))
    connection_failure = bool(transfer_missing and not decode_failure)
    upstream_failure = bool(connection_failure or decode_failure)
    if decode_failure:
        stages.append(_stage(
            "S1 Cyton command-response decoding", "FAIL",
            "BrainFlow received a non-UTF-8 Cyton command response; this is a "
            "driver/library response-decoding failure, not proof that the board "
            f"is disconnected: {source_error}",
            {**ground_details, "decode_failure": True,
             "source_error": source_error}))
    elif connection_failure:
        stages.append(_stage(
            "S1 Cyton connection and internal-ground transfer", "FAIL",
            "Cyton board/dongle is not connected or produced no usable EEG "
            f"samples: {source_error}",
            {**ground_details, "connection_failure": True,
             "source_error": source_error}))
    else:
        passed = bool(
            ground.get("status") == "PASS" and len(rms) == 8 and
            np.isfinite(rms).all() and np.all(rms < 100.0))
        stages.append(_stage(
            "S1 all-channel internal-ground transfer",
            "PASS" if passed else "FAIL",
            ("8/8 channels transferred a finite low-noise internal ground"
             if passed else
             "all 8 channels must transfer internal ground below 100 µV RMS"),
            ground_details))

    square = source.get("L0B known square-wave transfer")
    if upstream_failure:
        reason = ("Cyton command-response decoding failed" if decode_failure else
                  "Cyton connection/sample flow failed")
        stages.append(_stage(
            "S2 all-channel square-wave transfer", "SKIP",
            f"ignored because {reason}"))
    elif square is None:
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
    if upstream_failure:
        reason = ("Cyton command-response decoding failed" if decode_failure else
                  "Cyton connection/sample flow failed")
        stages.append(_stage(
            "S3 configured Normal input and live electrodes", "SKIP",
            f"ignored because {reason}"))
    elif normal is None:
        stages.append(_stage(
            "S3 configured Normal input and live electrodes", "FAIL",
            "live-electrode test did not complete after applying the selected "
            "Cyton channel profile"))
    else:
        rows = normal.get("details", {}).get("channels", [])
        passed = bool(normal.get("status") == "PASS" and len(rows) == 8 and
                      all(row.get("ok") for row in rows))
        stages.append(_stage(
            "S3 configured Normal input and live electrodes",
            "PASS" if passed else "FAIL",
            (f"8/8 {electrode_mode} normal-input channels passed"
             if passed else
             "one or more normal-input channels are flat, excessive, "
             "discontinuous, or missing"),
            normal.get("details", {})))

    return stages


def startup_failure_hints(stages=None, exception=None, port="COM8",
                          montage="cyton8_motor", seconds=6.0,
                          electrode_mode=None, cyton_gain=None):
    """Return concise, stage-specific fixes plus a reproducible debug command."""
    rows = list(stages or [])
    failed = [row for row in rows if row.get("status") == "FAIL"]
    mode = normalize_electrode_mode(electrode_mode or next(
        (row.get("details", {}).get("electrode_mode") for row in rows
         if row.get("details", {}).get("electrode_mode")), "thinkpulse"))
    gain = int(cyton_gain if cyton_gain is not None else next(
        (row.get("details", {}).get("cyton_gain") for row in rows
         if row.get("details", {}).get("cyton_gain") is not None),
        8 if mode == "thinkpulse" else 24))
    hints = []

    for stage in failed:
        name = str(stage.get("name", ""))
        details = stage.get("details", {}) or {}
        if name.startswith("S0"):
            if details.get("electrode_mode") == "thinkpulse":
                hints.append(
                    "S0 ThinkPulse: power the active electrodes, connect one "
                    "earlobe reference to SRB2, leave the physical BIAS lead "
                    "disconnected, and inspect all eight active-sensor leads.")
            else:
                hints.append(
                    "S0: firmly reconnect SRB/reference and BIAS, then inspect all "
                    "eight lead snaps and board-side pins.")
        elif name.startswith("S1"):
            if details.get("decode_failure"):
                hints.append(
                    "BrainFlow could not decode the Cyton command response. Upgrade "
                    "the active environment with: python -m pip install --upgrade "
                    "\"brainflow>=5.22.0\"")
                hints.append(
                    "Then unplug/replug the dongle, verify its GPIO6 switch and COM "
                    "port, power-cycle Cyton, and make sure this is the FTDI Cyton "
                    "port rather than another serial device.")
            elif details.get("connection_failure"):
                hints.append(
                    "Cyton not detected/sample flow unavailable: turn on the board, "
                    "check its battery and radio pairing, confirm the dongle COM "
                    "port, and close OpenBCI GUI/other serial programs.")
            else:
                rms = np.asarray(details.get("rms_uv", []), dtype=float)
                bad = [f"CH{index + 1}" for index, value in enumerate(rms)
                       if not np.isfinite(value) or value >= 100.0]
                suffix = f" Suspect: {', '.join(bad)}." if bad else ""
                hints.append(
                    "S1 ground: power-cycle Cyton/dongle, close other programs "
                    "using the COM port, and reseat suspect channel leads."
                    f"{suffix}")
        elif name.startswith("S2"):
            channels = details.get("channels", [])
            bad = [f"CH{int(row.get('channel', index)) + 1}"
                   for index, row in enumerate(channels)
                   if row.get("kind") != "square" or
                   row.get("fundamental") is None or
                   abs(float(row.get("fundamental") or 0.0) - 2.0) > .6]
            suffix = f" Suspect: {', '.join(bad)}." if bad else ""
            hints.append(
                "S2 square wave: verify Cyton test-signal commands reach every "
                "channel; reset channels with 'd', power-cycle, and rerun."
                f"{suffix}")
        elif name.startswith("S3"):
            channels = details.get("channels", [])
            bad = [f"{row.get('label', f'CH{index + 1}')}="
                   f"{','.join(row.get('reasons', [])) or 'failed'}"
                   for index, row in enumerate(channels) if not row.get("ok")]
            suffix = f" Suspect: {'; '.join(bad)}." if bad else ""
            if details.get("electrode_mode") == "thinkpulse":
                hints.append(
                    "S3 ThinkPulse: keep physical BIAS disconnected, confirm active "
                    "sensor power and the earlobe reference on SRB2, use Cyton gain "
                    f"8 or lower, improve contact, and keep cables/body still.{suffix}")
            else:
                hints.append(
                    "S3 live signal: send/reset channels to Normal input ('d'), "
                    "improve electrode contact, keep cables/body still, and inspect "
                    f"SRB/BIAS.{suffix}")
            for message in details.get("debug", [])[:2]:
                hints.append(f"S3 detail: {message}")

    if exception is not None:
        text = f"{type(exception).__name__}: {exception}".lower()
        if isinstance(exception, UnicodeDecodeError):
            hints.append(
                "BrainFlow received a non-UTF-8 board response. Upgrade with: "
                "python -m pip install --upgrade \"brainflow>=5.22.0\"")
            hints.append(
                "After upgrading, unplug/replug the FTDI dongle, confirm its GPIO6 "
                "switch/COM port, and power-cycle Cyton.")
        elif "brainflow" in text and ("module" in text or "import" in text):
            hints.append(
                "Install hardware support: python -m pip install -e \".[hardware]\"")
        elif any(word in text for word in ("serial", "com", "port", "session")):
            hints.append(
                "Confirm the Windows Device Manager COM number, close OpenBCI GUI "
                "or any other process holding it, then unplug/replug the dongle and "
                "power-cycle Cyton.")
        elif "no eeg samples" in text or "no samples" in text:
            hints.append(
                "The port opened but no stream arrived: check Cyton power, dongle "
                "radio pairing/channel, battery, and board mode.")
        else:
            hints.append(
                "Check Cyton power/battery, dongle radio, COM port ownership, and "
                "the terminal exception above before the automatic retry.")

    if failed or exception is not None:
        hints.append(
            "Detailed debugger: python tools/debug_bci_pipeline.py "
            f"--port {port} --montage {montage} "
            f"--electrode-mode {mode} --cyton-gain {gain} "
            f"--hardware-seconds {max(2.0, float(seconds)):g} "
            "--out reports/debug/headband")
    # Preserve order while removing duplicate advice from overlapping failures.
    return list(dict.fromkeys(hints))


def _startup_payload(stages, port, montage, seconds=6.0,
                     electrode_mode="thinkpulse", cyton_gain=None):
    electrode_mode = normalize_electrode_mode(electrode_mode)
    resolved_gain = int(cyton_gain if cyton_gain is not None else
                        (8 if electrode_mode == "thinkpulse" else 24))
    passed = bool(stages and all(stage["status"] == "PASS" for stage in stages))
    debug_hints = startup_failure_hints(
        stages, port=port, montage=montage, seconds=seconds,
        electrode_mode=electrode_mode, cyton_gain=resolved_gain)
    physical_limit = (
        "ThinkPulse power, earlobe reference on SRB2, physical BIAS "
        "disconnection and lead attachment are operator responsibilities and "
        "are not machine-visible; no typed confirmation is requested."
        if electrode_mode == "thinkpulse" else
        "Physical SRB/reference, BIAS and lead attachment are operator "
        "responsibilities, not machine-visible; no typed confirmation is requested.")
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "port": port, "montage": montage,
        "electrode_mode": electrode_mode, "cyton_gain": resolved_gain,
        "passed": passed,
        "decision": "READY" if passed else "BLOCKED",
        "stages": stages, "debug_hints": debug_hints,
        "limitations": [
            physical_limit,
            "A startup PASS proves transfer and gross raw quality, not "
            "motor-imagery accuracy.",
        ],
    }


def save_startup_report(stages, out_base, port, montage, seconds=6.0,
                        electrode_mode="thinkpulse", cyton_gain=None):
    base = Path(out_base)
    if base.suffix.lower() in (".html", ".json"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    payload = _startup_payload(
        stages, port, montage, seconds=seconds,
        electrode_mode=electrode_mode, cyton_gain=cyton_gain)
    passed = payload["passed"]
    debug_hints = payload["debug_hints"]
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
    hint_html = "".join(f"<li>{html.escape(value)}</li>" for value in debug_hints)
    hint_section = (f'<section class="card"><h2>How to fix/debug</h2>'
                    f'<ul>{hint_html}</ul></section>' if hint_html else "")
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cyton headset startup check</title><style>
body{{font-family:Segoe UI,Arial;background:#071018;color:#dce8eb;margin:24px}}
.card{{background:#0d1e29;border:1px solid #31505d;border-radius:12px;padding:15px;margin:13px 0}}
p{{padding:9px;border-radius:7px}}.good{{background:#174b3a}}.bad{{background:#5c2929}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.decision{{font-size:22px;font-weight:700}}
</style></head><body><h1>Cyton Headset Startup Check</h1>
<p class="decision">{'READY — recording/use allowed' if passed else 'BLOCKED — do not record/use'}</p>
<p>Port {html.escape(str(port))} · montage {html.escape(str(montage))} · electrode mode {html.escape(str(payload['electrode_mode']))} · Cyton gain {payload['cyton_gain']}</p>
{''.join(cards)}
{hint_section}
<p>The mode-specific physical setup cannot be inspected by software. A PASS is a transfer/raw-quality gate, not biological MI validation.</p>
</body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return html_path, json_path, payload


def run_headband_check(port, montage="cyton8_motor", seconds=6.0,
                       physical_confirmed=False, out_base=None, verbose=True,
                       save_failed=False, electrode_mode="thinkpulse",
                       cyton_gain=None):
    """Run one startup check; write reports only for READY unless requested."""
    if not port:
        raise ValueError("direct Cyton startup check requires --port COM8")
    from tools.debug_bci_pipeline import hardware_layers

    duration = max(2.0, float(seconds))
    if verbose:
        print("\nCYTON HEADSET STARTUP GATE")
        print("  Reset/test sequence: internal ground -> square wave -> "
              "selected Normal-input channel profile")
        hardware = hardware_layers(
            port, montage, seconds=duration,
            electrode_mode=electrode_mode, cyton_gain=cyton_gain)
    else:
        with redirect_stdout(io.StringIO()):
            hardware = hardware_layers(
                port, montage, seconds=duration,
                electrode_mode=electrode_mode, cyton_gain=cyton_gain)
    electrode_mode = normalize_electrode_mode(electrode_mode)
    resolved_gain = int(cyton_gain if cyton_gain is not None else
                        (8 if electrode_mode == "thinkpulse" else 24))
    stages = evaluate_hardware_stages(
        hardware, physical_confirmed, electrode_mode=electrode_mode)
    payload = _startup_payload(
        stages, port, montage, seconds=seconds,
        electrode_mode=electrode_mode, cyton_gain=resolved_gain)
    html_path = json_path = None
    if payload["passed"] or save_failed:
        if out_base is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_base = ROOT / "reports" / "startup" / f"headband_{stamp}"
        html_path, json_path, payload = save_startup_report(
            stages, out_base, port, montage, seconds=seconds,
            electrode_mode=electrode_mode, cyton_gain=resolved_gain)
    payload["report_written"] = bool(html_path and json_path)
    payload["html_report"] = str(html_path) if html_path else None
    payload["json_report"] = str(json_path) if json_path else None

    if verbose:
        print("\nSTARTUP CHECK SUMMARY")
        print("-" * 86)
        shown = stages if payload["passed"] else [
            stage for stage in stages if stage["status"] == "FAIL"]
        for stage in shown:
            print(f"  [{stage['status']:<4}] {stage['name']}: "
                  f"{stage['summary']}")
        print(f"\n  DECISION: {payload['decision']}")
        if payload["report_written"]:
            print(f"  HTML -> {html_path}")
            print(f"  JSON -> {json_path}")
        else:
            print("  No report written; reports are saved only after READY.")
    return payload


def _boardwide_range_failure(stages):
    """True only when every live channel is clipped/railed or over-range."""
    normal = next((stage for stage in stages
                   if str(stage.get("name", "")).startswith("S3")), None)
    channels = (normal or {}).get("details", {}).get("channels", [])
    return bool(len(channels) == 8 and all(
        {"rail", "amplitude"}.intersection(row.get("reasons", []))
        for row in channels))


def _next_thinkpulse_gain(gain):
    gains = (8, 4, 2, 1)
    gain = int(gain)
    try:
        return gains[gains.index(gain) + 1]
    except (ValueError, IndexError):
        return None


def _render_silent_counter(label, count, delay):
    print(f"\r  {label}: {count} | retry every {delay:g}s | Ctrl+C aborts",
          end="", flush=True)


def wait_for_headband_ready(port, montage="cyton8_motor", seconds=6.0,
                            physical_confirmed=False, out_base=None,
                            retry_delay=2.0, confirm_callback=None,
                            check_fn=None, sleep_fn=None,
                            electrode_mode="thinkpulse", cyton_gain=None):
    """Poll until READY and update one in-place duplicate-failure counter.

    Physical setup remains the operator's responsibility but no YES prompt is
    required. Failed polls create no reports. A changed root failure prints one
    new diagnostic block; identical failures only replace the counter line.
    """
    if not port:
        raise ValueError("direct Cyton startup check requires a serial port")
    electrode_mode = normalize_electrode_mode(electrode_mode)
    check = check_fn or run_headband_check
    pause = sleep_fn or time.sleep
    delay = max(0.0, float(retry_delay))
    active_gain = int(cyton_gain if cyton_gain is not None else
                      (8 if electrode_mode == "thinkpulse" else 24))
    # Retained in the public signature so existing callers do not break; startup
    # no longer asks for manual confirmation.
    _ = physical_confirmed, confirm_callback
    attempt = 0
    last_failure = None
    silent_repeats = 0
    counter_visible = False

    while True:
        attempt += 1
        failure_exception = None
        try:
            kwargs = {
                "montage": montage, "seconds": seconds,
                "physical_confirmed": False, "out_base": out_base,
                "electrode_mode": electrode_mode, "cyton_gain": active_gain,
            }
            if check_fn is None:
                kwargs.update(verbose=False, save_failed=False)
            payload = check(port, **kwargs)
        except KeyboardInterrupt:
            if counter_visible:
                print()
            raise
        except Exception as exc:  # noqa: BLE001 - hardware faults must retry
            payload = None
            failure_exception = exc

        if payload is not None and payload.get("passed"):
            payload["cyton_gain"] = active_gain
            if counter_visible:
                print()
                counter_visible = False
            print(f"\nSTARTUP READY after {attempt} check(s). Continuing the "
                  "executing script.")
            if payload.get("report_written"):
                print(f"  HTML -> {payload['html_report']}")
                print(f"  JSON -> {payload['json_report']}")
            return payload

        stages = (payload or {}).get("stages", [])
        if (failure_exception is None and electrode_mode == "thinkpulse" and
                _boardwide_range_failure(stages)):
            lower_gain = _next_thinkpulse_gain(active_gain)
            if lower_gain is not None:
                if counter_visible:
                    print()
                    counter_visible = False
                print(f"\nAUTO-GAIN: all eight channels are over-range/railed; "
                      f"reducing Cyton gain {active_gain} -> {lower_gain}.")
                active_gain = lower_gain
                last_failure = None
                silent_repeats = 0
                continue

        failed = [stage for stage in stages if stage.get("status") == "FAIL"]
        signature = (
            ("exception", type(failure_exception).__name__, str(failure_exception))
            if failure_exception is not None else
            tuple((stage.get("name"), stage.get("summary")) for stage in failed))
        if signature != last_failure:
            if counter_visible:
                print()
            print("\nSTARTUP CHECK WAITING — recording/live BCI remains blocked")
            if failure_exception is not None:
                print(f"  [FAIL] {type(failure_exception).__name__}: "
                      f"{failure_exception}")
            elif failed:
                for stage in failed:
                    print(f"  [FAIL] {stage.get('name')}: {stage.get('summary')}")
            else:
                print("  [FAIL] startup check returned no usable READY result")
            hints = startup_failure_hints(
                stages, exception=failure_exception, port=port,
                montage=montage, seconds=seconds,
                electrode_mode=electrode_mode, cyton_gain=active_gain)
            if hints:
                print("\n  HOW TO FIX / DEBUG:")
                for hint in hints:
                    print(f"   - {hint}")
            silent_repeats = 0
            last_failure = signature
        else:
            silent_repeats += 1
        _render_silent_counter(
            "Silent repeated failures", silent_repeats, delay)
        counter_visible = True
        pause(delay)


def main():
    parser = argparse.ArgumentParser(
        description="Mandatory pre-use/pre-record Cyton headset startup gate")
    parser.add_argument("--port", required=True, help="Cyton dongle COM port")
    parser.add_argument("--montage", default="cyton8_motor")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--retry-delay", type=float, default=2.0,
                        help="seconds between silent failed polls; default 2")
    parser.add_argument("--electrode-mode", choices=("passive", "thinkpulse"),
                        default="thinkpulse")
    parser.add_argument("--cyton-gain", type=int,
                        choices=(1, 2, 4, 6, 8, 12, 24), default=None,
                        help="default 8 for ThinkPulse, 24 for passive electrodes")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if (args.electrode_mode == "thinkpulse" and
            args.cyton_gain is not None and args.cyton_gain > 8):
        parser.error("ThinkPulse requires --cyton-gain 8 or lower")
    try:
        wait_for_headband_ready(
            args.port, montage=args.montage, seconds=args.seconds,
            out_base=args.out, retry_delay=args.retry_delay,
            electrode_mode=args.electrode_mode, cyton_gain=args.cyton_gain)
    except KeyboardInterrupt:
        print("\nStartup check aborted by operator; no recording/live BCI started.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
