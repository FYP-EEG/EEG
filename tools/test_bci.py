"""
Author: Anson
Created on: 6/10/2026
Purpose: the whole test suite, in one file, organised by failure layer.

    python tests/test_bci.py              # everything
    python tests/test_bci.py --layer 3    # one layer

WHY LAYERS AND NOT FEATURES
---------------------------
A BCI can fail at five independent layers and every one of them presents
identically: nothing happens when the user tries. Grouping the tests by layer
means a failure already tells you where to look, and that the layers must be
read IN ORDER - a layer-3 result computed on layer-1-broken data is noise.

TWO MODES, SAME FIVE LAYERS
---------------------------
    python tests/test_bci.py                     code tests, all layers
    python tests/test_bci.py --layer 3           code tests, one layer
    python tests/test_bci.py --user anson        diagnostics on real data
    python tests/test_bci.py --recording "p/*.npz"   same, explicit file
    python tests/test_bci.py --user anson --code     both

CODE mode fabricates its own data and asks "is the software correct?" - it is
deterministic, needs no hardware, and a failure is a bug. DATA mode reads an
actual recording and asks "is THIS person\'s signal usable?" - a failure there
is not a bug, it is a finding, which is why it reports PASS/WARN/FAIL verdicts
rather than assertions and does not set a non-zero exit code on its own.

Layer 2 in DATA mode is the go/no-go gate: a recording that cannot clear it
will not be rescued by a better classifier.
"""
import argparse
import glob
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# The original suites each pushed their own paths; several modules are still
# imported bare (stream_bridge, stimulus) because EEG/ and pygame_lib/ have no
# __init__.py. Declared once here instead of ten times.
for _sub in ("", "EEG", "pygame_lib"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

ok = 0
fail = []
_layer_start = 0


def _record(n, c, detail=""):
    global ok
    if c:
        ok += 1
        print(f"  PASS  {n}")
    else:
        fail.append(n)
        print(f"  FAIL  {n}" + (f"\n        {detail}" if detail else ""))


def check(n, c, detail=""):
    """Standard order, used by nine of the ten original suites."""
    _record(n, c, detail)


def check_cond_first(c, n, detail=""):
    """Reversed order, as the realtime section was originally written.

    Calls _record directly. Routing it through check() would recurse, because
    `check` is rebound to THIS function for the duration of that section.
    """
    _record(n, c, detail)


def banner(title, why):
    global _layer_start
    _layer_start = ok + len(fail)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for line in why.split("\n"):
        print("  " + line)
    print()


def layer_total(title):
    n = ok + len(fail) - _layer_start
    print(f"\n  -- {title.split(' - ')[0]}: {n} checks")


def layer1():
    global check
    banner('LAYER 1 - ACQUISITION AND SIGNAL QUALITY', 'Can the system get clean, correctly-timed samples off the hardware at all?')

    # --------------------------------------------------------------
    # from test_step4_bridge.py
    # --------------------------------------------------------------
    import sys, time, glob
    from pathlib import Path
    import numpy as np

    from stream_bridge import (RingBuffer, StreamBridge, SyntheticBackend,
                               ReplayBackend, OpenBCICSVBackend,
                               OpenBCIGUIUDPBackend, load_openbci_gui_csv,
                               BrainFlowBackend)  # noqa
    from ML.bci_engine import BCIEngine                # noqa
    from ML.hybrid_classifier import HybridSSVEPClassifier  # noqa
    import inspect
    brainflow_start_source = inspect.getsource(BrainFlowBackend.start)
    brainflow_signature = inspect.signature(BrainFlowBackend)
    check("direct Cyton acquisition resets persistent test inputs to defaults",
          'send_cyton_command(self.board, "d")' in brainflow_start_source and
          "self.reset_channels" in brainflow_start_source)
    check("current hardware defaults are COM8 and ThinkPulse",
          brainflow_signature.parameters["serial_port"].default == "COM8" and
          brainflow_signature.parameters["electrode_mode"].default ==
          "thinkpulse")
    from EEG.cyton_commands import (channel_settings_command,
                                    cyton_profile_settings,
                                    live_channel_commands,
                                    send_cyton_command)

    class NonUtf8ResponseBoard:
        def __init__(self):
            self.commands = []

        def config_board(self, command):
            self.commands.append(command)
            raise UnicodeDecodeError("utf-8", b"\xa0", 0, 1,
                                     "invalid start byte")

    response_board = NonUtf8ResponseBoard()
    command_result = send_cyton_command(response_board, "d")
    check("successful Cyton command tolerates only optional response decode failure",
          response_board.commands == ["d"] and
          command_result["response_decode_ignored"] and
          "0xa0" in command_result["decode_error"])
    thinkpulse_commands = live_channel_commands("thinkpulse")
    check("ThinkPulse profile uses gain 8, Normal, BIAS off, SRB2 on for 8 channels",
          len(thinkpulse_commands) == 8 and
          thinkpulse_commands[0] == "x1040010X" and
          thinkpulse_commands[-1] == "x8040010X" and
          channel_settings_command(1, gain=24) == "x1060110X")
    gui_profile = cyton_profile_settings("openbci_gui_default")
    active_profile = cyton_profile_settings("thinkpulse_active", gain=4)
    check("explicit GUI parity and active Cyton profiles cannot be conflated",
          gui_profile["gain"] == 24 and gui_profile["include_bias"] and
          gui_profile["srb2"] and not gui_profile["srb1"] and
          active_profile["gain"] == 4 and not active_profile["include_bias"] and
          active_profile["electrode_mode"] == "thinkpulse" and
          live_channel_commands(profile="openbci_gui_default")[0] ==
          "x1060110X")
    try:
        live_channel_commands("thinkpulse", gain=12)
        check("ThinkPulse rejects Cyton gains above 8", False)
    except ValueError as exc:
        check("ThinkPulse rejects Cyton gains above 8", "8 or lower" in str(exc))

    class FakeCytonBoard:
        def __init__(self):
            self.commands = []
            self.prepared = False

        def prepare_session(self):
            self.prepared = True

        def config_board(self, command):
            self.commands.append(command)
            return ""

        def start_stream(self, _buffer_size):
            pass

        def get_board_data_count(self):
            return 1

        def is_prepared(self):
            return self.prepared

        def stop_stream(self):
            pass

        def release_session(self):
            self.prepared = False

    class FakeBoardShim:
        @staticmethod
        def get_sampling_rate(_board_id):
            return 250

        @staticmethod
        def get_eeg_channels(_board_id):
            return list(range(8))

    fake_cyton = FakeCytonBoard()
    thinkpulse_backend = BrainFlowBackend.__new__(BrainFlowBackend)
    thinkpulse_backend.board = fake_cyton
    thinkpulse_backend.board_id = 0
    thinkpulse_backend.descriptor_board_id = 0
    thinkpulse_backend.n_channels = 8
    thinkpulse_backend.reset_channels = True
    thinkpulse_backend.BoardShim = FakeBoardShim
    thinkpulse_backend._live_channel_commands = thinkpulse_commands
    thinkpulse_backend.start()
    thinkpulse_backend.stop()
    check("recording backend reapplies all eight ThinkPulse settings after reset",
          fake_cyton.commands == ["d", *thinkpulse_commands])

    import json
    gui_udp = OpenBCIGUIUDPBackend(startup_timeout=0)
    gui_payload = json.dumps({
        "type": "timeSeriesRaw",
        "data": [list(np.arange(5, dtype=float) + channel)
                 for channel in range(8)],
    }).encode("utf-8")
    gui_chunk = gui_udp._decode_packet(gui_payload, ("127.0.0.1", 54321))
    filtered_chunk = gui_udp._decode_packet(json.dumps({
        "type": "timeSeriesFilt", "data": [[0.0]] * 8,
    }).encode("utf-8"), ("127.0.0.1", 54321))
    check("GUI UDP backend accepts only raw 8-channel JSON in microvolts",
          gui_chunk.shape == (8, 5) and filtered_chunk is None and
          gui_udp.info()["units"] == "uV" and
          gui_udp.wrong_type_packets == 1 and gui_udp.packet_count == 1)

    from EEG.raw_transfer import analyse_raw_transfer
    transfer_report = analyse_raw_transfer(
        clean_transfer := np.random.default_rng(771).normal(0, 5, (8, 1500)),
        fs=250, elapsed=6.0,
        source_info={"backend": "fixture", "units": "uV",
                     "eeg_channel_rows": list(range(1, 9))},
        transport={"packet_continuity_exact": True,
                   "missing_packets_estimate": 0})
    check("raw transfer evidence reports rows, physical variance and no filtering",
          transfer_report["decision"] == "RAW_TRANSFER_PASS" and
          transfer_report["raw_only"] and
          not transfer_report["filtering_applied"] and
          transfer_report["channels"][0]["board_row"] == 1 and
          transfer_report["channels"][0]["std_uv"] > 1 and
          transfer_report["packet_continuity"]["exact"])

    from ML.montage import get as get_montage
    motor_montage = get_montage("cyton8_motor")
    check("current motor montage matches the confirmed Cyton CH1-CH8 order",
          motor_montage["labels"] ==
          ["F3", "C3", "CZ", "P3", "PZ", "F4", "C4", "P4"] and
          motor_montage["artifact"] == [0, 5] and
          motor_montage["motor_lateral"] == [1, 6])

    print("\n== Pre-recording connection/signal preflight ==")
    from EEG.signal_preflight import analyse_preflight, save_preflight_report
    preflight_rng = np.random.default_rng(77)
    clean_preflight = preflight_rng.normal(0, 5, (8, 1500))
    clean_result = analyse_preflight(
        clean_preflight, fs=250, elapsed=6.0,
        channel_labels=["F3", "C3", "Cz", "P3", "Pz", "F4", "C4", "P4"])
    check("clean 8-channel/250-Hz baseline passes preflight",
          clean_result["connection_ok"] and clean_result["signal_ok"])
    scaled_result = analyse_preflight(
        clean_preflight * 100.0, fs=250, elapsed=6.0)
    check("adaptive preflight retains stable large-but-valid common scaling",
          scaled_result["signal_ok"] and scaled_result["adaptive_amplitude"] and
          all(not row["reasons"] for row in scaled_result["channels"]))
    offset_result = analyse_preflight(
        clean_preflight + 200_000.0, fs=250, elapsed=6.0)
    check("stable raw offset is measured but not mistaken for amplitude failure",
          offset_result["signal_ok"] and
          any("stable raw offset was retained" in line
              for line in offset_result["debug"]))
    local_outlier = clean_preflight.copy()
    local_outlier[4] *= 1000.0
    local_result = analyse_preflight(local_outlier, fs=250, elapsed=6.0)
    check("adaptive preflight still rejects a gross local scale outlier",
          not local_result["signal_ok"] and
          "amplitude" in local_result["channels"][4]["reasons"])
    clipped = np.clip(clean_preflight * 100_000.0, -187_500.0, 187_500.0)
    clipped_result = analyse_preflight(clipped, fs=250, elapsed=6.0)
    check("repeated extrema identify actual board-wide clipping",
          not clipped_result["signal_ok"] and
          clipped_result["boardwide_range_failure"] and
          all("rail" in row["reasons"] for row in clipped_result["channels"]))
    square = np.tile(np.where(np.arange(1500) % 2, 2000.0, -2000.0), (8, 1))
    square_result = analyse_preflight(square, fs=250, elapsed=6.0)
    check("board-wide square/test signal is blocked with debug diagnosis",
          not square_result["signal_ok"] and
          square_result["possible_test_signal"] and
          all(not row["ok"] for row in square_result["channels"]))
    flat = clean_preflight.copy(); flat[3] = 0
    flat_result = analyse_preflight(flat, fs=250, elapsed=6.0)
    check("one flat electrode blocks recording and identifies its channel",
          not flat_result["signal_ok"] and
          flat_result["channels"][3]["reasons"] == ["flat"])
    from ML.quality import screen_trials
    epoch_set = np.stack([
        preflight_rng.normal(0, 20, (8, 750)) for _ in range(20)])
    scaled_screen = screen_trials(epoch_set * 100.0, verbose=False)
    check("offline trial screening also adapts to a stable recording scale",
          scaled_screen["keep"].all())
    corrupted_epochs = epoch_set.copy(); corrupted_epochs[3, 4] *= 1000.0
    corrupted_screen = screen_trials(corrupted_epochs, verbose=False)
    check("offline adaptive screening rejects an isolated gross channel fault",
          not corrupted_screen["keep"][3] and
          corrupted_screen["channel_reasons"]["amplitude"][3, 4])
    import tempfile
    with tempfile.TemporaryDirectory() as preflight_tmp_text:
        preflight_html, preflight_json = save_preflight_report(
            flat_result, flat, Path(preflight_tmp_text) / "preflight")
        check("failed preflight writes HTML and JSON debugging reports",
              preflight_html.exists() and preflight_json.exists() and
              preflight_html.stat().st_size > 1000)

    print("\n== Mandatory headset startup gate ==")
    from startup.check_headband import (evaluate_hardware_stages,
                                        run_headband_check,
                                        save_startup_report,
                                        startup_failure_hints,
                                        wait_for_headband_ready)
    startup_source = [
        {"name": "L0A known internal ground", "status": "PASS",
         "summary": "ground", "details": {"rms_uv": [2.0] * 8,
                                             "samples": 1500, "fs": 250}},
        {"name": "L0B known square-wave transfer", "status": "PASS",
         "summary": "square", "details": {
             "channels": [{"channel": index, "kind": "square",
                           "fundamental": 2.0, "h3_ratio": .3, "snr": 100}
                          for index in range(8)],
             "samples": 1500, "fs": 250}},
        {"name": "L1 live normal-electrode signal", "status": "PASS",
         "summary": "normal", "details": clean_result},
    ]
    startup_stages = evaluate_hardware_stages(startup_source, True)
    check("startup requires 8/8 ground/square/normal without a typed confirmation",
          len(startup_stages) == 4 and
          all(row["status"] == "PASS" for row in startup_stages) and
          not startup_stages[0]["details"]["manual_confirmation_required"])
    thinkpulse_stages = evaluate_hardware_stages(
        startup_source, False, electrode_mode="thinkpulse")
    check("ThinkPulse startup states SRB2/BIAS wiring without claiming verification",
          thinkpulse_stages[0]["status"] == "PASS" and
          not thinkpulse_stages[0]["details"]["bias_expected_attached"] and
          not thinkpulse_stages[0]["details"]["software_verified"] and
          "wiring contract" in thinkpulse_stages[0]["name"] and
          "SRB2" in thinkpulse_stages[0]["summary"])
    thinkpulse_hints = startup_failure_hints(
        [{**thinkpulse_stages[0], "status": "FAIL"}], port="COM8",
        electrode_mode="thinkpulse", cyton_gain=8)
    check("ThinkPulse failure reproducer preserves electrode mode and gain",
          any("--electrode-mode thinkpulse --cyton-gain 8" in value
              for value in thinkpulse_hints))
    disconnected_stages = evaluate_hardware_stages([{
        "name": "L0A known internal ground", "status": "FAIL",
        "summary": "BrainFlowError: unable to open board on COM8",
        "details": {},
    }], True)
    check("disconnected Cyton is one root failure, not three misleading stages",
          [row["status"] for row in disconnected_stages] ==
          ["PASS", "FAIL", "SKIP", "SKIP"] and
          "not connected or produced no usable EEG samples" in
          disconnected_stages[1]["summary"])
    decode_stages = evaluate_hardware_stages([{
        "name": "L0A known internal ground", "status": "FAIL",
        "summary": ("UnicodeDecodeError: 'utf-8' codec can't decode byte "
                    "0xa0 in position 0"), "details": {},
    }], True)
    decode_hints = startup_failure_hints(decode_stages, port="COM8")
    check("non-UTF8 Cyton response is not misreported as a disconnected board",
          [row["status"] for row in decode_stages] ==
          ["PASS", "FAIL", "SKIP", "SKIP"] and
          decode_stages[1]["details"]["decode_failure"] and
          "not proof that the board is disconnected" in
          decode_stages[1]["summary"] and
          any("brainflow>=5.22.0" in value for value in decode_hints))
    seven_channel_source = [dict(row) for row in startup_source]
    seven_channel_source[1] = {
        **startup_source[1], "details": {
            **startup_source[1]["details"],
            "channels": [dict(row) for row in
                         startup_source[1]["details"]["channels"]]}}
    seven_channel_source[1]["details"]["channels"][7]["kind"] = "sine"
    blocked_stages = evaluate_hardware_stages(seven_channel_source, True)
    check("startup blocks recording when even one known-signal channel fails",
          blocked_stages[2]["status"] == "FAIL" and
          "7/8" in blocked_stages[2]["summary"])
    blocked_hints = startup_failure_hints(
        blocked_stages, port="COM8", montage="cyton8_motor")
    check("startup failure names a likely fix and reproducible debugger command",
          any("S2 square wave" in value and "CH8" in value
              for value in blocked_hints) and
          any("debug_bci_pipeline.py --port COM8" in value
              for value in blocked_hints))
    with tempfile.TemporaryDirectory() as startup_tmp_text:
        startup_html, startup_json, startup_payload = save_startup_report(
            blocked_stages, Path(startup_tmp_text) / "startup", "COM8",
            "cyton8_motor")
        check("startup report helper can render READY/BLOCKED evidence",
              not startup_payload["passed"] and
              startup_payload["decision"] == "BLOCKED" and
              startup_payload["debug_hints"] and
              "How to fix/debug" in startup_html.read_text(encoding="utf-8") and
              startup_html.exists() and startup_json.exists())

        import tools.debug_bci_pipeline as debug_module
        original_hardware_layers = debug_module.hardware_layers
        failed_base = Path(startup_tmp_text) / "failed_poll"
        ready_base = Path(startup_tmp_text) / "valid_session"
        try:
            debug_module.hardware_layers = lambda *_args, **_kwargs: [{
                "name": "L0A known internal ground", "status": "FAIL",
                "summary": "board not connected", "details": {},
            }]
            failed_poll = run_headband_check(
                "COM8", physical_confirmed=True, out_base=failed_base,
                verbose=False)
            debug_module.hardware_layers = lambda *_args, **_kwargs: startup_source
            ready_poll = run_headband_check(
                "COM8", physical_confirmed=True, out_base=ready_base,
                verbose=False)
        finally:
            debug_module.hardware_layers = original_hardware_layers
        check("automatic startup writes no failed-poll reports and one READY report",
              not failed_poll["report_written"] and
              not failed_base.with_suffix(".html").exists() and
              not failed_base.with_suffix(".json").exists() and
              ready_poll["report_written"] and
              ready_base.with_suffix(".html").exists() and
              ready_base.with_suffix(".json").exists())

        import io
        from contextlib import redirect_stdout
        retry_calls, retry_sleeps = [], []
        repeated_failure = {
            "passed": False, "decision": "BLOCKED",
            "stages": [{"name": "S2 square", "status": "FAIL",
                        "summary": "square wave missing"}],
        }
        retry_payloads = [repeated_failure, repeated_failure, {
            "passed": True, "decision": "READY", "report_written": False,
            "stages": [{"name": "S2 square", "status": "PASS"}],
        }]

        def fake_startup_check(port, **kwargs):
            retry_calls.append((port, kwargs["out_base"]))
            return retry_payloads.pop(0)

        retry_output = io.StringIO()
        with redirect_stdout(retry_output):
            retry_result = wait_for_headband_ready(
                "COM8", physical_confirmed=True,
                out_base=Path(startup_tmp_text) / "retry",
                retry_delay=0, check_fn=fake_startup_check,
                sleep_fn=retry_sleeps.append)
        check("identical startup failure prints once and replaces one counter line",
              retry_result["passed"] and len(retry_calls) == 3 and
              retry_sleeps == [0.0, 0.0] and
              retry_output.getvalue().count("STARTUP CHECK WAITING") == 1 and
              "\r  Silent repeated failures: 1" in retry_output.getvalue() and
              retry_calls[0][1] == retry_calls[1][1] == retry_calls[2][1])

        gain_calls = []

        def fake_gain_check(_port, **kwargs):
            gain_calls.append(kwargs["cyton_gain"])
            if kwargs["cyton_gain"] == 8:
                return {
                    "passed": False, "stages": [{
                        "name": "S3 configured Normal input and live electrodes",
                        "status": "FAIL", "summary": "railed",
                        "details": {"channels": [
                            {"reasons": ["rail"]} for _ in range(8)]},
                    }],
                }
            return {"passed": True, "decision": "READY",
                    "report_written": False, "stages": []}

        gain_output = io.StringIO()
        with redirect_stdout(gain_output):
            gain_payload = wait_for_headband_ready(
                "COM8", out_base=Path(startup_tmp_text) / "gain",
                retry_delay=0, check_fn=fake_gain_check,
                sleep_fn=lambda _delay: None,
                electrode_mode="thinkpulse", cyton_gain=8)
        check("board-wide clipping lowers ThinkPulse gain and returns selected gain",
              gain_calls == [8, 4] and gain_payload["cyton_gain"] == 4 and
              "reducing Cyton gain 8 -> 4" in gain_output.getvalue())

    import EEG.data_record as recorder_module
    direct_meta = recorder_module._acquisition_report_metadata({
        "backend": "brainflow", "acquisition_route": "direct_cyton",
        "fs": 250, "n_channels": 8, "units": "uV", "filtered": False,
        "cyton_profile": "thinkpulse_active", "cyton_gain": 4,
        "electrode_mode": "thinkpulse", "eeg_channel_rows": list(range(1, 9)),
        "profile_settings": {"include_bias": False},
    })
    gui_meta = recorder_module._acquisition_report_metadata({
        "backend": "openbci_gui_udp", "fs": 250, "n_channels": 8,
        "units": "uV", "filtered": False, "bind_ip": "127.0.0.1",
        "port": 12345, "packet_schema": "timeSeriesRaw",
    })
    csv_meta = recorder_module._acquisition_report_metadata({
        "backend": "openbci_csv", "fs": 250, "n_channels": 8,
        "path": "OpenBCI-RAW.csv", "input_unit": "uV", "scale_to_uv": 1.0,
        "sample_rate_source": "metadata", "rows_dropped": 2,
        "loop": True, "speed": 1.0,
    })
    check("recording metadata serializes direct, GUI UDP and CSV routes distinctly",
          direct_meta["route"] == "direct_cyton" and
          direct_meta["cyton_profile"] == "thinkpulse_active" and
          direct_meta["eeg_channel_rows"] == list(range(1, 9)) and
          gui_meta["route"] == "openbci_gui_udp" and
          gui_meta["packet_sequence_available"] is False and
          csv_meta["route"] == "openbci_gui_csv_replay" and
          csv_meta["cue_alignment_valid"] is False and
          csv_meta["rows_dropped"] == 2)
    import startup.check_headband as startup_module
    from types import SimpleNamespace
    original_wait_ready = startup_module.wait_for_headband_ready
    propagated_args = SimpleNamespace(
        dry_run=False, eeg_csv=None, board_id=0, serial_port="COM8",
        startup_retry_delay=0, subject="test", montage="cyton8_motor",
        preflight_seconds=2.0, electrode_mode="thinkpulse", cyton_gain=8)
    try:
        startup_module.wait_for_headband_ready = lambda *_args, **_kwargs: {
            "passed": True, "cyton_gain": 4}
        recorder_module._run_required_headband_startup(propagated_args)
    finally:
        startup_module.wait_for_headband_ready = original_wait_ready
    check("startup-selected gain propagates into recording backend arguments",
          propagated_args.cyton_gain == 4)

    original_live_preflight = recorder_module._start_real_capture_with_preflight
    preflight_calls, ready_capture = [], object()

    def fake_live_preflight(_backend, _args, quiet=False):
        preflight_calls.append(quiet)
        return None if len(preflight_calls) == 1 else ready_capture

    try:
        recorder_module._start_real_capture_with_preflight = fake_live_preflight
        retained_capture = recorder_module._start_real_capture_until_ready(
            object(), object(), retry_delay=0)
    finally:
        recorder_module._start_real_capture_with_preflight = original_live_preflight
    check("failed live preflight keeps recorder/example alive until fixed",
          retained_capture is ready_capture and len(preflight_calls) == 2)

    def interrupting_cli():
        raise KeyboardInterrupt

    check("data recorder Ctrl+C aborts cleanly without a traceback",
          recorder_module._run_cli_safely(interrupting_cli) == 130)
    from tools.test_signal_check import resolve_board
    check("signal checker treats a supplied COM port as real Cyton by default",
          resolve_board(None, "COM8") == "cyton" and
          resolve_board(None, None) == "synthetic" and
          resolve_board("cyton_daisy", "COM8") == "cyton_daisy")

    print("\n== RingBuffer ==")
    rb = RingBuffer(3, 100)
    check("empty buffer returns None", rb.latest(10) is None)
    rb.write(np.ones((3,50)))
    check("partial fill: not enough samples yet", rb.latest(60) is None)
    check("partial fill: enough samples", rb.latest(50).shape == (3,50))
    rb.write(np.full((3,80), 2.0))
    got = rb.latest(100)
    check("wraps around capacity", got.shape == (3,100))
    check("wrap keeps newest values", np.allclose(got[:, -80:], 2.0))
    check("len caps at capacity", len(rb) == 100)
    rb2 = RingBuffer(2, 50)
    rb2.write(np.arange(200).reshape(1,-1).repeat(2,0))
    check("oversized write keeps newest", rb2.latest(50)[0,-1] == 199)
    try:
        rb2.write(np.ones((5,10))); check("rejects wrong channel count", False)
    except ValueError:
        check("rejects wrong channel count", True)

    print("\n== RingBuffer reconstructs a continuous signal exactly ==")
    fs, total = 250, 4000
    sig = np.vstack([np.sin(2*np.pi*15*np.arange(total)/fs)]*2)
    rb3 = RingBuffer(2, 2000); i=0; rng=np.random.default_rng(0)
    while i < total:
        n = min(int(rng.integers(1,97)), total-i)
        rb3.write(sig[:, i:i+n]); i += n
    check("ragged chunks reconstruct exactly",
          np.allclose(rb3.latest(750), sig[:, total-750:total]))

    print("\n== Windowing cadence & overlap ==")
    be = SyntheticBackend(seed=2); be.set_active(0)
    br = StreamBridge(backend=be, window=750, hop=250, max_pending=100)
    br.start(); time.sleep(6.0); br.stop()
    r = br.report()
    # 6s elapsed - 3s priming = 3s steady-state at 1 window/s
    expected_w = int((6.0 - 750/250) * 250/250)
    check(f"windows produced ({r['windows_made']}, expected ~{expected_w})",
          r["windows_made"] >= expected_w - 1)
    check(f"steady-state rate ~{r['expected_windows_per_s']:.1f}/s (got {r['windows_per_s']:.2f})",
          abs(r["windows_per_s"] - r["expected_windows_per_s"]) < 0.35)
    check(f"effective fs within 3% (err {r['fs_error_pct']:+.2f}%)",
          abs(r["fs_error_pct"]) < 3.0)
    ws = br.poll_all()
    check("all windows are (8,750)", all(w.shape == (8,750) for w in ws))
    if len(ws) >= 2:
        a, b = ws[0], ws[1]
        check("consecutive windows overlap by window-hop",
              np.allclose(a[:, 250:], b[:, :500]))

    print("\n== Backpressure: slow consumer drops oldest, never blocks ==")
    be2 = SyntheticBackend(seed=3)
    br2 = StreamBridge(backend=be2, window=750, hop=250, max_pending=2)
    br2.start(); time.sleep(6.5)
    pend = len(br2._pending)
    br2.stop()
    check("pending never exceeds max_pending", pend <= 2)
    check("drops were counted", br2.report()["windows_dropped"] > 0)
    w = br2.poll()
    check("still serves a valid window after drops", w is not None and w.shape==(8,750))

    print("\n== poll() is non-blocking ==")
    be3 = SyntheticBackend(seed=4)
    br3 = StreamBridge(backend=be3); br3.start()
    t0=time.time(); [br3.poll() for _ in range(2000)]; dt=time.time()-t0
    check(f"2000 polls took {dt*1000:.0f}ms (non-blocking)", dt < 0.5)
    check("bridge thread is alive", br3.is_running)
    br3.stop()
    check("stops cleanly", not br3.is_running)

    print("\n== Live-path classification (the whole point of Step 4) ==")
    clf = HybridSSVEPClassifier()
    for tgt in (0,1):
        be4 = SyntheticBackend(seed=1); be4.set_active(tgt)
        br4 = StreamBridge(backend=be4); br4.start()
        w = br4.wait_for_window(timeout=8); br4.stop()
        sc = clf.fbcca_scores(clf.apply_filter(w))
        check(f"gaze target {tgt} classified correctly through the bridge",
              int(np.argmax(sc)) == tgt and sc.max() > clf.confidence_threshold)

    be5 = SyntheticBackend(seed=1); be5.set_idle(True)
    br5 = StreamBridge(backend=be5); br5.start()
    w = br5.wait_for_window(timeout=8); br5.stop()
    check("idle stays below threshold through the bridge",
          clf.fbcca_scores(clf.apply_filter(w)).max() < clf.confidence_threshold)

    print("\n== Phase continuity regression (bug found by Step 4) ==")
    be6 = SyntheticBackend(seed=5); be6.set_active(0); be6.start()
    chunks = []
    for _ in range(12):                       # accumulate >= 750 samples in many chunks
        time.sleep(0.3)
        c = be6.read_available()
        if c is not None:
            chunks.append(c)
        if sum(c.shape[1] for c in chunks) >= 900:
            break
    joined = np.concatenate(chunks, axis=1)
    check(f"gathered {len(chunks)} separate chunks", len(chunks) >= 3)
    sc = clf.fbcca_scores(clf.apply_filter(joined[:, -750:]))
    check(f"stitched chunks preserve SSVEP (conf {sc.max():.2f}, phase continuous)",
          int(np.argmax(sc)) == 0 and sc.max() > clf.confidence_threshold)

    print("\n== End-to-end with BCIEngine ==")
    be7 = SyntheticBackend(seed=1); be7.set_active(0)
    eng = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=0)
    br7 = StreamBridge(backend=be7); br7.start()
    fired=[]; lat=[]
    end=time.time()+9
    while time.time()<end:
        w = br7.poll()
        if w is not None:
            t0=time.time(); d,f = eng.process_frame(w); lat.append((time.time()-t0)*1000)
            if f: fired.append(d)
        time.sleep(1/60)
    br7.stop()
    check(f"engine dispatched a command ({fired})", len(fired) >= 1)
    check("dispatched the correct target", all(d=="SSVEP_0" for d in fired))
    check(f"classify latency < hop budget (mean {np.mean(lat):.0f}ms vs 1000ms)",
          np.mean(lat) < 1000)
    check("no windows dropped at 60fps", br7.report()["windows_dropped"] == 0)

    print("\n== Blink through the live path ==")
    be8 = SyntheticBackend(seed=6); be8.set_active(0)
    eng8 = BCIEngine(mode="SSVEP", debounce_window=2, artifact_lockout_ms=800)
    br8 = StreamBridge(backend=be8); br8.start()
    br8.wait_for_window(timeout=8)
    be8.trigger_blink()
    seen=[]
    end=time.time()+4
    while time.time()<end:
        w=br8.poll()
        if w is not None: seen.append(eng8.process_frame(w)[0])
        time.sleep(1/60)
    br8.stop()
    check(f"blink detected in live stream ({seen})", "BLINK" in seen or "LOCKOUT" in seen)

    print("\n== ReplayBackend (real recorded EEG through the live path) ==")
    recs = sorted(glob.glob(str(ROOT/"dataset"/"calib_*.npz")))
    if recs:
        rp = ReplayBackend(recs[-1], speed=4.0)
        info = rp.info()
        check("replay reports duration", info["duration_s"] > 0)
        br9 = StreamBridge(backend=rp); br9.start()
        w = br9.wait_for_window(timeout=10); br9.stop()
        check("replay yields a valid window", w is not None and w.shape[0]==8)
        check("replay window has real signal variance", w is not None and w.std() > 0)
    else:
        print("  SKIP  no calibration recording found (run Step 3 first)")

    print("\n== OpenBCI GUI CSV offline backend ==")
    with tempfile.TemporaryDirectory() as csv_tmp_text:
        csv_path = Path(csv_tmp_text) / "OpenBCI-RAW-test.csv"
        csv_time = np.arange(1000) / 250.0
        csv_signal = np.vstack([
            20.0 * np.sin(2 * np.pi * 10.0 * csv_time + channel)
            for channel in range(8)])
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            stream.write("%OpenBCI Raw EEG Data\n")
            stream.write("%Number of channels = 8\n")
            stream.write("%Sample Rate = 250 Hz\n")
            stream.write("%Board = Cyton\n")
            stream.write("Sample Index," + ",".join(
                f"EXG Channel {index}" for index in range(8)) +
                ",Accel Channel 0,Timestamp,Timestamp (Formatted)\n")
            for sample in range(1000):
                stream.write(str(sample) + "," + ",".join(
                    f"{csv_signal[channel, sample]:.6f}"
                    for channel in range(8)) +
                    f",0,{1_700_000_000 + sample / 250.0:.6f},formatted\n")
        csv_loaded = load_openbci_gui_csv(csv_path)
        check("OpenBCI CSV metadata and exact eight EXG columns are recognized",
              csv_loaded["data"].shape == (8, 1000) and
              csv_loaded["fs"] == 250 and
              csv_loaded["channel_numbers"] == list(range(8)) and
              csv_loaded["rows_dropped"] == 0)
        csv_preflight = analyse_preflight(
            csv_loaded["data"], fs=csv_loaded["fs"], elapsed=4.0)
        check("OpenBCI CSV signal feeds the same adaptive preflight",
              csv_preflight["acceptable"])
        csv_backend = OpenBCICSVBackend(csv_path, loop=True, speed=100.0)
        csv_backend.start(); time.sleep(.02)
        csv_chunk = csv_backend.read_available(); csv_backend.stop()
        check("OpenBCI CSV backend replays channel-major µV samples",
              csv_chunk is not None and csv_chunk.shape[0] == 8 and
              csv_chunk.shape[1] > 0 and np.std(csv_chunk) > 0)

        timestamp_only_path = Path(csv_tmp_text) / "timestamp-only.csv"
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        timestamp_only_path.write_text("\n".join(
            line for line in lines if "Sample Rate" not in line),
            encoding="utf-8")
        timestamp_loaded = load_openbci_gui_csv(timestamp_only_path)
        check("OpenBCI CSV sample rate falls back to numeric timestamps",
              abs(timestamp_loaded["fs"] - 250.0) < .1 and
              timestamp_loaded["sample_rate_source"] == "timestamp")

        labelled_path = Path(csv_tmp_text) / "electrode-labelled.csv"
        electrode_labels = ["Fp1", "Fp2", "Fz", "Cz",
                            "FC1", "FC2", "C3", "C4"]
        offsets = np.asarray([
            171259.88, 197867.2, 114104.27, 161374.05,
            258360.95, 117286.94, 118994.92, 135717.16])
        with labelled_path.open("w", encoding="utf-8", newline="") as stream:
            stream.write("session_time," + ",".join(electrode_labels) + "\n")
            for sample in range(1000):
                values = offsets + csv_signal[:, sample]
                stream.write(f"{1_791_882_133 + sample / 250.0:.6f}," +
                             ",".join(f"{value:.6f}" for value in values) + "\n")
        labelled_loaded = load_openbci_gui_csv(labelled_path)
        labelled_preflight = analyse_preflight(
            labelled_loaded["data"], fs=labelled_loaded["fs"], elapsed=4.0,
            channel_labels=labelled_loaded["channel_labels"])
        check("session_time plus standard electrode-labelled CSV is accepted",
              labelled_loaded["column_schema"] == "electrode_labels" and
              labelled_loaded["channel_labels"] == electrode_labels and
              abs(labelled_loaded["fs"] - 250.0) < .1 and
              labelled_preflight["acceptable"])

    # --------------------------------------------------------------
    # from test_step5_stimulus.py
    # --------------------------------------------------------------
    import os, sys
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from pathlib import Path
    import numpy as np
    import pygame; pygame.init(); pygame.display.set_mode((200,200))
    from stimulus import (FlickerPlan, StimulusEngine, open_display,
                          suggest_frequencies, detect_refresh_rate)  # noqa

    print("\n== The bug: naive half-period rounding ==")
    def naive(f, refresh=60):
        hp = max(1, round(refresh/(2*f)))
        return refresh/(2*hp)
    check("naive rounding collapses 20Hz onto 15Hz @60Hz", naive(20)==15.0 and naive(15)==15.0)
    p15 = FlickerPlan(15.0, 60); p20 = FlickerPlan(20.0, 60)
    check("period-quantised 15Hz is exact", p15.actual==15.0 and p15.is_exact)
    check("period-quantised 20Hz is exact", p20.actual==20.0 and p20.is_exact)
    check("20Hz uses an asymmetric 3-frame period",
          p20.period_frames==3 and p20.on_frames==2 and not p20.is_symmetric)
    check("15/20 no longer collide", p15.period_frames != p20.period_frames)

    print("\n== FlickerPlan frame patterns ==")
    seq15 = [p15.is_on(f) for f in range(8)]
    seq20 = [p20.is_on(f) for f in range(6)]
    check(f"15Hz pattern 2on/2off {seq15[:4]}", seq15[:4]==[True,True,False,False])
    check(f"20Hz pattern 2on/1off {seq20[:3]}", seq20[:3]==[True,True,False])
    check("patterns repeat exactly by period",
          all(p15.is_on(f)==p15.is_on(f+4) for f in range(40)) and
          all(p20.is_on(f)==p20.is_on(f+3) for f in range(40)))
    check("duty cycles reported", abs(p15.duty-0.5)<1e-9 and abs(p20.duty-2/3)<1e-9)
    po = FlickerPlan(15.0, 60, phase_offset=2)
    check("phase offset shifts the pattern", po.is_on(0) != p15.is_on(0))

    print("\n== FFT proves the rendered fundamental ==")
    eng = StimulusEngine((15.0,20.0), 60.0)
    for i, want in enumerate((15.0, 20.0)):
        r = eng.spectrum_check(i, n_frames=1200)
        check(f"target {i}: FFT peak {r['fft_peak']:.2f}Hz == {want}Hz",
              r["matches"] and abs(r["fft_peak"]-want) < 0.1)

    print("\n== Non-integer refresh rates are reported honestly ==")
    e144 = StimulusEngine((15.0,20.0), 144.0)
    check("144Hz: 15Hz is NOT exact and says so", not e144.plans[0].is_exact)
    check("144Hz: reports actual 14.40Hz", abs(e144.plans[0].actual-14.4)<0.01)
    check("144Hz: error percentage computed", abs(e144.plans[0].error_pct+4.0)<0.1)
    e75 = StimulusEngine((15.0,20.0), 75.0)
    check("75Hz: 15Hz exact, 20Hz becomes 18.75Hz",
          e75.plans[0].is_exact and abs(e75.plans[1].actual-18.75)<0.01)

    print("\n== suggest_frequencies gives exactly-renderable targets ==")
    s60 = suggest_frequencies(60.0)
    check(f"60Hz suggestions {s60}", 15.0 in s60 and 20.0 in s60 and 30.0 in s60)
    check("all suggestions are exactly renderable @60",
          all(FlickerPlan(f,60).is_exact for f in s60))
    check("suggestions avoid the alpha band", all(f > 13.0 for f in s60))
    s144 = suggest_frequencies(144.0)
    # suggestions are rounded to 4dp for display, so allow sub-0.01Hz error
    check(f"144Hz suggestions are renderable {s144}",
          all(abs(FlickerPlan(f,144).error_hz) < 0.01 for f in s144))
    check("144Hz suggestions differ from 60Hz ones", set(s144) != set(s60))

    print("\n== V-synced display opens and stays blit-compatible ==")
    surf, info = open_display((320,240), refresh_hint=60.0)
    check("display opened", surf is not None)
    check("uses SCALED (not OPENGL)", "SCALED" in info["flags"] or info["flags"]=="DOUBLEBUF")
    tmp = pygame.Surface((10,10)); tmp.fill((255,0,0))
    try:
        surf.blit(tmp,(0,0)); blit_ok=True
    except Exception:
        blit_ok=False
    check("blit works (OPENGL would break this)", blit_ok)
    check("refresh reported", info["refresh"] > 0)

    print("\n== Frame counter & timing audit ==")
    eng2 = StimulusEngine((15.0,20.0), 60.0)
    for _ in range(200):
        pygame.display.flip(); eng2.tick()
    check("frame counter advances", eng2.frame==200)
    r = eng2.timing_report()
    check("timing report produced", r is not None and r["frames"]==200)
    check("reports jitter and drops", "jitter_pct" in r and "dropped" in r)
    eng2.reset()
    check("reset clears state", eng2.frame==0 and eng2.timing_report() is None)

    print("\n== Collision detection ==")
    ecol = StimulusEngine((20.0, 20.5), 60.0)   # both round to 3 frames
    periods = {p.period_frames for p in ecol.plans}
    check("detects two targets sharing a frame period", len(periods)==1)

    print("\n== Speller integration ==")
    from example.bci_speller import Speller  # noqa
    app = Speller(source="keyboard", flicker=True, refresh=60.0)
    check("speller opens a vsync display", app.display_info["refresh"]==60.0)
    check("speller exposes ACTUAL rendered freqs", app.actual_freqs==(15.0,20.0))
    before = [t.flicker_on for row in app.tiles for t in row]
    for _ in range(3):
        app._refresh_states(); app._apply_flicker(); app.draw()
    after = [t.flicker_on for row in app.tiles for t in row]
    check("tiles actually toggle", before != after)
    check("frame counter ticks with draw()", app.stim.frame >= 3)
    check("tiles carry their flicker frequency",
          all(t.freq in app.actual_freqs for row in app.tiles for t in row))

    print("\n== Classifier tracks ACTUAL, not requested, frequency ==")
    app2 = Speller(source="keyboard", flicker=True, refresh=144.0)
    check("144Hz speller reports shifted freqs",
          abs(app2.actual_freqs[0]-14.4)<0.01)
    from ML.bci_engine import BCIEngine  # noqa
    eng3 = BCIEngine(mode="SSVEP", target_freqs=app2.actual_freqs)
    check("engine reference signals match the rendered frequency",
          eng3.ssvep_clf.target_freqs == list(app2.actual_freqs))

    pygame.quit()
    layer_total('LAYER 1 - ACQUISITION AND SIGNAL QUALITY')


def layer2():
    global check
    banner('LAYER 2 - CALIBRATION AND SEPARABILITY', 'Is there a measurable difference between the two intentions, before any\nclassifier is involved, and can a per-user threshold be fitted?')

    # --------------------------------------------------------------
    # from test_step3_calibration.py
    # --------------------------------------------------------------
    import glob, json, sys, tempfile
    from pathlib import Path
    import numpy as np

    from ML.hybrid_classifier import HybridSSVEPClassifier, NO_ACTION  # noqa
    from ML.bci_engine import BCIEngine  # noqa
    from ML import calibration  # noqa
    import importlib.util  # noqa
    spec = importlib.util.spec_from_file_location("calrec", ROOT/"EEG"/"calibration_record.py")
    calrec = importlib.util.module_from_spec(spec); spec.loader.exec_module(calrec)

    print("\n== Protocol design ==")
    names = [b for b,_,_,_,_ in calrec.DEFAULT_PROTOCOL]
    check("has both SSVEP target blocks", "target_0" in names and "target_1" in names)
    check("has an idle block", "idle" in names)
    check("has a distraction block", "idle_distracted" in names)
    check("has an artifact/blink block", "artifact" in names)
    idle_s = sum(r*s for b,_,r,s,_ in calrec.DEFAULT_PROTOCOL if b.startswith("idle"))
    tgt_s  = sum(r*s for b,_,r,s,_ in calrec.DEFAULT_PROTOCOL if b.startswith("target"))
    check(f"idle time ({idle_s:.0f}s) exceeds target time ({tgt_s:.0f}s) - realistic",
          idle_s > tgt_s)
    total = sum(r*s for _,_,r,s,_ in calrec.DEFAULT_PROTOCOL)/60
    check(f"protocol is a reasonable length ({total:.1f} min)", 3 <= total <= 40)

    print("\n== Simulate backend ==")
    be = calrec.SimulateBackend(seed=0)
    be.start(); be.set_state("target_0", 0)
    w = be.read(750)
    check("emits (8,750) uV windows", w.shape == (8,750))
    be.set_state("artifact", 2); wa = be.read(750)
    check("blink block produces >100uV frontal peaks", np.max(np.abs(wa[[0,1]])) > 100)
    be.set_state("idle", -1); wi = be.read(750)
    check("idle block stays low-voltage frontally", np.max(np.abs(wi[[0,1]])) < 100)

    print("\n== Artifact-band regression (bug found by Step 3) ==")
    clf = HybridSSVEPClassifier()
    check("blink caught on RAW data", clf.artifact_detection(wa, is_raw=True))
    check("blink NOT caught after SSVEP highpass (why the bug existed)",
          not clf.artifact_detection(clf.apply_filter(wa), is_raw=False))
    lab,_,_ = clf.predict_proba(wa)
    check("predict_proba returns BLINK for a blink window", lab == "BLINK")
    check("idle window is not flagged as artifact", not clf.artifact_detection(wi, is_raw=True))

    print("\n== Recording -> analysis -> profile round trip ==")
    be2 = calrec.SimulateBackend(seed=7); be2.start()
    quick = [(b,c,1,min(s,6.0),i) for b,c,r,s,i in calrec.DEFAULT_PROTOCOL]
    X,y,labels,aborted = calrec.run_protocol(be2, None, quick, 250, 750, 250, verbose=False)
    check("recorded windows", len(X) > 0 and not aborted)
    check("labels cover all blocks", set(labels) == set(names))
    tmp = Path(tempfile.mkdtemp())
    saved_acquisition = {"route": "synthetic", "backend": "simulate"}
    p = calrec.save(
        X, y, labels, "TEST", 250, "cyton8_ssvep", [15.0, 20.0], tmp,
        acquisition=saved_acquisition)
    check("npz saved", p.exists())
    with np.load(p, allow_pickle=False) as saved_npz:
        acquisition_round_trip = json.loads(str(
            np.asarray(saved_npz["acquisition_json"]).item()))
    check("recording NPZ retains acquisition-route provenance",
          acquisition_round_trip == saved_acquisition)
    Xl,yl,ll,fs,mont,tf = calibration.load(p)
    check("round-trips shape", Xl.shape == np.asarray(X,dtype=np.float32).shape)
    check("round-trips montage/fs", mont=="cyton8_ssvep" and fs==250)

    print("\n== Threshold fitting from the user's own idle data ==")
    prof = calibration.analyse(str(p), percentile=95.0, out=tmp/"prof.json")
    check("profile produced", prof is not None)
    check("threshold is subject-fitted (not the 0.15 default)",
          prof["confidence_threshold"] != 0.15)
    check("profile records idle minutes", prof["idle_minutes"] > 0)
    check("json written", (tmp/"prof.json").exists())

    print("\n== Engine loads the profile ==")
    eng = BCIEngine.from_profile(tmp/"prof.json")
    check("engine threshold matches profile",
          abs(eng.ssvep_clf.confidence_threshold - prof["confidence_threshold"]) < 1e-9)
    check("engine targets match profile", eng.ssvep_clf.target_freqs == [15.0,20.0])
    be3 = calrec.SimulateBackend(seed=11); be3.start(); be3.set_state("idle",-1)
    res = [eng.process_frame(be3.read(750)) for _ in range(15)]
    check("calibrated engine emits no commands while idle", not any(t for _,t in res))

    print("\n== ITR metric ==")
    check("ITR zero at chance", calibration.itr_bits_per_min(2, 0.5, 10) == 0.0)
    check("ITR positive when accurate", calibration.itr_bits_per_min(2, 0.95, 10) > 5)
    layer_total('LAYER 2 - CALIBRATION AND SEPARABILITY')


def layer3():
    global check
    banner('LAYER 3 - OFFLINE MODEL', 'Does a model trained on that data generalise, under leakage-safe CV?')

    # --------------------------------------------------------------
    # from test_step6_models.py
    # --------------------------------------------------------------
    import sys
    from pathlib import Path
    import numpy as np
    from ML.trca import trca_weights, corr2, TRCAClassifier, HybridTRCAClassifier  # noqa
    from ML.hybrid_classifier import NO_ACTION  # noqa
    from ML.mi_advanced import (CSP, FilterBankCSP, TangentSpaceMapper,
                                riemannian_mean, PIPELINES, _cov)  # noqa
    from ML.evaluate import infer_groups, subdivide_groups, grouped_folds  # noqa

    FS,N=250,750
    rng=np.random.default_rng(0)

    def ssvep_trials(freq,n,n_ch=6,occ=(2,3,4,5),snr=1.0,mix=None,phase=0.4):
        if mix is None: mix=np.eye(n_ch)
        out=[]
        for _ in range(n):
            t=np.arange(N)/FS; x=rng.standard_normal((n_ch,N))*2.0
            for ch in range(n_ch): x[ch]+=3.0*np.sin(2*np.pi*10.2*t+rng.uniform(0,6.28))
            for ch in occ:
                for h,a in ((1,1.0),(2,0.4)): x[ch]+=snr*a*np.sin(2*np.pi*h*freq*t+phase)
            out.append(mix@x)
        return np.array(out)

    print("\n== TRCA core ==")
    tr_in = ssvep_trials(15.0, 12)
    w = trca_weights(tr_in)
    check("returns one weight per channel", w.shape == (6,))
    check("filter is unit-norm", abs(np.linalg.norm(w)-1.0) < 1e-9)
    try:
        trca_weights(tr_in[:1]); check("rejects <2 trials", False)
    except ValueError: check("rejects <2 trials", True)
    try:
        trca_weights(np.zeros((5,6))); check("rejects wrong ndim", False)
    except ValueError: check("rejects wrong ndim", True)
    check("corr2 self-correlation is 1", abs(corr2(np.arange(10),np.arange(10))-1)<1e-9)
    check("corr2 anti-correlation is -1", abs(corr2(np.arange(10),-np.arange(10))+1)<1e-9)
    check("corr2 handles constant input", corr2(np.ones(10), np.arange(10)) == 0.0)

    print("\n== TRCA classifier ==")
    mix = np.eye(6)+0.4*rng.standard_normal((6,6))
    X = np.concatenate([ssvep_trials(15.0,20,mix=mix), ssvep_trials(20.0,20,mix=mix)])
    y = np.array([0]*20+[1]*20)
    clf = TRCAClassifier(target_freqs=(15.0,20.0), sample_freq=FS)
    clf.fit(X,y)
    check("fit marks fitted", clf.is_fitted and clf.n_train_==40)
    check("templates per band and class", len(clf.templates_)==3 and len(clf.templates_[0])==2)
    Xte = np.concatenate([ssvep_trials(15.0,15,mix=mix), ssvep_trials(20.0,15,mix=mix)])
    yte = np.array([0]*15+[1]*15)
    pred = np.array([int(np.argmax(clf.scores(x))) for x in Xte])
    acc = np.mean(pred==yte)
    check(f"held-out accuracy {acc:.0%} > chance", acc > 0.7)
    lab,conf,sc = clf.predict_proba(Xte[0])
    check("predict_proba returns triple", len(sc)==2 and isinstance(conf,float))
    clf.confidence_threshold = 99.0
    check("low confidence -> NO_ACTION", clf.predict_proba(Xte[0])[0] == NO_ACTION)
    clf.confidence_threshold = 0.2
    try:
        TRCAClassifier().fit(X[:1], y[:1]); check("errors clearly on too-few trials", False)
    except ValueError as e: check("errors clearly on too-few trials", "TRCA needs" in str(e))
    idle = ssvep_trials(15.0, 10, snr=0.0, mix=mix)
    t = clf.calibrate_threshold(idle, percentile=95)
    check("calibrate_threshold from idle", t > 0)

    print("\n== HybridTRCA falls back safely ==")
    h = HybridTRCAClassifier(target_freqs=(15.0,20.0), sample_freq=FS)
    check("untrained -> uses FBCCA", h.active == "fbcca")
    w8 = np.zeros((8,N)); w8[6]=np.sin(2*np.pi*15*np.arange(N)/FS)*8
    lab,_,_ = h.predict_proba(w8)
    check("works untrained", lab in (0,1,NO_ACTION,"BLINK"))
    X8 = np.concatenate([ssvep_trials(15.0,20,n_ch=8,occ=(6,7)),
                         ssvep_trials(20.0,20,n_ch=8,occ=(6,7))])
    h.fit(X8, y)
    check("trained -> switches to TRCA", h.active == "trca")
    h.prefer="fbcca"
    check("prefer overrides", h.active=="fbcca")
    blink = np.zeros((8,N)); blink[0,300:340]=400.0
    check("blink still detected", h.predict_proba(blink)[0]=="BLINK")

    print("\n== MI: CSP ==")
    def mi_trials(n, erd_ch, n_ch=8):
        out=[]
        for _ in range(n):
            x=rng.standard_normal((n_ch,N))
            for ch in range(n_ch): x[ch]*=1.0
            x[erd_ch]*=0.4                       # band-power reduction
            out.append(x)
        return np.array(out)
    Xm=np.concatenate([mi_trials(30,2), mi_trials(30,4)]); ym=np.array([0]*30+[1]*30)
    csp=CSP(n_components=4).fit(Xm,ym)
    F=csp.transform(Xm)
    check("CSP outputs n_components features", F.shape==(60,4))
    check("CSP features are finite", np.all(np.isfinite(F)))
    try:
        CSP().fit(Xm, np.array([0]*20+[1]*20+[2]*20)); check("CSP rejects 3 classes", False)
    except ValueError: check("CSP rejects 3 classes", True)
    fb=FilterBankCSP(sample_freq=FS).fit(Xm,ym)
    Ffb=fb.transform(Xm)
    check(f"FBCSP concatenates bands ({Ffb.shape[1]} features)", Ffb.shape[1]==6*4)

    print("\n== MI: Riemannian geometry ==")
    covs=np.array([_cov(x) for x in Xm[:20]])
    M=riemannian_mean(covs)
    check("Riemannian mean is symmetric", np.allclose(M,M.T,atol=1e-8))
    check("Riemannian mean is positive definite", np.all(np.linalg.eigvalsh(M)>0))
    single=riemannian_mean(covs[:1])
    check("mean of one matrix is itself", np.allclose(single,covs[0],atol=1e-6))
    ts=TangentSpaceMapper(sample_freq=FS).fit(Xm)
    T=ts.transform(Xm)
    check(f"tangent space dim = n(n+1)/2 = 36 ({T.shape[1]})", T.shape[1]==36)
    check("tangent features finite", np.all(np.isfinite(T)))

    print("\n== MI pipelines run end to end ==")
    for name,fac in PIPELINES.items():
        try:
            p=fac(sample_freq=FS); p.fit(Xm,ym)
            a=np.mean(p.predict(Xm)==ym)
            check(f"{name} fits and predicts (train acc {a:.0%})", 0.0<=a<=1.0)
        except Exception as e:
            check(f"{name} fits and predicts", False); print("      ",e)

    print("\n== Leakage-safe grouping (the important part) ==")
    labels=np.array(["target_0"]*9+["target_1"]*9+["idle"]*6)
    g=infer_groups(labels)
    check("groups follow label blocks", len(np.unique(g))==3)
    yb=np.array([0]*9+[1]*9+[-1]*6)
    sub,keep=subdivide_groups(g[:18],yb[:18],n_sub=3)
    check("subdivides into more groups", len(np.unique(sub))==6)
    check("drops overlapping seam windows", (~keep).sum()>0)
    gg=np.repeat(np.arange(8),5); yy=np.repeat([0,1],20)
    seen_test=[]
    for tr,te in grouped_folds(gg,yy,n_splits=4):
        check_overlap = set(gg[tr]) & set(gg[te])
        if check_overlap: seen_test.append(check_overlap)
    check("no group appears in both train and test", not seen_test)
    folds=list(grouped_folds(gg,yy,n_splits=4))
    check(f"produced {len(folds)} usable folds", len(folds)>=3)
    check("every fold has both classes in test",
          all(len(np.unique(yy[te]))==2 for _,te in folds))

    # --------------------------------------------------------------
    # from test_step8_datasets_models.py
    # --------------------------------------------------------------
    import os, sys, tempfile
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from pathlib import Path
    import numpy as np
    import bci_sdk
    from ML.datasets import (load_calibration_npz, pick_channels, resample_to,
                             crop_or_pad, combine, load_benchmark_mat)
    from ML.trca import TRCAClassifier
    from ML.hybrid_classifier import HybridSSVEPClassifier

    print("\n== Dataset loaders ==")
    X,y,meta = load_calibration_npz(str(ROOT/"dataset"/"calib_*.npz"))
    check("own recording loads (trials,ch,samples)", X.ndim==3 and X.shape[1:]==(8,750))
    check("labels balanced", set(np.unique(y))=={0,1})
    check("groups exposed for leakage-safe CV", len(np.unique(meta["groups"]))>=2)
    check("idle windows exposed", meta["idle"].shape[0]>0)
    check("artifact windows exposed", meta["artifact"].shape[0]>0)
    check("montage + freqs in meta", meta["montage"]=="cyton8_ssvep" and len(meta["target_freqs"])==2)

    check("pick_channels 64->8", pick_channels(np.zeros((4,64,750)),[53,54,55,56,57,60,61,62]).shape==(4,8,750))
    check("resample 1000->250", resample_to(np.zeros((2,8,1000)),1000,250).shape==(2,8,250))
    check("crop_or_pad up", crop_or_pad(np.zeros((2,8,600)),750).shape==(2,8,750))
    check("crop_or_pad down", crop_or_pad(np.zeros((2,8,900)),750).shape==(2,8,750))
    try:
        pick_channels(np.zeros((2,8,750)),[53]); check("pick_channels guards range", False)
    except ValueError: check("pick_channels guards range", True)

    Xc,yc,src = combine((np.zeros((5,8,750)),[0]*5),(np.ones((3,8,750)),[1]*3))
    check("legacy combine joins already compatible arrays",
          Xc.shape==(8,8,750) and list(np.bincount(src))==[5,3])
    try:
        combine((np.zeros((2,8,750)),[0]*2),(np.zeros((2,64,750)),[1]*2))
        check("combine rejects shape mismatch", False)
    except ValueError: check("combine rejects shape mismatch", True)
    try:
        load_benchmark_mat("/nope/*.mat",freq_map={10:0}); check("missing .mat guarded", False)
    except FileNotFoundError: check("missing .mat guarded", True)
    try:
        load_benchmark_mat(str(ROOT/"dataset"/"calib_*.npz")); check("freq_map required", False)
    except (ValueError,Exception): check("freq_map required", True)

    print("\n== Model persistence ==")
    fb=HybridSSVEPClassifier(target_freqs=meta["target_freqs"],montage=meta["montage"])
    Xf=np.array([fb.apply_filter(x) for x in X])
    t=TRCAClassifier(target_freqs=tuple(meta["target_freqs"]),channels=fb.ssvep_channels)
    t.fit(Xf,y)
    tmp=Path(tempfile.mkdtemp())/"m.joblib"
    t.save(tmp)
    check("save() writes a file", tmp.exists())
    t2=TRCAClassifier.load(tmp)
    check("load() restores fitted state", t2.is_fitted)
    a=[int(np.argmax(t.scores(x))) for x in Xf]
    b=[int(np.argmax(t2.scores(x))) for x in Xf]
    check("round-trip predictions identical", a==b)
    check("hyper-params preserved",
          t2.target_freqs==t.target_freqs and t2.channels==t.channels
          and t2.ensemble==t.ensemble)
    try:
        TRCAClassifier().save(tmp); check("refuses to save unfitted", False)
    except RuntimeError: check("refuses to save unfitted", True)
    bad=Path(tempfile.mkdtemp())/"bad.joblib"
    import joblib; joblib.dump({"kind":"something_else"},bad)
    try:
        TRCAClassifier.load(bad); check("rejects foreign model file", False)
    except ValueError: check("rejects foreign model file", True)

    print("\n== Session uses a trained model ==")
    t.calibrate_threshold([fb.apply_filter(w) for w in meta["idle"][:40]])
    t.save(tmp)
    s=bci_sdk.BCISession(source="sim",model=str(tmp))
    check("session accepts model=", s.model is not None)
    check("engine classifier was swapped", "TRCA" in s.report()["classifier"])
    check("report records model path", s.report()["model"]==str(tmp))
    import time
    s.start(); e=time.time()+5
    while time.time()<e: s.poll(); time.sleep(1/60)
    s.stop()
    check("runs without error using trained model", s.bridge.report()["windows_made"]>=1)
    try:
        bci_sdk.BCISession(source="sim",model=str(tmp),target_freqs=(10.,12.))
        check("frequency mismatch raises", False)
    except bci_sdk.CalibrationError: check("frequency mismatch raises", True)
    try:
        bci_sdk.BCISession(source="sim",model="/nonexistent/m.joblib")
        check("missing model raises CalibrationError", False)
    except bci_sdk.CalibrationError: check("missing model raises CalibrationError", True)
    s2=bci_sdk.BCISession(source="sim")
    check("no model -> FBCCA fallback", "TRCA" not in s2.report()["classifier"])

    print("\n== SDK exports ==")
    for n in ["load_benchmark_mat","load_physionet_edf","load_calibration_npz",
              "pick_channels","combine","datasets"]:
        check(f"exports {n}", hasattr(bci_sdk,n))

    print("\n== Experimental direct-button MI is isolated and multi-class ==")
    from ML.button_mi import (make_button_label, parse_button_label,
                              button_profile_dir, ButtonMIModel)
    from ML.train_button_mi import train_button_user
    button_labels = [make_button_label(i, name)
                     for i, name in enumerate(("rock", "paper", "scissors"))]
    check("button label round-trip preserves screen index and name",
          parse_button_label(button_labels[2]) == (2, "scissors"))

    from EEG.data_record import (CuedRecorder, epoch_from_log,
                                 epoch_button_mi_from_log,
                                 epoch_intent_mi_from_log)
    probe_stream = np.zeros((8, 1200))
    direct_log = [{"label": button_labels[1], "button_index": 1,
                   "cue_sample": 20, "cue_start_ms": 0}]
    bX, by, bl = epoch_button_mi_from_log(probe_stream, 250, direct_log, 0)
    hX, hy, hl = epoch_from_log(probe_stream, 250, direct_log, 0)
    check("direct recorder writes a 3-second button-labelled epoch",
          bX.shape == (1, 8, 750) and by.tolist() == [1] and
          bl.tolist() == [button_labels[1]])
    check("existing LEFT/RIGHT epocher ignores direct-button labels",
          len(hX) == len(hy) == len(hl) == 0)
    intent_log = [{"label": "intent::right_index", "class_index": 1,
                   "cue_sample": 20, "cue_start_ms": 0}]
    iX, iy2, il2 = epoch_intent_mi_from_log(probe_stream, 250, intent_log, 0)
    check("reusable intent epocher writes physical labels, not button names",
          iX.shape == (1, 8, 750) and iy2.tolist() == [1] and
          il2.tolist() == ["intent::right_index"])

    class RecorderClock:
        def __init__(self): self.now = 0.0
        def __call__(self): return self.now
        def advance(self, ms): self.now += ms

    class RecorderButton:
        def __init__(self, index, name):
            self.id, self.name, self.is_flashing = index, name, False
        def flash(self, **kwargs): self.is_flashing = True
        def stop_flash(self): self.is_flashing = False

    recorder_clock = RecorderClock()
    recorder = CuedRecorder(
        None, trials=3, cue_s=4, rest_s=10, button_mi=True,
        clock=recorder_clock)
    recorder.buttons = [RecorderButton(i, name) for i, name in enumerate(
        ("rock", "paper", "scissors"))]
    recorder.update(900000)       # pygame tick before a display reset
    recorder_clock.advance(10001)
    recorder.update(0)            # pygame tick restarted, monotonic time did not
    check("10-second REST survives a pygame timer reset and starts trial 1",
          recorder.phase == recorder.CUE and recorder.trial == 1)

    with tempfile.TemporaryDirectory() as button_tmp_text:
        button_root = Path(button_tmp_text)
        button_dir = button_profile_dir("demo", "rockpaperscissors",
                                        root=button_root)
        button_dir.mkdir(parents=True)
        brng = np.random.default_rng(123)
        # Three complete sessions. Each class has a distinct imagined-action
        # covariance (a strong synthetic ERD only for software verification).
        for session in range(3):
            bx, by, bl = [], [], []
            for cls in range(3):
                for _ in range(10):
                    epoch = brng.normal(0, 5.0, (8, 750))
                    epoch[2 + cls] *= 0.18
                    bx.append(epoch); by.append(cls); bl.append(button_labels[cls])
            order = brng.permutation(len(by))
            np.savez_compressed(
                button_dir / f"calib_demo_session{session}.npz",
                X=np.asarray(bx)[order], y=np.asarray(by)[order],
                labels=np.asarray(bl)[order], fs=250,
                montage="cyton8_motor")

        button_result = train_button_user(
            "demo", "rockpaperscissors", root=button_root,
            n_splits=3, screen=False, verbose=False)
        check("three-class button trainer produces its separate model",
              button_result is not None and
              (button_dir / "button_model.joblib").exists())
        check("button model never overwrites LEFT/RIGHT model.joblib",
              not (button_root / "demo" / "model.joblib").exists())
        check("held-out-session button accuracy is above 3-class chance",
              button_result["accuracy"] is not None and
              button_result["accuracy"] > 0.70)
        button_model = ButtonMIModel.load(
            "demo", "rockpaperscissors", root=button_root,
            confidence_threshold=0.0)
        command, button_conf = button_model.predict_command(np.asarray(bx)[0])
        check("loaded button model emits an actual button name directly",
              command in {"rock", "paper", "scissors"} and button_conf > 0)
        check("button model records app-specific three-class chance",
              abs(button_model.describe()["chance_accuracy"] - 1/3) < 1e-12)

    print("\n== Reusable scalable motor-intent profiles ==")
    from ML.motor_intent import (CategoryMotorIntentRouter, MotorIntentMap,
        MotorIntentModel, FINGER_INTENTS, UNIVERSAL_CATEGORY_INTENTS,
        UNIVERSAL_INTENTS,
        assess_intent_set, category_profile_dir, general_recording_dir,
        intent_profile_dir,
        make_intent_label, parse_intent_map, task_set_key)
    from ML.train_intent_mi import (load_intent_data, train_intent_user,
                                    train_universal_category_models)

    default_map = MotorIntentMap({
        "rock": "left_index", "paper": "right_index",
        "scissors": "right_pinky"})
    check("RPS default maps actions to physical finger labels",
          default_map.intents == ("left_index", "right_index", "right_pinky"))
    check("task-set key is mapping-order independent",
          task_set_key(default_map.intents) ==
          task_set_key(reversed(default_map.intents)))
    parsed_map = parse_intent_map(
        "rock=left_index,paper=right_index,scissors=right_pinky")
    check("CLI action=intent mapping parses",
          parsed_map == default_map.action_to_intent)
    adjacent = assess_intent_set(
        ["right_thumb", "right_index", "left_pinky"])
    check("adjacent fingers produce an accuracy warning",
          adjacent["adjacent_pairs"] == [("right_thumb", "right_index")] and
          any("adjacent" in w for w in adjacent["warnings"]))
    ten = assess_intent_set(FINGER_INTENTS)
    check("all ten fingers are accepted but marked research-only",
          ten["count"] == 10 and ten["difficulty"] == "RESEARCH_ONLY" and
          len(ten["warnings"]) >= 2)
    check("universal recording vocabulary is ten fingers plus 3 body tasks",
          len(UNIVERSAL_INTENTS) == 13 and
          set(UNIVERSAL_INTENTS[-3:]) == {"left_hand", "right_hand", "both_feet"})
    check("universal labels map into separate finger, hand and feet models",
          set(UNIVERSAL_CATEGORY_INTENTS) == {"finger", "hand", "feet"} and
          UNIVERSAL_CATEGORY_INTENTS["finger"] == FINGER_INTENTS and
          UNIVERSAL_CATEGORY_INTENTS["hand"] == ("left_hand", "right_hand") and
          UNIVERSAL_CATEGORY_INTENTS["feet"] == ("both_feet",))

    class StubCategoryModel:
        sample_freq = 250
        montage = "cyton8_motor"

        def __init__(self, result):
            self.result = result

        def predict_command(self, _window):
            return self.result

        def describe(self):
            return {"result": self.result}

    route_probe = CategoryMotorIntentRouter(
        "demo", ["left_hand", "right_hand", "both_feet"],
        {"hand": StubCategoryModel(("left_hand", .61)),
         "feet": StubCategoryModel(("both_feet", .82))})
    check("mixed-category router selects the highest accepted specialist output",
          route_probe.predict_command(np.zeros((8, 750))) == ("both_feet", .82) and
          route_probe.last_category == "feet")

    print("\n== Public MI registry, semantics and leakage-safe archives ==")
    from ML.motor_intent import (MI_CATEGORIES, UPPER_LIMB_INTENTS,
        category_for_intents, intent_category)
    from ML.public_mi import (CYTON_MOTOR_CHANNELS, PUBLIC_DATA_FORMATS,
        PUBLIC_MI_DATASETS, _validate_raw_format, get_public_dataset,
        load_standardized_archive, moabb_motor_imagery_options,
        physionet_imagery_label_map, public_archive_path, public_channel_plan,
        public_native_sfreqs, read_ds008446_events, subject_folds,
        write_standardized_archive)
    from ML.train_public_mi import (
        PUBLIC_PIPELINES, apply_public_training_policy,
        train_public_category)
    check("four specialist categories are explicit",
          MI_CATEGORIES == ("finger", "hand", "upper_limb", "feet"))
    check("runtime intents map to one intended category",
          intent_category("left_index") == "finger" and
          intent_category("left_hand") == "hand" and
          intent_category("right_elbow_flexion") == "upper_limb" and
          intent_category("both_feet") == "feet")
    check("Ofner upper-limb vocabulary keeps exact movement semantics",
          {"right_elbow_flexion", "right_forearm_pronation",
           "right_hand_open"}.issubset(UPPER_LIMB_INTENTS))
    try:
        category_for_intents(["left_hand", "both_feet"])
        check("category specialists reject mixed hand/feet task sets", False)
    except ValueError:
        check("category specialists reject mixed hand/feet task sets", True)
    finger_registry = get_public_dataset("openneuro_ds008446")
    check("finger public labels remain explicitly unsided",
          set(finger_registry["source_label_map"].values()) == {
              "finger_thumb_unsided", "finger_index_unsided",
              "finger_middle_unsided", "finger_ring_unsided",
              "finger_pinky_unsided"})
    with tempfile.TemporaryDirectory() as openneuro_tmp_text:
        openneuro_tmp = Path(openneuro_tmp_text)
        eeg_path = openneuro_tmp / "sub-01_task-random_run-01_eeg.edf"
        eeg_path.touch()
        events_path = openneuro_tmp / "sub-01_task-random_run-01_events.tsv"
        fs = 512.0
        event_rows = ["onset\tduration\ttrial_type"]
        for sample in range(6144):
            if sample < 1536:
                code = 1
            elif sample < 3072:
                code = 3
            elif sample < 4608:
                code = 2
            else:
                code = 7
            event_rows.append(f"{sample / fs**2:.12f}\t0\t{code}")
        events_path.write_text("\n".join(event_rows) + "\n", encoding="utf-8")
        reconstructed = read_ds008446_events(
            eeg_path, fs, 7000, finger_registry["source_label_map"])
        check("ds008446 repeated per-sample labels collapse to cue blocks",
              reconstructed.tolist() ==
              [[1536, 0, 3], [4608, 0, 7]])
        check("ds008446 onsets use the release-specific squared-rate scale",
              reconstructed[0, 0] ==
              round((1536 / fs**2) * fs**2) and
              reconstructed[0, 0] != round((1536 / fs**2) * fs))

        events_path.write_text(
            "onset\tduration\ttrial_type\n"
            f"0\t0\t1\n{8000 / fs**2:.12f}\t0\t3\n",
            encoding="utf-8")
        try:
            read_ds008446_events(
                eeg_path, fs, 7000, finger_registry["source_label_map"])
            check("ds008446 out-of-range cues fail with file diagnostics", False)
        except ValueError as exc:
            diagnostic = str(exc)
            check("ds008446 out-of-range cues fail with file diagnostics",
                  eeg_path.name in diagnostic and
                  "sample=8000" in diagnostic and "0..6999" in diagnostic)
    bnci_events = PUBLIC_MI_DATASETS["bnci2014_001"]["source_label_map"]
    bnci_options = moabb_motor_imagery_options(bnci_events)
    check("MOABB receives explicit classes and the 1-40 Hz import passband",
          bnci_options["events"] == list(bnci_events) and
          bnci_options["n_classes"] == len(bnci_events) == 3 and
          bnci_options["fmin"] == 1.0 and bnci_options["fmax"] == 40.0)
    automated_format_ids = {
        "openneuro_ds008446", "physionet_eegmmidb", "bnci2014_001",
        "ofner2017", "cho2017", "lee2019_mi"}
    check("every automated public dataset has a verified format contract",
          set(PUBLIC_DATA_FORMATS) == automated_format_ids and all(
              {event.rsplit(":", 1)[-1] for event in
               PUBLIC_MI_DATASETS[value]["source_label_map"]}.issubset(
                  PUBLIC_DATA_FORMATS[value]["source_events"])
              for value in automated_format_ids))

    class PhysionetFormatProbe:
        def __init__(self, sfreq):
            self.info = {"sfreq": sfreq}
            self.ch_names = list(CYTON_MOTOR_CHANNELS) + [
                f"EEG{index:02d}" for index in range(56)]

        def get_channel_types(self):
            return ["eeg"] * 64

    physio_128 = PhysionetFormatProbe(128.0)
    try:
        _validate_raw_format(
            "physionet_eegmmidb", physio_128,
            Path("S088R04.edf"), CYTON_MOTOR_CHANNELS)
        known_128_accepted = True
    except ValueError:
        known_128_accepted = False
    try:
        _validate_raw_format(
            "physionet_eegmmidb", physio_128,
            Path("release/S087/S087R04.edf"), CYTON_MOTOR_CHANNELS)
        unexpected_128_rejected = False
    except ValueError:
        unexpected_128_rejected = True
    check("PhysioNet contract accepts only its three known 128 Hz subjects",
          public_native_sfreqs("physionet_eegmmidb") == (128.0, 160.0) and
          set(PUBLIC_DATA_FORMATS["physionet_eegmmidb"][
              "native_sfreq_exceptions"]) == {"S088", "S092", "S100"} and
          known_128_accepted and unexpected_128_rejected)
    bnci_channel_plan = public_channel_plan("bnci2014_001")
    check("only BNCI uses explicit interpolation for four absent Cyton nodes",
          bnci_channel_plan["interpolated_channels"] ==
          ("F3", "P3", "F4", "P4") and
          set(bnci_channel_plan["native_channels"]) ==
          {"C3", "CZ", "PZ", "C4"} and
          all(not public_channel_plan(value)["interpolated_channels"]
              for value in automated_format_ids - {"bnci2014_001"}))
    check("verified formats distinguish BNCI MAT data and Lee EEG licensing",
          PUBLIC_MI_DATASETS["bnci2014_001"]["format"] ==
          "MAT via MOABB/NEMAR" and
          PUBLIC_MI_DATASETS["lee2019_mi"]["license"] == "CC-BY-4.0")
    check("registry reports cohort demographics without inventing missing values",
          PUBLIC_MI_DATASETS["ofner2017"]["cohort"][
              "sex_or_gender_as_reported"]["female"] == 9 and
          PUBLIC_MI_DATASETS["cho2017"]["cohort"]["age"][
              "mean_years"] == 24.8 and
          PUBLIC_MI_DATASETS["physionet_eegmmidb"]["cohort"][
              "sex_or_gender_as_reported"] is None and
          "does not publish" in PUBLIC_MI_DATASETS[
              "physionet_eegmmidb"]["cohort"]["completeness_note"])
    check("restricted arm dataset is catalogued but cannot auto-import",
          PUBLIC_MI_DATASETS["zenodo_18470278"]["access"] == "restricted" and
          PUBLIC_MI_DATASETS["zenodo_18470278"]["importer"] is None)
    check("PhysioNet T1/T2 mapping is qualified by imagery run",
          physionet_imagery_label_map("S001R04.edf") ==
          {"T1": "left_hand", "T2": "right_hand"} and
          physionet_imagery_label_map("S001R06.edf") ==
          {"T1": "both_hands", "T2": "both_feet"})
    try:
        physionet_imagery_label_map("S001R03.edf")
        check("PhysioNet execution runs cannot be imported as imagery", False)
    except ValueError:
        check("PhysioNet execution runs cannot be imported as imagery", True)

    from tools.prepare_public_mi import (
        AUTOMATED_DATASETS, ESTIMATED_FULL_DOWNLOAD_GB, build_plan,
        cleanup_processed_datasets, ensure_disk_space,
        existing_dataset_archives, make_parser as make_public_workflow_parser,
        physionet_download_specs, resolve_bandpass, resolve_datasets,
        safe_cleanup, select_best_candidate)
    workflow_plan = build_plan(
        resolve_datasets("all"), ("riemann_lr", "riemann_svm"),
        subject_limit=3)
    check("automated workflow includes six public datasets but skips restricted data",
          tuple(row["id"] for row in workflow_plan["datasets"]) ==
          AUTOMATED_DATASETS and
          workflow_plan["restricted_skipped"] == ["zenodo_18470278"])
    check("automation keeps all four model categories separate",
          set(workflow_plan["categories"]) ==
          {"finger", "hand", "upper_limb", "feet"} and
          not workflow_plan["model_strategy"].startswith("universal"))
    cleanup_args = make_public_workflow_parser().parse_args([
        "--dry-run", "--cleanup-policy", "models-reports",
        "--hash-evidence-inputs"])
    check("local full-run options retain models/reports after hashed evidence",
          cleanup_args.cleanup_policy == "models-reports" and
          cleanup_args.hash_evidence_inputs and cleanup_args.dry_run)
    bnci_workflow_row = next(
        row for row in workflow_plan["datasets"]
        if row["id"] == "bnci2014_001")
    physio_workflow_row = next(
        row for row in workflow_plan["datasets"]
        if row["id"] == "physionet_eegmmidb")
    check("dry-run exposes format, interpolation, cohort and native rates",
          bnci_workflow_row["source_format"] == "MAT via MOABB/NEMAR" and
          bnci_workflow_row["interpolated_channels"] ==
          ["F3", "P3", "F4", "P4"] and
          bnci_workflow_row["cohort"]["participants"] == 9 and
          bnci_workflow_row["cohort"]["age"] is None and
          physio_workflow_row["verified_native_sfreqs"] == [128.0, 160.0])
    check("disk planning covers verified large MOABB sourcedata manifests",
          ESTIMATED_FULL_DOWNLOAD_GB["ofner2017"] >= 27.4 and
          ESTIMATED_FULL_DOWNLOAD_GB["lee2019_mi"] >= 64.9)
    broadband_plan = build_plan(
        resolve_datasets("all"), ("riemann_lr",), subject_limit=3,
        import_bandpass=resolve_bandpass("0.3,40"))
    check("separate experiment plans preserve a declared broad import passband",
          broadband_plan["import_bandpass_hz"] == [0.3, 40.0] and
          workflow_plan["import_bandpass_hz"] == [1.0, 40.0])
    physio_specs = physionet_download_specs(subject_limit=2)
    downloaded_runs = {int(Path(row["relative"]).stem[-2:])
                       for row in physio_specs}
    check("automation downloads only six PhysioNet imagery runs per subject",
          len(physio_specs) == 12 and downloaded_runs == {4, 6, 8, 10, 12, 14})
    selected_candidate = select_best_candidate([
        {"status": "success", "pipeline": "riemann_lr",
         "accuracy": .61, "std": .05},
        {"status": "success", "pipeline": "riemann_svm",
         "accuracy": .67, "std": .08},
        {"status": "failed", "pipeline": "tcformer", "error": "optional"},
    ])
    check("automation selects the best successful subject-held-out candidate",
          selected_candidate["pipeline"] == "riemann_svm")
    robust_candidate = select_best_candidate([
        {"status": "success", "pipeline": "subject_winner",
         "accuracy": .75, "std": .04,
         "dataset_accuracy": .61, "dataset_std": .08},
        {"status": "success", "pipeline": "dataset_winner",
         "accuracy": .70, "std": .05,
         "dataset_accuracy": .66, "dataset_std": .03},
    ])
    check("multi-source public selection prioritizes dataset-held-out evidence",
          robust_candidate["pipeline"] == "dataset_winner" and
          "leave-one-dataset-out" in
          robust_candidate["selection_criterion"])
    with tempfile.TemporaryDirectory() as cleanup_tmp_text:
        cleanup_root = Path(cleanup_tmp_text)
        owned_stage = cleanup_root / "auto"
        owned_stage.mkdir(); (owned_stage / "partial.bin").write_bytes(b"test")
        safe_cleanup(owned_stage, cleanup_root)
        outside_blocked = False
        try:
            safe_cleanup(cleanup_root, cleanup_root)
        except ValueError:
            outside_blocked = True
        processed_root = cleanup_root / "processed"
        cho_processed = processed_root / "cho2017" / "hand"
        other_processed = processed_root / "unrelated_local_data"
        cho_processed.mkdir(parents=True)
        other_processed.mkdir(parents=True)
        (cho_processed / "epochs.npz").write_bytes(b"archive")
        (other_processed / "keep.txt").write_text("keep", encoding="utf-8")
        processed_cleanup = cleanup_processed_datasets(
            ("cho2017",), processed_root)
        check("models-reports cleanup removes only named processed datasets",
              not (processed_root / "cho2017").exists() and
              other_processed.exists() and processed_root.exists() and
              processed_cleanup["bytes_removed"] == len(b"archive"))
        check("cleanup deletes only workflow-owned children and protects its root",
              not owned_stage.exists() and outside_blocked)
        disk_guard_blocked = False
        try:
            ensure_disk_space(cleanup_root, 1e12)
        except RuntimeError:
            disk_guard_blocked = True
        check("automation fails early when the free-space threshold is unmet",
              ensure_disk_space(cleanup_root, 0) >= 0 and disk_guard_blocked)

    from ML.public_mi_experiments import (
        CATEGORY_EXPERIMENT_RECIPES, OneVsRestFilterBankCSP, dataset_folds,
        harmonize_experiment_data, recipe_frequency_compatible)
    experiment_X = np.random.default_rng(778).normal(0, 1e-6, (14, 8, 300))
    experiment_data = {
        "X": experiment_X,
        "labels": np.asarray(
            ["idle", "both_feet"] * 3 + ["both_feet"] * 4 +
            ["idle", "both_feet"] * 2),
        "subjects": np.asarray([f"s{i // 2}" for i in range(14)]),
        "sessions": np.asarray([f"session-{i // 2}" for i in range(14)]),
        "source_dataset": np.asarray(
            ["physionet_eegmmidb"] * 6 + ["bnci2014_001"] * 4 +
            ["other_complete"] * 4),
        "source_labels": np.asarray(["event"] * 14),
        "category": "feet", "fs": 250.0, "units": "volts",
        "channel_labels": np.asarray(CYTON_MOTOR_CHANNELS),
        "imagery_or_execution": "imagery",
    }
    balanced_feet, feet_audit = harmonize_experiment_data(
        experiment_data, "feet", random_state=42)
    production_feet, production_feet_policy = apply_public_training_policy(
        experiment_data, "feet")
    feet_cells = Counter(zip(
        balanced_feet["source_dataset"], balanced_feet["labels"]))
    production_feet_cells = Counter(zip(
        production_feet["source_dataset"], production_feet["labels"]))
    check("feet experiments reject source-only classes and balance source-label cells",
          "bnci2014_001" not in set(balanced_feet["source_dataset"]) and
          set(feet_audit["excluded_incomplete_sources"]) == {"bnci2014_001"} and
          len(set(feet_cells.values())) == 1)
    check("production feet model enforces the same source-complete balance",
          "bnci2014_001" not in set(production_feet["source_dataset"]) and
          production_feet_policy["id"] ==
          "source-complete-idle-versus-both-feet-v1" and
          len(set(production_feet_cells.values())) == 1)
    hand_data = dict(experiment_data)
    hand_data["category"] = "hand"
    hand_data["labels"] = np.asarray(
        ["left_hand", "right_hand", "idle", "both_hands", "left_hand",
         "right_hand", "left_hand", "right_hand", "idle", "both_hands",
         "left_hand", "right_hand", "left_hand", "right_hand"])
    harmonized_hand, hand_audit = harmonize_experiment_data(hand_data, "hand")
    production_hand, production_policy = apply_public_training_policy(
        hand_data, "hand")
    check("hand experiments harmonize labels to left versus right only",
          set(harmonized_hand["labels"]) == {"left_hand", "right_hand"} and
          hand_audit["target_labels"] == ["left_hand", "right_hand"])
    check("public hand model adopts the same binary cross-source task",
          set(production_hand["labels"]) == {"left_hand", "right_hand"} and
          production_policy["id"] ==
          "cross-source-left-versus-right-hand-v1" and
          "fbcsp_lda" in PUBLIC_PIPELINES)
    source_split = list(dataset_folds(
        ["a", "a", "b", "b"],
        ["left_hand", "right_hand", "left_hand", "right_hand"]))
    check("experiment CV leaves out whole class-complete datasets",
          len(source_split) == 2 and all(
              not (set(np.asarray(["a", "a", "b", "b"])[train]) &
                   set(np.asarray(["a", "a", "b", "b"])[test]))
              for train, test, _source in source_split))
    upper_low_recipe = CATEGORY_EXPERIMENT_RECIPES["upper_limb"][0]
    check("frequency provenance blocks irrecoverable low-frequency experiments",
          not recipe_frequency_compatible(upper_low_recipe, (1.0, 40.0)) and
          recipe_frequency_compatible(upper_low_recipe, (0.3, 40.0)))
    sample_times = np.arange(300) / 250.0
    fbcsp_X = np.asarray([
        np.tile(np.sin(2 * np.pi * (10 if i < 6 else 18) * sample_times),
                (8, 1)) +
        np.random.default_rng(900 + i).normal(0, .05, (8, 300))
        for i in range(12)
    ])
    fbcsp = OneVsRestFilterBankCSP(
        bands=((8.0, 12.0), (16.0, 20.0)), sample_freq=250)
    fbcsp_features = fbcsp.fit_transform(fbcsp_X, np.repeat([0, 1], 6))
    check("experimental filter-bank CSP produces finite binary features",
          fbcsp_features.shape == (12, 8) and
          np.isfinite(fbcsp_features).all())

    with tempfile.TemporaryDirectory() as public_tmp_text:
        public_root = Path(public_tmp_text)
        archive_path = public_archive_path("cho2017", "hand", root=public_root)
        public_X = np.random.default_rng(777).normal(0, 1e-6, (12, 8, 750))
        public_labels = np.asarray(["left_hand", "right_hand"] * 6)
        public_subjects = np.repeat(["s1", "s2", "s3"], 4)
        public_sessions = np.asarray([f"session-{i // 4}" for i in range(12)])
        write_standardized_archive(
            archive_path, public_X, public_labels, public_subjects,
            public_sessions, source_dataset="cho2017",
            source_labels=public_labels, channel_labels=CYTON_MOTOR_CHANNELS,
            fs=250, units="volts", category="hand",
            preprocessing={"bandpass_hz": [1, 40]})
        public_loaded = load_standardized_archive(archive_path)
        check("standardized public archive retains provenance and source labels",
              public_loaded["X"].shape == (12, 8, 750) and
              public_loaded["source_dataset"] == "cho2017" and
              public_loaded["source_labels"].tolist() == public_labels.tolist() and
              public_loaded["preprocessing"]["bandpass_hz"] == [1, 40])
        check("full automation does not mistake a pilot archive for all subjects",
              existing_dataset_archives(
                  "cho2017", public_root, subject_limit=3) is not None and
              existing_dataset_archives(
                  "cho2017", public_root, subject_limit=None) is None)
        public_folds = list(subject_folds(
            public_loaded["subjects"], public_loaded["labels"], n_splits=3))
        check("public CV holds out whole subjects",
              len(public_folds) == 3 and all(
                  not (set(public_loaded["subjects"][train]) &
                       set(public_loaded["subjects"][test]))
                  for train, test in public_folds))
        lee_path = public_archive_path("lee2019_mi", "hand", root=public_root)
        write_standardized_archive(
            lee_path, public_X * 1.01, public_labels, public_subjects,
            public_sessions, source_dataset="lee2019_mi",
            source_labels=public_labels, channel_labels=CYTON_MOTOR_CHANNELS,
            fs=250, units="volts", category="hand",
            preprocessing={"bandpass_hz": [1, 40]})
        binary_result = train_public_category(
            [archive_path, lee_path], category="hand", pipeline="fbcsp_lda",
            output_root=public_root / "models", n_splits=3, verbose=False)
        binary_bundle = joblib.load(binary_result["path"])
        check("production hand bundle records binary labels and dataset-held-out CV",
              binary_bundle["version"] == 2 and
              binary_bundle["labels"] == ["left_hand", "right_hand"] and
              binary_bundle["training_policy"]["id"] ==
              "cross-source-left-versus-right-hand-v1" and
              binary_bundle["dataset_cv_folds"] == 2 and
              binary_result["dataset_accuracy"] is not None)
        from ML.public_mi_evidence import generate_public_mi_evidence
        evidence_root = public_root / "evidence"
        evidence, evidence_md, evidence_json, evidence_csv = \
            generate_public_mi_evidence(
                public_root, evidence_root,
                model_root=public_root / "models", categories=("hand",),
                n_splits=3, seed=42, project_root=ROOT)
        evidence_hand = evidence["categories"]["hand"]
        check("public evidence generator writes archive-backed grouped metrics",
              evidence_hand["status"] == "success" and
              evidence_hand["subject_held_out"]["available"] and
              evidence_hand["dataset_held_out"]["available"] and
              evidence_md.exists() and evidence_json.exists() and
              evidence_csv.exists() and
              "Saved selected model" in evidence_md.read_text(encoding="utf-8"))
        import csv as csv_module
        with evidence_csv.open(newline="", encoding="utf-8") as evidence_fh:
            source_rows = list(csv_module.DictReader(evidence_fh))
        check("public comparison table includes demographics and citable references",
              len(source_rows) == 6 and
              all(row["References"].startswith("https://")
                  for row in source_rows) and
              "female=9" in next(
                  row["Sex/gender as reported"] for row in source_rows
                  if row["Dataset"] == "ofner2017") and
              next(row["Age"] for row in source_rows
                   if row["Dataset"] == "physionet_eegmmidb").startswith(
                       "Not reported") and
              "10.3389/fnins.2012.00039" in next(
                  row["References"] for row in source_rows
                  if row["Dataset"] == "bnci2014_001"))
        inventory_path = evidence_root / "dataset_archive_inventory.csv"
        with inventory_path.open(newline="", encoding="utf-8") as inventory_fh:
            inventory_rows = list(csv_module.DictReader(inventory_fh))
        cho_inventory = next(
            row for row in inventory_rows
            if row["Dataset"] == "cho2017" and row["Category"] == "hand")
        check("public report inventories actual candidates, trials and samples",
              cho_inventory["Status"] == "available" and
              cho_inventory["Imported participants"] == "3" and
              cho_inventory["Trials"] == "12" and
              cho_inventory["Channels"] == "8" and
              cho_inventory["Samples per trial"] == "750" and
              cho_inventory["Total channel samples"] == str(12 * 8 * 750))
        check("public evidence generator emits class-colored CSP and tangent figures",
              {row["kind"] for row in evidence_hand["figures"]} == {
                  "csp_class_separation", "tangent_pca",
                  "class_covariance_summary", "subject_oof_confusion",
                  "subject_fold_scores"} and
              all(Path(row["path"]).is_file()
                  for row in evidence_hand["figures"]) and
              next(row for row in evidence_hand["figures"]
                   if row["kind"] == "csp_class_separation")["classes"] ==
              ["left_hand", "right_hand"])
        try:
            write_standardized_archive(
                public_root / "invented.npz", public_X[:2],
                ["left_thumb", "right_thumb"], ["s1", "s2"], ["a", "b"],
                source_dataset="openneuro_ds008446",
                source_labels=["3", "3"], channel_labels=CYTON_MOTOR_CHANNELS,
                fs=250, units="volts", category="finger")
            check("archive writer rejects invented finger laterality", False)
        except ValueError:
            check("archive writer rejects invented finger laterality", True)
        execution_path = public_root / "execution.npz"
        write_standardized_archive(
            execution_path, public_X[:2], ["left_hand", "right_hand"],
            ["s1", "s2"], ["a", "b"], source_dataset="cho2017",
            source_labels=["left_hand", "right_hand"],
            channel_labels=CYTON_MOTOR_CHANNELS, fs=250, units="volts",
            category="hand", imagery_or_execution="execution")
        try:
            load_standardized_archive(execution_path)
            check("imagery loader rejects motor-execution archives", False)
        except ValueError:
            check("imagery loader rejects motor-execution archives", True)

    with tempfile.TemporaryDirectory() as pool_tmp_text:
        pool_root = Path(pool_tmp_text)
        prng = np.random.default_rng(999)
        pX, pL = [], []
        for class_i, intent in enumerate((*UNIVERSAL_INTENTS, "idle")):
            for _ in range(8):
                epoch = prng.normal(0, 5, (8, 750))
                epoch[1 + class_i % 7] *= .2
                pX.append(epoch); pL.append(make_intent_label(intent))
        general_root = pool_root / "training-dataset"
        general_dir = general_recording_dir(general_root)
        general_dir.mkdir(parents=True)
        np.savez_compressed(
            general_dir / "calib_runtime_user_session0.npz",
            X=pX, y=np.arange(len(pX)), labels=pL,
            fs=250, montage="cyton8_motor", subject="runtime_user")
        np.savez_compressed(
            general_dir / "calib_other_subject_session0.npz",
            X=pX, y=np.arange(len(pX)), labels=pL,
            fs=250, montage="cyton8_motor", subject="other_subject")
        model_root = pool_root / "profiles"
        generalized = load_intent_data(
            "runtime_user", ["left_index", "right_index", "right_pinky"],
            root=model_root, data_root=general_root)
        check("universal training uses only the requested subject's data",
              generalized["X"].shape[0] == 32 and
              generalized["files"] ==
              ["general/calib_runtime_user_session0.npz"])
        category_results = train_universal_category_models(
            "runtime_user", root=model_root, data_root=general_root,
            screen=False, verbose=False, pipelines=("riemann_lr",))
        check("one universal pool trains separate finger, hand and feet models",
              set(category_results) == {"finger", "hand", "feet"} and
              all((category_profile_dir(
                  "runtime_user", category, root=model_root) /
                  "intent_model.joblib").exists()
                  for category in category_results))
        finger_game_model = MotorIntentModel.load_resolved(
            "runtime_user", ["left_index", "right_index", "right_pinky"],
            root=model_root, confidence_threshold=0.0)
        mixed_game_model = MotorIntentModel.load_resolved(
            "runtime_user", ["left_hand", "right_hand", "both_feet"],
            root=model_root, confidence_threshold=0.0)
        check("finger games select the personal finger category model",
              finger_game_model.describe()["model_scope"] ==
              "personal-category:finger")
        check("mixed games route through their hand and feet category models",
              mixed_game_model.describe()["model_scope"] ==
              "personal-category-router" and
              set(mixed_game_model.describe()["categories"]) == {"hand", "feet"})

    with tempfile.TemporaryDirectory() as intent_tmp_text:
        intent_root = Path(intent_tmp_text)
        intentions = default_map.intents
        intent_dir = intent_profile_dir("demo", intentions, root=intent_root)
        intent_dir.mkdir(parents=True)
        irng = np.random.default_rng(321)
        intent_labels = [make_intent_label(v) for v in intentions]
        for session in range(3):
            ix, iy, il = [], [], []
            for cls in range(3):
                for _ in range(10):
                    epoch = irng.normal(0, 5.0, (8, 750))
                    epoch[2 + cls] *= 0.15
                    ix.append(epoch); iy.append(cls); il.append(intent_labels[cls])
            order = irng.permutation(len(iy))
            np.savez_compressed(
                intent_dir / f"calib_demo_session{session}.npz",
                X=np.asarray(ix)[order], y=np.asarray(iy)[order],
                labels=np.asarray(il)[order], fs=250,
                montage="cyton8_motor")
        intent_result = train_intent_user(
            "demo", intentions, root=intent_root,
            data_root=intent_root / "empty-general", n_splits=3,
            screen=False, verbose=False)
        check("task-set trainer writes reusable intent_model.joblib",
              intent_result is not None and
              (intent_dir / "intent_model.joblib").exists())
        check("reusable profile is not stored under an app name",
              "rockpaperscissors" not in str(intent_result["path"]))
        check("session-held-out intent model exceeds synthetic chance",
              intent_result["accuracy"] is not None and
              intent_result["accuracy"] > .70)
        intent_model = MotorIntentModel.load(
            "demo", reversed(intentions), root=intent_root,
            confidence_threshold=0.0)
        predicted, predicted_conf = intent_model.predict_command(np.asarray(ix)[0])
        check("loaded reusable model emits a physical intention",
              predicted in set(intentions) and predicted_conf > 0)
        resolved_exact = MotorIntentModel.load_resolved(
            "demo", intentions, root=intent_root, confidence_threshold=0.0)
        check("runtime retains an exact task-set model only as fallback",
              resolved_exact.describe()["model_scope"] ==
              "exact-task-set-fallback")

        category_dir = category_profile_dir("demo", "finger", root=intent_root)
        category_dir.mkdir(parents=True)
        import joblib
        exact_bundle = joblib.load(intent_dir / "intent_model.joblib")
        exact_bundle["model_scope"] = "personal-category"
        exact_bundle["category"] = "finger"
        joblib.dump(exact_bundle, category_dir / "intent_model.joblib")
        (intent_dir / "intent_model.joblib").unlink()
        resolved_category = MotorIntentModel.load_resolved(
            "demo", intentions[:2], root=intent_root, confidence_threshold=0.0)
        resolved_description = resolved_category.describe()
        check("runtime selects a compatible personal category model",
              resolved_description["model_scope"] ==
              "personal-category:finger" and
              resolved_description["intents"] == list(intentions[:2]) and
              resolved_description["trained_intents"] == list(intentions))

    layer_total('LAYER 3 - OFFLINE MODEL')


def layer4():
    global check
    banner('LAYER 4 - ONLINE DECISION', 'With feedback and idle periods, does it emit the right command and stay\nsilent when it should? This is where false activations are caught.')

    # --------------------------------------------------------------
    # from test_gap4_idle.py
    # --------------------------------------------------------------
    import sys, time
    from pathlib import Path
    import numpy as np

    from ML.hybrid_classifier import HybridSSVEPClassifier, NO_ACTION  # noqa
    from ML.bci_engine import BCIEngine  # noqa
    from ML.sim_stream import SyntheticSSVEPStream  # noqa
    from ML import montage  # noqa

    print("\n== Montage / channel-map fix (blocker B1) ==")
    clf = HybridSSVEPClassifier()
    check("defaults to 8-ch Cyton montage", clf.ssvep_channels == [6,7])
    check("default targets are 15/20 Hz", clf.target_freqs == [15.0,20.0])
    check("sub-bands start above alpha", min(clf.sub_band_starts) >= 12.0)
    try:
        HybridSSVEPClassifier(montage="benchmark64").predict_proba(np.zeros((8,750)))
        check("64-ch config on 8-ch data raises clear error", False)
    except ValueError as e:
        check("64-ch config on 8-ch data raises clear error", "8" in str(e))

    print("\n== Confidence + NO_ACTION (blocker B4) ==")
    s = SyntheticSSVEPStream(seed=3)
    lab, conf, scores = clf.predict_proba(s.next_window())
    check("predict_proba returns 3-tuple", isinstance(scores, np.ndarray) and len(scores)==2)
    check("attentive window is confident", conf > clf.confidence_threshold)
    check("attentive window classified correctly", lab == 0)

    idle_stream = SyntheticSSVEPStream(seed=99, idle=True)
    idle_windows = [idle_stream.next_window() for _ in range(25)]
    idle_labels = [clf.predict_proba(w)[0] for w in idle_windows]
    n_noact = sum(1 for l in idle_labels if l == NO_ACTION)
    check(f"idle -> NO_ACTION ({n_noact}/25)", n_noact >= 20)

    print("\n== Threshold calibration from user's own idle data ==")
    c2 = HybridSSVEPClassifier()
    thr = c2.calibrate_threshold(idle_windows, percentile=95.0)
    check("calibrate_threshold sets a value", thr > 0)
    s2 = SyntheticSSVEPStream(seed=5)
    check("still detects target after calibration", c2.predict_proba(s2.next_window())[0] in (0,1))

    print("\n== BCIEngine: consecutive debounce + lockout + refractory ==")
    eng = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=0)
    st = SyntheticSSVEPStream(seed=1)
    states=[]
    for i in range(3):
        states.append(eng.process_frame(st.next_window()))
    check("needs 3 consecutive windows", states[0][1] is False and states[1][1] is False)
    check("fires on 3rd consecutive window", states[2][1] is True and states[2][0]=="SSVEP_0")

    eng2 = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=0)
    idle2 = SyntheticSSVEPStream(seed=42, idle=True)
    res = [eng2.process_frame(idle2.next_window()) for _ in range(20)]
    check("idle never dispatches a command", not any(t for _,t in res))
    check("idle reports NO_ACTION", sum(1 for d,_ in res if d=="NO_ACTION") >= 15)

    eng3 = BCIEngine(mode="SSVEP", debounce_window=2, artifact_lockout_ms=800, refractory_ms=0)
    blink = st.next_window(); blink = st.inject_blink(blink)
    d,t = eng3.process_frame(blink)
    check("blink detected", d=="BLINK" and not t)
    check("engine is locked out after blink", eng3.is_locked_out)
    d2,t2 = eng3.process_frame(st.next_window())
    check("output suppressed during lockout", d2=="LOCKOUT" and not t2)

    eng4 = BCIEngine(mode="SSVEP", debounce_window=2, refractory_ms=5000)
    st4 = SyntheticSSVEPStream(seed=2)
    fires = [eng4.process_frame(st4.next_window())[1] for _ in range(10)]
    check("refractory prevents key repeat (<=1 fire in 10)", sum(fires) <= 1)

    print("\n== BCIEngine: filter bypass fixed (blocker B3) ==")
    import inspect
    src = inspect.getsource(BCIEngine._classify)
    check("engine uses predict_proba (filtered path)", "predict_proba" in src)
    check("engine no longer calls raw cca_ssvep_detection", "cca_ssvep_detection" not in src)

    print("\n== End-to-end: realistic distracted session ==")
    eng5 = BCIEngine(mode="SSVEP", debounce_window=3, refractory_ms=1200)
    att = SyntheticSSVEPStream(seed=7); idl = SyntheticSSVEPStream(seed=8, idle=True)
    rng = np.random.default_rng(0)
    false_pos = 0; true_pos = 0
    for i in range(150):
        if i % 10 < 3:                       # 30% attentive, in runs (realistic gaze)
            att.set_active(0)
            d,t = eng5.process_frame(att.next_window())
            if t: true_pos += 1
        else:
            d,t = eng5.process_frame(idl.next_window())
            if t: false_pos += 1
    rep = eng5.report()
    print(f"    dispatched={rep['dispatched']} true={true_pos} false_idle={false_pos} "
          f"no_action_rate={rep['no_action_rate']:.1%}")
    check("zero false commands during idle/distraction", false_pos == 0)
    check("still issues real commands", true_pos >= 1)
    check("majority of session is NO_ACTION", rep["no_action_rate"] > 0.5)

    # --------------------------------------------------------------
    # from test_step10_realtime.py
    # --------------------------------------------------------------
    _std = check
    check = check_cond_first
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np


    from bci_sdk.selection import SelectionTree, LEFT, RIGHT          # noqa: E402
    from bci_sdk.direct import (DirectButtonSelector,                  # noqa: E402
                                IntentActionSelector)
    from bci_sdk.runtime import (RealtimeDecider, ArtifactDetector,    # noqa: E402
                                 BLINK, LOCKOUT, BUILDING, REFRACTORY,
                                 REJECTED, DISPATCHED)
    from ML.local_model import LocalMIModel, NO_ACTION                 # noqa: E402



    class Btn:
        def __init__(self, name):
            self.name, self.fired = name, 0

        def trigger(self):
            self.fired += 1

        def __repr__(self):
            return f"Btn({self.name})"


    class FakeModel:
        """Scripted classifier so the gates can be tested without EEG."""

        def __init__(self, script):
            self.script = list(script)        # list of (command|None, confidence)
            self.i = 0
            self.sample_freq = 250
            self.montage = None

        def predict_command(self, window):
            v = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
            return v


    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

        def advance(self, ms):
            self.t += ms


    # =====================================================================
    print("\n1. SelectionTree geometry")
    # =====================================================================
    btns = [Btn(f"b{i}") for i in range(5)]
    t = SelectionTree(btns)
    check(t.max_depth() == 3, "5 buttons need at most 3 decisions",
          f"got {t.max_depth()}")
    check(abs(t.mean_depth() - 2.4) < 1e-9, "mean depth is 2.4",
          f"got {t.mean_depth()}")
    check(sorted(t.depths()) == [2, 2, 2, 3, 3], "depths are 2,2,2,3,3",
          f"got {sorted(t.depths())}")

    for n, want in ((2, 1), (3, 2), (4, 2), (8, 3), (9, 4), (16, 4)):
        tt = SelectionTree([Btn(str(i)) for i in range(n)])
        check(tt.max_depth() == want, f"{n} buttons -> ceil(log2) = {want}",
              f"got {tt.max_depth()}")

    # =====================================================================
    print("\n2. Both classes select; every button is reachable")
    # =====================================================================
    reached = set()
    for target in range(5):
        tr = SelectionTree([Btn(f"b{i}") for i in range(5)])
        # walk down to `target` by choosing the half that contains it
        guard = 0
        while len(tr.candidates) > 1 and guard < 10:
            lo, _ = tr._split(tr.candidates)
            tr.decide(LEFT if target in lo else RIGHT)
            guard += 1
        reached.add(target)
    check(reached == set(range(5)), "all 5 buttons reachable by LEFT/RIGHT alone")

    tr = SelectionTree(btns)
    out = [tr.decide(RIGHT), tr.decide(RIGHT)]
    check(out[-1] is not None, "RIGHT,RIGHT resolves a leaf in 2 decisions")
    check(sum(b.fired for b in btns) == 1, "exactly one button fired",
          f"fired={[b.fired for b in btns]}")
    check(len(tr.candidates) == 5, "tree resets itself after a selection")

    tr2 = SelectionTree([Btn(f"b{i}") for i in range(8)])
    tr2.decide(LEFT)
    before = list(tr2.candidates)
    tr2.decide(RIGHT)
    check(tr2.candidates != before, "second decision narrows further")
    tr2.undo()
    check(tr2.candidates == before, "undo() steps back exactly one decision")

    tr3 = SelectionTree([Btn(f"b{i}") for i in range(5)])
    tr3.decide(RIGHT); tr3.decide(RIGHT)          # resolves and fires
    check(tr3.undo() == list(range(5)),
          "undo() after a selection cannot un-fire it; tree stays reset")

    # =====================================================================
    print("\n2b. Experimental direct-button dispatch is not a tree")
    # =====================================================================
    direct_buttons = [Btn("rock"), Btn("paper"), Btn("scissors")]
    direct_picks = []
    direct = DirectButtonSelector(direct_buttons,
                                  on_select=lambda b: direct_picks.append(b.name))
    check(direct.names == ["rock", "paper", "scissors"],
          "direct selector exposes exact button names")
    clk_direct = Clock()
    direct_decider = RealtimeDecider(
        FakeModel([("paper", .95)] * 3), direct, clock=clk_direct,
        debounce_window=3, refractory_ms=0)
    direct_results = [direct_decider.step(np.zeros((8, 750))) for _ in range(3)]
    check(direct_results[-1]["selected"] is direct_buttons[1],
          "one accepted class selects Paper directly (no LEFT/RIGHT sequence)")
    check(direct_picks == ["paper"] and
          [b.fired for b in direct_buttons] == [0, 1, 0],
          "direct prediction fires exactly the named button")
    try:
        direct.decide("unknown")
        check(False, "unknown direct-button output fails loudly")
    except ValueError:
        check(True, "unknown direct-button output fails loudly")

    print("\n2c. Reusable motor intentions map to developer actions")
    intent_buttons = [Btn("rock"), Btn("paper"), Btn("scissors")]
    intent_picks = []
    intent_selector = IntentActionSelector(
        intent_buttons,
        {"rock": "left_index", "paper": "right_index",
         "scissors": "right_pinky"},
        on_select=lambda b: intent_picks.append(b.name))
    intent_decider = RealtimeDecider(
        FakeModel([("right_pinky", .96)] * 3), intent_selector,
        clock=Clock(), debounce_window=3, refractory_ms=0)
    intent_results = [intent_decider.step(np.zeros((8, 750))) for _ in range(3)]
    check(intent_results[-1]["selected"] is intent_buttons[2],
          "right_pinky selects developer action Scissors directly")
    check(intent_picks == ["scissors"] and
          [b.fired for b in intent_buttons] == [0, 0, 1],
          "intent mapping fires exactly one configured action")

    # =====================================================================
    print("\n3. Gates: consecutive agreement")
    # =====================================================================
    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", 0.9)] * 10), tree, clock=clk,
                        debounce_window=3, refractory_ms=0)
    states = [d.step(np.zeros((8, 750)))["state"] for _ in range(3)]
    check(states == [BUILDING, BUILDING, DISPATCHED],
          "3 agreeing windows required before dispatch", f"got {states}")

    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", .9), ("RIGHT", .9), ("LEFT", .9),
                                   ("RIGHT", .9), ("LEFT", .9), ("RIGHT", .9)]),
                        tree, clock=clk, debounce_window=3, refractory_ms=0)
    states = [d.step(np.zeros((8, 750)))["state"] for _ in range(6)]
    check(DISPATCHED not in states, "alternating predictions never dispatch",
          f"got {states}")

    # =====================================================================
    print("\n4. Gates: confidence rejection (the reject option)")
    # =====================================================================
    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([(None, 0.4)] * 20), tree, clock=clk,
                        refractory_ms=0)
    states = {d.step(np.zeros((8, 750)))["state"] for _ in range(20)}
    check(states == {REJECTED}, "sub-threshold windows are all rejected")
    check(d.stats["dispatched"] == 0, "no command from 20 rejected windows")
    check(d.report()["reject_rate"] == 1.0, "reject_rate reports 1.0")

    # a run broken by one rejection must restart from zero
    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", .9), ("LEFT", .9), (None, .1),
                                   ("LEFT", .9), ("LEFT", .9)]),
                        tree, clock=clk, debounce_window=3, refractory_ms=0)
    states = [d.step(np.zeros((8, 750)))["state"] for _ in range(5)]
    check(DISPATCHED not in states, "a rejected window breaks the run",
          f"got {states}")

    # =====================================================================
    print("\n5. Gates: refractory period")
    # =====================================================================
    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", 0.9)] * 20), tree, clock=clk,
                        debounce_window=1, refractory_ms=1200)
    r1 = d.step(np.zeros((8, 750)))
    clk.advance(500)
    r2 = d.step(np.zeros((8, 750)))
    clk.advance(800)                       # 1300 ms total
    r3 = d.step(np.zeros((8, 750)))
    check(r1["state"] == DISPATCHED, "first command dispatches")
    check(r2["state"] == REFRACTORY, "second at +500 ms is suppressed")
    check(r3["state"] == DISPATCHED, "third at +1300 ms dispatches")

    # =====================================================================
    print("\n6. Artifact detection runs on RAW data")
    # =====================================================================
    fs = 250
    tt = np.arange(750) / fs
    clean = np.random.default_rng(0).normal(0, 5, (8, 750))
    blink = clean.copy()
    # a blink: large, slow, frontal
    blink[0] += 220 * np.exp(-((tt - 1.5) ** 2) / (2 * 0.12 ** 2))
    blink[1] += 200 * np.exp(-((tt - 1.5) ** 2) / (2 * 0.12 ** 2))

    det = ArtifactDetector(channels=(0, 1), sample_freq=fs)
    check(not det.is_artifact(clean), "clean window is not flagged",
          f"peak {det.peak_uv(clean):.1f} uV")
    check(det.is_artifact(blink), "220 uV blink is flagged",
          f"peak {det.peak_uv(blink):.1f} uV")

    # the reason the check must come first: the motor band removes the blink
    from scipy.signal import butter, sosfiltfilt                      # noqa: E402
    sos = butter(4, [8, 30], btype="band", fs=fs, output="sos")
    after = float(np.max(np.abs(sosfiltfilt(sos, blink[:2], axis=-1))))
    check(after < 100.0,
          f"8-30 Hz filtering hides the blink ({det.peak_uv(blink):.0f} uV raw "
          f"-> {after:.0f} uV filtered, under the 100 uV threshold)")

    clk = Clock()
    tree = SelectionTree([Btn(f"b{i}") for i in range(4)])
    d = RealtimeDecider(FakeModel([("LEFT", 0.95)] * 20), tree, artifact=det,
                        clock=clk, debounce_window=1, artifact_lockout_ms=800,
                        refractory_ms=0)
    s1 = d.step(blink)
    s2 = d.step(clean)
    clk.advance(900)
    s3 = d.step(clean)
    check(s1["state"] == BLINK, "blink window reports BLINK")
    check(s2["state"] == LOCKOUT, "next window is locked out")
    check(s3["state"] == DISPATCHED, "output resumes after the 800 ms lockout")

    # =====================================================================
    print("\n7. Full loop: train -> save -> load -> decide")
    # =====================================================================
    tmp = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(7)
        C, T = 8, 750

        def epoch(label, n):
            """ERD: class 1 suppresses power on ch2, class 0 on ch4."""
            out = []
            for _ in range(n):
                s = rng.normal(0, 10.0, (C, T))
                s[2] *= 0.55 if label == 1 else 1.0
                s[4] *= 1.0 if label == 1 else 0.55
                out.append(s)
            return out

        X, y, lab = [], [], []
        for _ in range(3):                    # 3 blocks so grouped CV is valid
            for cls in (0, 1):
                for w in epoch(cls, 8):
                    X.append(w); y.append(cls)
                    lab.append(f"target_{cls}")
        X = np.array(X)
        udir = tmp / "demo"
        udir.mkdir(parents=True)
        for b in range(3):
            sl = slice(b * 16, (b + 1) * 16)
            np.savez_compressed(udir / f"calib_{b}.npz", X=X[sl],
                                labels=np.array(lab[sl]), fs=250,
                                montage="cyton8_motor")

        from ML.train_local import train_user
        res = train_user("demo", root=tmp, verbose=False)
        check(res is not None, "train_user produced a model")
        check((udir / "model.joblib").exists(), "model.joblib written to disk")

        m = LocalMIModel.load("demo", root=tmp, confidence_threshold=0.6)
        check(m.source == "personal", "LocalMIModel loads the personal model")
        check(m.sample_freq == 250, "sample rate carried through")
        check(m.describe()["pipeline"] == res["pipeline"],
              "pipeline name round-trips")

        # scale invariance: _cov() trace-normalises, so overall gain cannot move
        # the features. This is why electrode gain drift between sessions is not a
        # source of error, and why the uV/volts question is moot.
        probe = X[0]
        lab0, conf0 = m.predict_proba(probe)
        lab1, conf1 = m.predict_proba(probe * 1e-6)
        lab2, conf2 = m.predict_proba(probe * 1e3)
        check(lab0 == lab1 == lab2 and abs(conf0 - conf1) < 1e-9
              and abs(conf0 - conf2) < 1e-9,
              "predictions are invariant to overall signal scale",
              f"uV {conf0:.6f}  x1e-6 {conf1:.6f}  x1e3 {conf2:.6f}")

        kb = (udir / "model.joblib").stat().st_size / 1024
        check(kb < 100, f"model is small enough to ship ({kb:.1f} KB)")

        # drive the real-time path with class-1 windows
        buttons = [Btn(f"b{i}") for i in range(5)]
        tree = SelectionTree(buttons)
        clk = Clock()
        dec = RealtimeDecider(m, tree, artifact=ArtifactDetector(sample_freq=250),
                              clock=clk, debounce_window=3, refractory_ms=1200)
        fired = 0
        for i in range(40):
            w = epoch(1, 1)[0]
            r = dec.step(w)
            clk.advance(1000)               # 1 s hop
            if r["selected"] is not None:
                fired += 1
        check(dec.stats["windows"] == 40, "all 40 windows processed")
        check(fired >= 1, f"at least one button selected end-to-end (got {fired})")
        check(sum(b.fired for b in buttons) == fired,
              "button.trigger() fired exactly once per selection")
        print(f"         {dec.report()}")

        # no-model case must fail loudly
        try:
            LocalMIModel.load("nobody", root=tmp)
            check(False, "missing model raises FileNotFoundError")
        except FileNotFoundError:
            check(True, "missing model raises FileNotFoundError")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check = _std

    layer_total('LAYER 4 - ONLINE DECISION')


def layer5():
    global check
    banner('LAYER 5 - PRODUCT AND USABILITY', 'Does the library a developer actually touches behave correctly?')

    # --------------------------------------------------------------
    # from test_speller.py
    # --------------------------------------------------------------
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, str(ROOT / "pygame_lib"))

    from ML.text_buffer import TextBuffer  # noqa: E402



    print("\n== TextBuffer ==")
    b = TextBuffer()
    for ch in "HELLO":
        b.accept(ch)
    check("letters accumulate + auto-case", b.text == "Hello")
    b.accept(TextBuffer.SPACE)
    b.accept(TextBuffer.SPACE)
    check("double space collapsed", b.text == "Hello ")
    b.accept("W"); b.accept("O"); b.accept("R"); b.accept("L"); b.accept("D")
    check("second word", b.text == "Hello world")
    b.accept(TextBuffer.BACKSPACE)
    check("backspace", b.text == "Hello worl")
    b.accept(TextBuffer.UNDO)
    check("undo restores", b.text == "Hello world")
    b.accept(".")
    b.accept(TextBuffer.SPACE)
    b.accept("i")
    check("capitalise after sentence end", b.text == "Hello world. I")

    spoken = {}
    b2 = TextBuffer(on_speak=lambda t: spoken.setdefault("t", t))
    b2.accept("Yes. ")
    b2.accept(TextBuffer.SPEAK)
    check("phrase tile inserts whole phrase", b2.text == "Yes. ")
    check("speak callback fires", spoken.get("t") == "Yes. ")
    b2.accept(TextBuffer.CLEAR)
    check("clear empties", b2.text == "")
    check("selection log recorded", len(b2.selection_log) == 3)

    b3 = TextBuffer()
    for c in "the quick brown fox jumps over the lazy dog again and again":
        b3.accept(c)
    w = b3.wrapped(chars_per_line=20, max_lines=3)
    check("wrapped returns <= max_lines", len(w) <= 3)
    check("wrapped respects width", all(len(l) <= 20 for l in w))

    print("\n== Grid + navigation ==")
    import pygame  # noqa: E402
    pygame.init()
    pygame.display.set_mode((1024, 700))
    from example.bci_speller import Speller, build_layout  # noqa: E402

    rows = build_layout()
    cells = [c for r in rows for c in r]
    labels = [c[0] for c in cells]
    check("alphabet present", all(ch in labels for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    check("digits present", all(d in labels for d in "0123456789"))
    check("phrase tiles present", "HELP" in labels and "WATER" in labels)
    check("action tiles present", {"SPACE", "DEL", "CLEAR", "SPEAK"} <= set(labels))
    check("grid is 7 wide", all(len(r) <= 7 for r in rows))

    app = Speller(source="keyboard")
    check("tiles built", sum(len(r) for r in app.tiles) == len(cells))
    check("no tile overlaps panel", all(t.rect.top > 0 for r in app.tiles for t in r))
    check("tiles fit window", all(t.rect.right <= 1024 and t.rect.bottom <= 700
                                  for r in app.tiles for t in r))

    # spell "HI" by scanning: find H then I
    def select_label(app, label):
        for r, row in enumerate(app.tiles):
            for c, t in enumerate(row):
                if t.label == label:
                    app.stage, app.row_idx, app.col_idx = "ROW", 0, 0
                    for _ in range(r):
                        app.cmd_next()
                    app.cmd_select()               # enter row
                    for _ in range(c):
                        app.cmd_next()
                    app.cmd_select()               # commit tile
                    return True
        return False

    select_label(app, "H")
    select_label(app, "I")
    check("row-column scanning spells 'Hi'", app.buffer.text == "Hi")
    check("returns to ROW stage after commit", app.stage == "ROW")

    app.stage, app.row_idx, app.col_idx = "COL", 0, len(app.tiles[0])
    app.cmd_select()
    check("escape cell returns to row scan", app.stage == "ROW")

    select_label(app, "SPACE")
    select_label(app, "HELP")
    check("phrase tile via scanning", app.buffer.text == "Hi I need help. ")

    app.draw()
    check("renders without error", True)

    print("\n== Synthetic stream ==")
    from ML.sim_stream import SyntheticSSVEPStream  # noqa: E402
    s = SyntheticSSVEPStream()
    w0 = s.next_window()
    check("window shape (8,750) - Cyton", w0.shape == (8, 750))
    s.set_active(1)
    check("active target switchable", s.active == 1)

    pygame.quit()

    # --------------------------------------------------------------
    # from test_step7_sdk.py
    # --------------------------------------------------------------
    import os, re, sys, time
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from pathlib import Path
    import numpy as np
    import bci_sdk

    print("\n== Package surface ==")
    check("has a version", bool(bci_sdk.__version__))
    check("info() reports capabilities", set(bci_sdk.info()) >=
          {"version","ui_available","hardware_available","montages"})
    for name in ["BCISession","Command","BCIEngine","StreamBridge","TextBuffer",
                 "HybridSSVEPClassifier","TRCAClassifier","BrainButton","ButtonGroup",
                 "Scanner","SyntheticBackend","ReplayBackend",
                 "OpenBCICSVBackend","CSVReplayBackend",
                 "load_openbci_gui_csv","NO_ACTION"]:
        check(f"exports {name}", hasattr(bci_sdk, name))
    check("__all__ matches real attributes",
          all(hasattr(bci_sdk,n) for n in bci_sdk.__all__))
    check("exports reusable in-app calibration panel",
          hasattr(bci_sdk, "IntentCalibrationPanel"))
    check("exports body map and Game/Calibration landing menu",
          hasattr(bci_sdk, "draw_body_intent_map") and
          hasattr(bci_sdk, "GameCalibrationMenu"))

    # A fresh universal-recorder process needs only the motor-intent vocabulary
    # and body-map renderer. Importing either must not drag in the scientific
    # training stack before the pygame window can open.
    import subprocess
    lazy_probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; from ML import motor_intent; "
         "from bci_sdk.body_map import draw_body_intent_map; "
         "print(int('scipy' in sys.modules), int('sklearn' in sys.modules))"],
        cwd=ROOT, text=True, capture_output=True, timeout=30)
    check("recorder startup imports avoid eager scipy/sklearn loading",
          lazy_probe.returncode == 0 and lazy_probe.stdout.strip() == "0 0",
          lazy_probe.stderr or lazy_probe.stdout)

    print("\n== In-app motor-intent calibration tab ==")
    pygame.init(); pygame.display.set_mode((800, 600))
    menu = bci_sdk.GameCalibrationMenu("TEST")
    menu.draw(pygame.display.get_surface())
    menu.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    menu_choice = menu.handle_event(
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    check("landing menu supports Right + Enter calibration selection",
          menu_choice == "calibration")
    body_probe = pygame.Surface((600, 430))
    body_probe.fill((9, 20, 29))
    bci_sdk.draw_body_intent_map(
        body_probe, pygame.Rect(0, 0, 600, 430), "right_index")
    body_rgb = pygame.surfarray.array3d(body_probe)
    active_pixels = np.all(body_rgb == np.array((244, 201, 93)), axis=2)
    check("anatomical two-hand/both-feet map renders a clear finger highlight",
          int(active_pixels.sum()) > 200)
    for intent in ("left_thumb", "left_pinky", "right_thumb", "right_pinky",
                   "left_hand", "right_hand", "both_feet", "idle"):
        compact_probe = pygame.Surface((300, 260))
        compact_probe.fill((9, 20, 29))
        bci_sdk.draw_body_intent_map(
            compact_probe, compact_probe.get_rect(), intent, compact=True)
    check("CC0 outline map renders hands, all edge fingers, and soles compactly",
          True)

    # Regression for both earlier failures: a selected finger must be filled,
    # and its mask must share no pixel with any neighbouring finger or palm.
    import bci_sdk.body_map as body_map_module
    body_assets = body_map_module._load_assets(pygame)
    body_colors = ((34, 62, 75), (220, 232, 235), (244, 201, 93))
    outline_pixels = (pygame.surfarray.array_alpha(
        body_assets["hand_outline"]) > 24)
    finger_pixels = {
        finger: (pygame.surfarray.array_alpha(
            body_assets[f"finger_{finger}"]) > 0)
        for finger in body_map_module._FINGER_MASK_ASSETS
    }
    overlaps = []
    finger_names = list(finger_pixels)
    for first_index, first in enumerate(finger_names):
        for second in finger_names[first_index + 1:]:
            overlaps.append(int((finger_pixels[first]
                                 & finger_pixels[second]).sum()))
    check("finger highlight masks are mutually exclusive (zero overlap)",
          max(overlaps, default=0) == 0, f"overlaps={overlaps}")

    finger_coverages = []
    for finger in finger_names:
        rendered = body_map_module._render_hand(
            pygame, "left", finger, False, body_colors)
        expected_pixels = finger_pixels[finger] & ~outline_pixels
        rendered_rgb = pygame.surfarray.array3d(rendered)
        gold_pixels = np.all(
            rendered_rgb == np.array((244, 201, 93)), axis=2)
        finger_coverages.append(
            float((gold_pixels & expected_pixels).sum())
            / max(1, int(expected_pixels.sum())))
    check("selected highlight fills every finger interior (all >95%)",
          all(value > .95 for value in finger_coverages),
          f"coverage={finger_coverages}")

    class CalTarget:
        def __init__(self, index, name):
            self.id, self.name = index, name
            self.rect = pygame.Rect(index * 20, 0, 18, 18)
            self.is_flashing = False
        def flash(self, **kwargs): self.is_flashing = True
        def stop_flash(self): self.is_flashing = False

    cal_targets = [CalTarget(0, "rock"), CalTarget(1, "paper"),
                   CalTarget(2, "scissors")]
    panel = bci_sdk.IntentCalibrationPanel(
        cal_targets,
        {"rock": "left_index", "paper": "right_index",
         "scissors": "right_pinky"},
        user="test", source="dry-run", action_trials=3, idle_trials=1,
        cue_s=.01, rest_s=.01)
    panel.visible = True
    panel.start()
    deadline = time.time() + 5
    while time.time() < deadline and panel.state not in (panel.RESULT, panel.ERROR):
        panel.update(); time.sleep(.005)
    check("dry-run calibration tab completes its full state machine",
          panel.state == panel.RESULT and panel.result.get("dry_run"))
    check("dry-run calibration deliberately saves no model",
          panel.result.get("n_trials") == 4 and not panel.error)
    panel.draw(pygame.display.get_surface())
    check("calibration setup/result renders without error", True)
    panel.result = {
        "pipeline": "riemann_lr", "accuracy": .5, "n_sessions": 1,
        "reliable": False, "comparison_kind": "training-set preview",
        "comparison_correct": 1,
        "prediction_comparison": [
            {"intended": "left_index", "predicted": "left_index",
             "correct": True},
            {"intended": "right_index", "predicted": "idle",
             "correct": False},
        ],
    }
    panel.state = panel.RESULT
    panel.draw(pygame.display.get_surface())
    check("calibration result renders intended versus model prediction", True)

    print("\n== Gameplay prediction feedback + monitor export ==")
    import json
    import tempfile
    from bci_sdk.performance import GameplayMonitor, TrainingDataPerformance

    class MonitorBridge:
        def __init__(self): self.drained = 0
        def poll_all(self): self.drained += 1; return []

    class MonitorBCI:
        def __init__(self):
            self.bridge = MonitorBridge()
            self.last_window = None
            self.count = 0
        def poll(self):
            self.count += 1
            self.last_window = np.full((8, 750), self.count, dtype=float)
            return {"state": "DISPATCHED", "command": "left_index",
                    "selected": "rock", "confidence": .82,
                    "candidates": ["rock"]}

    with tempfile.TemporaryDirectory() as monitor_tmp_text:
        gameplay = GameplayMonitor("test_game", user="anson",
                                   root=Path(monitor_tmp_text),
                                   feedback_lockout_s=0)
        monitor_bci = MonitorBCI()
        gameplay.poll(monitor_bci)
        left_feedback = gameplay.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(1, 1)),
            monitor_bci)
        gameplay.poll(monitor_bci)
        right_feedback = gameplay.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(1, 1)),
            monitor_bci)
        pause_used = gameplay.handle_event(
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE), monitor_bci)
        resume_used = gameplay.handle_event(
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE), monitor_bci)
        status_path = gameplay.logger.status_path
        eeg_path = gameplay.logger.eeg_path
        gameplay.close()
        monitor_status = json.loads(status_path.read_text())
        check("left/right clicks mark latest prediction correct/incorrect",
              left_feedback and right_feedback and
              monitor_status["stats"]["feedback_correct"] == 1 and
              monitor_status["stats"]["feedback_incorrect"] == 1)
        check("Space pauses/resumes and drains stale queued EEG on resume",
              pause_used and resume_used and not gameplay.paused and
              monitor_bci.bridge.drained == 1)
        check("separate dashboard receives status, events and 8-channel EEG",
              status_path.exists() and eeg_path.exists() and
              np.load(eeg_path).shape == (8, 750) and
              (status_path.parent / "events.jsonl").exists())

    print("\n== Saved training-data performance monitor ==")
    with tempfile.TemporaryDirectory() as training_tmp_text:
        training_root = Path(training_tmp_text)
        rng = np.random.default_rng(2026)
        training_X = rng.normal(0, 5, (16, 8, 750))
        training_labels = np.array(
            ["intent::left_hand"] * 8 + ["intent::right_hand"] * 8)
        training_file = training_root / "calib_brian_session0.npz"
        np.savez_compressed(
            training_file, X=training_X, labels=training_labels,
            y=np.array([0] * 8 + [1] * 8), fs=250,
            montage="cyton8_motor", subject="brian")
        training_monitor = TrainingDataPerformance([training_file])
        training_report = training_monitor.report()
        first_trial = training_monitor.trial(0)
        check("training monitor loads labels, sessions and 8-channel epochs",
              training_report["trials"] == 16 and
              training_report["sessions"] == 1 and
              training_report["shape"] == [16, 8, 750] and
              training_report["class_counts"]["intent::left_hand"] == 8 and
              training_report["channel_labels"] ==
              ["F3", "C3", "CZ", "P3", "PZ", "F4", "C4", "P4"])
        check("training monitor exposes screening verdict and selected EEG trial",
              isinstance(first_trial["keep"], bool) and
              first_trial["window"].shape == (8, 750) and
              first_trial["intended"] == "left_hand")
        check("training monitor labels model metrics as training-only preview",
              "not held-session" in training_report["warning"])
        check("training monitor reports aggregate rejection reasons",
              set(training_report["drop_reason_counts"]) ==
              {"amplitude", "flat", "jump", "outlier", "noncompliant"})
        check("NPZ report exposes ordered rejection-step and per-trial indicators",
              [row["reason"] for row in training_report["rejection_steps"]] ==
              ["amplitude", "flat", "jump", "outlier", "noncompliant"] and
              len(training_report["trial_rejection_indicators"]) == 16 and
              training_report["recording_decision"]["indicator"] in
              {"ACCEPT", "REJECT"})
        check("training monitor attributes amplitude/flat/jump failures per channel",
              len(training_report["channel_diagnostics"]) == 8 and
              all({"peak_p95_uv", "amplitude_fail_trials", "std_min_uv",
                   "flat_fail_trials", "jump_p95_uv", "jump_fail_trials"}
                  <= set(row) for row in training_report["channel_diagnostics"]))

        rejected_X = training_X[:4].copy()
        rejected_X[0, 0, :] = 0.0
        rejected_X[1, 1, 100] = 1000.0
        rejected_file = training_root / "calib_brian_rejected.npz"
        np.savez_compressed(
            rejected_file, X=rejected_X,
            labels=np.array(["intent::left_hand"] * 2 +
                            ["intent::right_hand"] * 2),
            y=np.array([0, 0, 1, 1]), fs=250,
            montage="cyton8_motor", subject="brian")
        rejected_report = TrainingDataPerformance([rejected_file]).report()
        indicators = rejected_report["trial_rejection_indicators"]
        check("rejected NPZ names exact rejecting steps and responsible channels",
              indicators[0]["indicator"] == "REJECT" and
              indicators[0]["channel_failures"]["flat"] == ["F3"] and
              indicators[1]["indicator"] == "REJECT" and
              indicators[1]["channel_failures"]["amplitude"] == ["C3"] and
              rejected_report["recording_decision"]["indicator"] == "REJECT")

        from EEG.relabel_recording_montage import relabel
        corrected = relabel(training_file, "cyton8_fp_occipital")
        with np.load(corrected, allow_pickle=True) as corrected_data:
            check("montage relabeller changes metadata but not EEG samples",
                  str(corrected_data["montage"]) == "cyton8_fp_occipital" and
                  corrected_data["channel_labels"].astype(str).tolist() ==
                  ["Fp1", "Fp2", "O1", "Oz", "O2", "PO3", "PO4", "Pz"] and
                  np.array_equal(corrected_data["X"], training_X))

        from tools.inspect_training_recordings import inspect_file, build_html
        inspected = inspect_file(
            corrected,
            {"amp_uv": 400, "jump_uv": 75, "flat_uv": .05, "z_max": 4})
        report_html, report_json = build_html(
            [(corrected, *inspected)], training_root / "inspection.html")
        inspection_payload = json.loads(report_json.read_text())
        check("tools training inspector reports partial/mixed-montage usefulness",
              report_html.exists() and report_html.stat().st_size > 1000 and
              inspection_payload[0]["partial_universal"] and
              not inspection_payload[0]["has_motor_core"] and
              "LIMITED" in inspection_payload[0]["verdict"] and
              "Rejection pipeline indicators" in
              report_html.read_text(encoding="utf-8"))

        from tools.debug_bci_pipeline import (
            _common_channel_overlay, _source_trace_svg, filtered_layer,
            load_reference_csv, raw_layer, reference_layer,
            separability_layer, spectral_comparison_layer, write_report)
        raw_stages, debug_monitors = raw_layer(
            [corrected], {"amp_uv": 400, "jump_uv": 75,
                          "flat_uv": .05, "z_max": 4})
        debug_path, debug_monitor, _, raw_passed = debug_monitors[0]
        layered_stages = [
            *raw_stages,
            filtered_layer(debug_path, debug_monitor, raw_passed),
            separability_layer(debug_path, debug_monitor, raw_passed),
        ]
        debug_html, debug_json = write_report(
            layered_stages, training_root / "layered_debug", [corrected])
        debug_payload = json.loads(debug_json.read_text())
        check("one-command debugger localizes raw/filter/separability layers",
              debug_html.exists() and debug_html.stat().st_size > 1000 and
              len(debug_payload["stages"]) == 3 and
              debug_payload["stages"][0]["name"].startswith("L1") and
              debug_payload["stages"][1]["name"].startswith("L2") and
              debug_payload["stages"][2]["name"].startswith("L3"))

        reference_csv = training_root / "other_project.csv"
        reference_labels = ["Fp1", "Fp2", "Fz", "Cz",
                            "FC1", "FC2", "C3", "C4"]
        lines = ["session_time," + ",".join(reference_labels)]
        for sample in range(2500):
            timestamp = sample / 250.0
            channels = [
                8 * np.sin(2 * np.pi * (9 + index * .2) * timestamp)
                + .2 * np.sin(sample * (index + 1))
                for index in range(8)]
            lines.append(
                f"{timestamp:.6f}," + ",".join(f"{value:.8f}" for value in channels))
        reference_csv.write_text("\n".join(lines), encoding="utf-8")
        reference = load_reference_csv(reference_csv, reference_labels)
        reference_stage = reference_layer(
            reference, {"amp_uv": 400, "jump_uv": 75,
                        "flat_uv": .05, "z_max": 4})
        check("unified debugger loads CSV, infers 250 Hz and applies same raw checks",
              reference["data"].shape == (8, 2500) and
              abs(reference["fs"] - 250) < .01 and
              reference_stage["status"] == "PASS" and
              reference_stage["details"]["scale_to_uv"] == 1.0 and
              reference_stage["details"]["kept"] == 3)
        spectral_stage = spectral_comparison_layer(reference, debug_monitor)
        check("cross-montage comparison uses only identically named EEG nodes",
              spectral_stage["status"] == "WARN" and
              spectral_stage["details"]["common_channels"] == ["FP1", "FP2"] and
              not spectral_stage["details"]["reference_band_power_very_low"])
        comparison_html = (
            _source_trace_svg(reference["data"], reference["fs"],
                              reference["labels"], "CSV")
            + _source_trace_svg(debug_monitor.X[0], debug_monitor.fs,
                                debug_monitor.channel_labels, "NPZ")
            + _common_channel_overlay(reference, debug_monitor, "Common"))
        plot_html, _ = write_report(
            [*layered_stages, reference_stage, spectral_stage],
            training_root / "comparison_debug", [corrected],
            comparison_html=comparison_html)
        check("unified HTML plots separate montages plus shared-node overlay",
              plot_html.exists() and plot_html.read_text(encoding="utf-8").count(
                  "<svg") == 3)

    print("\n== Typed errors ==")
    check("error hierarchy", issubclass(bci_sdk.HardwareError, bci_sdk.BCIError)
          and issubclass(bci_sdk.NotCalibratedError, bci_sdk.CalibrationError))
    try:
        bci_sdk.BCISession(source="sim", profile="/nonexistent/*.json")
        check("missing profile raises CalibrationError", False)
    except bci_sdk.CalibrationError:
        check("missing profile raises CalibrationError", True)
    try:
        bci_sdk.BCISession(source="hardware", fallback_to_sim=False)
        check("hardware failure raises HardwareError", False)
    except bci_sdk.HardwareError:
        check("hardware failure raises HardwareError", True)
    s = bci_sdk.BCISession(source="hardware", fallback_to_sim=True)
    check("fallback_to_sim degrades gracefully", "sim" in s.source)

    print("\n== BrainButton callbacks (Gap 3 item 2) ==")
    import pygame; pygame.init(); pygame.display.set_mode((400,300))
    b = bci_sdk.BrainButton("YES", rect=(0,0,120,60))
    fired=[]
    @b.on_brain_select
    def _sel(btn): fired.append(btn.label)
    check("on_brain_select works as a decorator", callable(_sel))
    b.select()
    check("select() fires callback", fired==["YES"])
    enter,leave=[],[]
    b.on_focus(lambda x: enter.append(1)); b.on_blur(lambda x: leave.append(1))
    b.arm(); check("arm() fires on_focus", len(enter)==1)
    check("is_armed", b.is_armed)
    b.disarm(); check("disarm() fires on_blur", len(leave)==1)
    b2 = bci_sdk.BrainButton("GO", rect=(0,0,120,60), dwell=0.05)
    hits=[]; b2.on_brain_select(lambda x: hits.append(1))
    b2.arm(); time.sleep(0.08)
    check("dwell auto-selects after timeout", b2.update() and hits==[1])
    check("progress reaches 0 after commit", b2.progress==0.0)
    multi=[]
    b3=bci_sdk.BrainButton("M",rect=(0,0,10,10))
    b3.on_brain_select(lambda x: multi.append(1)); b3.on_brain_select(lambda x: multi.append(2))
    b3.select(); check("multiple callbacks all fire", multi==[1,2])
    bad=bci_sdk.BrainButton("B",rect=(0,0,10,10))
    bad.on_brain_select(lambda x: 1/0)
    bad.select(); check("a throwing callback cannot crash the app", True)
    dis=bci_sdk.BrainButton("D",rect=(0,0,10,10)); dis.enabled=False
    d=[]; dis.on_brain_select(lambda x: d.append(1)); dis.select()
    check("disabled button does not fire", d==[])

    print("\n== ButtonGroup ==")
    btns=[bci_sdk.BrainButton(l,rect=(i*60,0,55,40)) for i,l in enumerate("ABCD")]
    g=bci_sdk.ButtonGroup(btns)
    check("len works", len(g)==4)
    g.assign_frequencies([15.0,20.0])
    check("frequencies assigned round-robin",
          [b.freq for b in g]==[15.0,20.0,15.0,20.0])
    cmd=bci_sdk.Command(name="SSVEP_1",index=1)
    check("routes command to matching button", g.handle_command(cmd) is btns[1])
    check("out-of-range command returns None",
          g.handle_command(bci_sdk.Command(name="SSVEP_9",index=9)) is None)
    check("at() hit-tests", g.at((10,10)) is btns[0] and g.at((5000,5000)) is None)
    stim=bci_sdk.StimulusEngine(target_freqs=(15.0,20.0),refresh_hz=60.0)
    g.apply_flicker(stim)
    check("apply_flicker sets tile state", all(isinstance(b.flicker_on,bool) for b in g))

    print("\n== Scanner (2 commands -> many buttons) ==")
    rows=[[bci_sdk.BrainButton(f"{r}{c}",rect=(c*60,r*50,55,40)) for c in range(3)]
          for r in range(2)]
    sc=bci_sdk.Scanner(rows)
    picked=[]
    for row in rows:
        for b in row: b.on_brain_select(lambda x: picked.append(x.label))
    check("starts in ROW stage", sc.stage=="ROW")
    sc.next(); check("next() advances row", sc.row_idx==1)
    sc.select(); check("select() descends to COL", sc.stage=="COL")
    sc.next(); check("next() advances column", sc.col_idx==1)
    sc.select()
    check("select() commits the button", picked==["11"] and sc.stage=="ROW")
    sc.stage="COL"; sc.row_idx=0; sc.col_idx=len(rows[0])
    sc.select(); check("escape cell returns to ROW", sc.stage=="ROW")
    stages=[]; sc.on_stage_change(lambda st: stages.append(st))
    sc.select(); check("on_stage_change fires", stages==["COL"])
    sc.stage="ROW"
    sc.handle_command(bci_sdk.Command(name="SSVEP_0",index=0))
    check("handle_command(0) == next()", sc.row_idx==1)
    sc.handle_command(bci_sdk.Command(name="SSVEP_1",index=1))
    check("handle_command(1) == select()", sc.stage=="COL")
    sc.update_highlight()
    check("highlight marks the scanned row",
          any(b.state=="scan" for b in rows[sc.row_idx]))

    print("\n== BCISession lifecycle ==")
    sess=bci_sdk.BCISession(source="sim")
    check("not running before start", not sess.is_running)
    check("poll() before start returns None", sess.poll() is None)
    sess.start(); check("running after start", sess.is_running)
    cmds=[]; states=[]; raws=[]
    sess.on_command(lambda c: cmds.append(c))
    sess.on_state(lambda s: states.append(s))
    sess.on_raw_window(lambda w: raws.append(w.shape))
    end=time.time()+8
    while time.time()<end:
        sess.poll(); time.sleep(1/60)
    sess.stop()
    check("not running after stop", not sess.is_running)
    check(f"delivered commands ({[str(c) for c in cmds]})", len(cmds)>=1)
    check("Command carries a positive confidence", all(c.confidence>0 for c in cmds))
    check("Command index parsed from name", all(c.index in (0,1) for c in cmds))
    check("state callback fired", len(states)>=1)
    check("raw window callback got (8,750)", raws and raws[0]==(8,750))
    r=sess.report()
    check("report has engine+stream keys",
          any(k.startswith("engine_") for k in r) and any(k.startswith("stream_") for k in r))

    print("\n== Context manager ==")
    with bci_sdk.BCISession(source="sim") as s2:
        check("context manager starts", s2.is_running)
    check("context manager stops", not s2.is_running)

    print("\n== Replay source ==")
    import glob
    if glob.glob(str(ROOT/"dataset"/"calib_*.npz")):
        s3=bci_sdk.BCISession(source="replay")
        check("replay session builds", s3.bridge.backend.name=="replay")
        s3.start(); time.sleep(4.5); s3.poll(); s3.stop()
        check("replay produced windows", s3.bridge.report()["windows_made"]>=1)
    else:
        print("  SKIP  no recording")

    print("\n== Third-party app uses NO internals (the real Gap 3 test) ==")
    demo=(ROOT/"example"/"sdk_demo.py").read_text()
    internal=re.findall(r"^\s*(?:from|import)\s+(ML|EEG|pygame_lib|stream_bridge|TileButton|stimulus)\b",
                        demo, re.M)
    check(f"sdk_demo.py imports no internal modules {internal}", not internal)
    check("sdk_demo.py imports bci_sdk", "import bci_sdk" in demo)
    uses=[a for a in ["BCISession","BrainButton","Scanner","ButtonGroup",
                      "StimulusEngine","open_display"] if f"bci_sdk.{a}" in demo]
    check(f"demo exercises {len(uses)} public APIs", len(uses)>=5)

    pygame.quit()

    # --------------------------------------------------------------
    # from test_step9_profiles_app.py
    # --------------------------------------------------------------
    import os, sys, json, tempfile, time
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from pathlib import Path
    import numpy as np
    import bci_sdk
    from bci_sdk.profiles import UserProfile, list_users, _slug
    from ML.datasets import load_freq_phase, build_freq_map

    tmp=Path(tempfile.mkdtemp())

    print("\n== Freq_Phase-derived label map (bug fix) ==")
    import scipy.io
    freqs=[]
    for j in range(5):
        for i in range(8): freqs.append(round(8.0+i*1.0+j*0.2,1))
    fp=tmp/"Freq_Phase.mat"
    scipy.io.savemat(fp,{'freqs':np.array(freqs),'phases':np.zeros(40)})
    f,ph=load_freq_phase(fp)
    check("loads 40 targets", len(f)==40 and abs(min(f)-8.0)<1e-9 and abs(max(f)-15.8)<1e-9)
    m,miss=build_freq_map(fp,(15.0,20.0))
    check("15 Hz maps to index 7 (NOT 35)", m=={7:0})
    check("20 Hz correctly reported missing", miss==[20.0])
    m2,miss2=build_freq_map(fp,(10.0,12.0))
    check("10/12 Hz both present", m2=={2:0,4:1} and miss2==[])
    try:
        load_freq_phase(tmp/"nope.mat"); check("missing file guarded", False)
    except FileNotFoundError: check("missing file guarded", True)
    bad=tmp/"bad.mat"; scipy.io.savemat(bad,{'nothing':np.zeros(3)})
    try:
        load_freq_phase(bad); check("non-FreqPhase file rejected", False)
    except ValueError: check("non-FreqPhase file rejected", True)

    print("\n== UserProfile ==")
    p=UserProfile("Anson", root=tmp/"profiles")
    check("slug is filesystem-safe", p.slug=="anson")
    check("weird names sanitised", UserProfile("A n/s..on!",root=tmp).slug.replace('_','')!="" )
    check("new user is uncalibrated", not p.is_calibrated and not p.exists)
    check("uncalibrated -> no session kwargs", p.as_session_kwargs()=={})
    p.update_from_analysis({"confidence_threshold":0.0812,"target_freqs":[15.0,20.0],
                            "montage":"cyton8_ssvep","sample_freq":250,
                            "debounce_window":3,"metrics":{"precision":1.0},
                            "idle_minutes":3.7})
    check("saved to disk", p.path.exists())
    check("threshold stored", abs(p.threshold-0.0812)<1e-9)
    check("now calibrated", p.is_calibrated)
    check("session kwargs point at profile", "profile" in p.as_session_kwargs())
    p2=UserProfile("Anson", root=tmp/"profiles")
    check("reloads from disk", abs(p2.threshold-0.0812)<1e-9)
    check("fresh profile is not stale", not p2.is_stale(max_age_days=7))
    p2.data["calibrated_at"]=time.time()-10*86400; p2.save()
    check("old profile flagged stale", UserProfile("Anson",root=tmp/"profiles").is_stale(7))
    rp=p.recording_path()
    check("recording path inside user dir", rp.parent.name=="anson" and rp.suffix==".npz")

    UserProfile("Brian",root=tmp/"profiles").update_from_analysis({"confidence_threshold":0.05})
    users=list_users(root=tmp/"profiles")
    check(f"lists both users {[u.name for u in users]}", len(users)==2)
    check("users are independent",
          abs(UserProfile("Anson",root=tmp/"profiles").threshold-0.0812)<1e-9 and
          abs(UserProfile("Brian",root=tmp/"profiles").threshold-0.05)<1e-9)

    print("\n== Profile drives the session threshold ==")
    real=UserProfile("anson")
    if real.is_calibrated:
        s=bci_sdk.BCISession(source="sim", **real.as_session_kwargs())
        check("session uses THIS user's threshold",
              abs(s.engine.ssvep_clf.confidence_threshold-real.threshold)<1e-9)
        check("session reports calibrated", s.report()["calibrated"])
    else:
        print("  SKIP  no real calibrated profile yet")
    s2=bci_sdk.BCISession(source="sim")
    check("uncalibrated session uses default", s2.engine.ssvep_clf.confidence_threshold==0.15)

    print("\n== App page flow ==")
    import pygame
    from example.bci_app import App
    a=App(source="sim")
    check("opens on USERS page", a.page==App.PAGE_USERS)
    a2=App(source="sim", user="__brand_new__")
    check("new user routed to CALIBRATE", a2.page==App.PAGE_CALIBRATE)
    if real.is_calibrated:
        a3=App(source="sim", user="anson")
        check("calibrated user routed to READY", a3.page==App.PAGE_READY)
    check("uncalibrated user CANNOT reach speller directly",
          App(source="sim", user="__brand_new2__").page != App.PAGE_SPELLER)
    a.win.fill((0,0,0)); a.draw_users()
    a.profile=UserProfile("anson"); a.page=App.PAGE_CALIBRATE
    a.win.fill((0,0,0)); a.draw_calibrate()
    if real.is_calibrated:
        a.page=App.PAGE_READY; a.win.fill((0,0,0)); a.draw_ready()
    check("all pages render without error", True)

    class E: key=None; unicode=""
    e=E(); e.key=pygame.K_DOWN
    a.page=App.PAGE_USERS; a.users=list_users(); a.sel=0
    a.users_key(e)
    check("DOWN moves selection", a.sel==min(1,len(a.users)))

    pygame.quit()
    layer_total('LAYER 5 - PRODUCT AND USABILITY')


# ====================================================================
#  DATA MODE - diagnostics on a real recording
#  (was tools/layer_tests.py)
# ====================================================================
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _v(tag, name, detail=""):
    mark = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]"}[tag]
    print(f"  {mark} {name}")
    if detail:
        for line in str(detail).split("\n"):
            print(f"         {line}")
    return tag


def _load(pattern):
    hits = sorted(glob.glob(str(pattern)))
    if not hits:
        raise FileNotFoundError(f"no recording matched {pattern}")
    d = np.load(hits[-1], allow_pickle=True)
    return (d["X"].astype(np.float64), d["labels"].astype(str),
            int(d["fs"]), str(d["montage"]), Path(hits[-1]).name)


def bandpower(x, fs, lo, hi):
    """Average power in a band. Welch is used because it averages over
    sub-windows, which is far less noisy than a single periodogram."""
    f, p = welch(x, fs=fs, nperseg=min(256, x.shape[-1]))
    m = (f >= lo) & (f <= hi)
    return float(np.mean(p[..., m]))


# =====================================================================
# LAYER 1 — HARDWARE
# =====================================================================
def data_layer1(X, labels, fs, montage):
    """Is the rig recording brain activity at all?

    Covers: dead channels, saturation, bad contact, mains interference,
    flat-lining, channel cross-talk, and the definitive alpha check.
    """
    print("\n" + "=" * 68)
    print("LAYER 1 · HARDWARE AND SIGNAL QUALITY")
    print("=" * 68)
    res = []

    from ML.montage import get
    m = get(montage)
    names = m.get("labels", [f"ch{i}" for i in range(X.shape[1])])
    flat = X.transpose(1, 0, 2).reshape(X.shape[1], -1)

    # 1.1 amplitude range -------------------------------------------------
    print("\n1.1 Per-channel amplitude (RMS)")
    rms = np.std(flat, axis=1)
    bad = []
    for i, r in enumerate(rms):
        state = "ok"
        if r < 1.0:
            state, tag = "DEAD - not connected", FAIL
        elif r > 200:
            state, tag = "SATURATED - bad contact or movement", FAIL
        elif r > 100:
            state, tag = "noisy", WARN
        else:
            tag = PASS
        if tag != PASS:
            bad.append(f"{names[i]}={r:.1f}uV ({state})")
        print(f"      {names[i]:<5} {r:7.1f} uV   {state}")
    res.append(_v(FAIL if any("DEAD" in b or "SAT" in b for b in bad)
                  else (WARN if bad else PASS),
                  "amplitude in 1-200 uV range",
                  "\n".join(bad) if bad else "all channels plausible"))

    # 1.2 flat-line / stuck ADC ------------------------------------------
    print("\n1.2 Flat-line detection")
    stuck = [names[i] for i in range(len(flat))
             if len(np.unique(np.round(flat[i], 3))) < 10]
    res.append(_v(FAIL if stuck else PASS, "no stuck/constant channels",
                  f"stuck: {stuck}" if stuck else "all channels varying"))

    # 1.3 mains interference ---------------------------------------------
    print("\n1.3 Mains interference (50 Hz in HK)")
    worst = []
    for i in range(len(flat)):
        mains = bandpower(flat[i], fs, 48, 52)
        sig = bandpower(flat[i], fs, 8, 30)
        ratio = mains / (sig + 1e-12)
        if ratio > 2.0:
            worst.append(f"{names[i]}: mains/signal = {ratio:.1f}x")
    res.append(_v(FAIL if len(worst) > 2 else (WARN if worst else PASS),
                  "mains not dominating the 8-30 Hz band",
                  "\n".join(worst) if worst else "50 Hz under control"))

    # 1.4 channel cross-talk / bridging -----------------------------------
    print("\n1.4 Electrode bridging (channels too similar)")
    C = np.corrcoef(flat)
    bridged = [f"{names[i]}~{names[j]} r={C[i, j]:.2f}"
               for i in range(len(C)) for j in range(i + 1, len(C))
               if C[i, j] > 0.98]
    res.append(_v(WARN if bridged else PASS, "no bridged electrode pairs",
                  "\n".join(bridged) if bridged
                  else "channels are electrically distinct"))

    # 1.5 alpha blocking — the definitive test ----------------------------
    print("\n1.5 Eyes-closed alpha (the definitive proof of real EEG)")
    print("      requires a recording with eyes-open and eyes-closed blocks")
    has = [l for l in np.unique(labels) if "close" in l.lower() or "open" in l.lower()]
    if len(has) >= 2:
        oc = X[np.array([("close" in l.lower()) for l in labels])]
        op = X[np.array([("open" in l.lower()) for l in labels])]
        post = m.get("ssvep", [len(names) - 2, len(names) - 1])
        a_c = np.mean([bandpower(t[post], fs, 8, 13) for t in oc])
        a_o = np.mean([bandpower(t[post], fs, 8, 13) for t in op])
        ratio = a_c / (a_o + 1e-12)
        res.append(_v(PASS if ratio > 1.5 else FAIL,
                      f"alpha increases with eyes closed ({ratio:.2f}x)",
                      "expect >1.5x. If not, electrodes are not on scalp "
                      "properly or are not posterior enough."))
    else:
        res.append(_v(WARN, "alpha test skipped",
                      "record 15 s eyes-open then 15 s eyes-closed and label "
                      "them 'eyes_open' / 'eyes_closed' to enable this"))
    return res


# =====================================================================
# LAYER 2 — SIGNAL / SEPARABILITY
# =====================================================================
def data_layer2(X, labels, fs, montage):
    """Is there a measurable difference between the two intentions,
    BEFORE any classifier is involved?

    Covers: ERD presence, lateralisation, effect size, per-channel
    contribution, class balance, and idle separability.
    """
    print("\n" + "=" * 68)
    print("LAYER 2 · SIGNAL SEPARABILITY  (no classifier involved)")
    print("=" * 68)
    res = []

    from ML.montage import get
    m = get(montage)
    names = m.get("labels", [f"ch{i}" for i in range(X.shape[1])])

    t0 = X[labels == "target_0"]
    t1 = X[labels == "target_1"]
    idle = X[np.isin(labels, ["idle", "idle_distracted"])]

    # 2.1 class balance ---------------------------------------------------
    print("\n2.1 Class balance")
    n0, n1 = len(t0), len(t1)
    bal = min(n0, n1) / max(n0, n1, 1)
    res.append(_v(PASS if bal > 0.8 and min(n0, n1) >= 8 else
                  (WARN if min(n0, n1) >= 8 else FAIL),
                  f"balanced classes (target_0={n0}, target_1={n1})",
                  "need >=8 per class; imbalance biases every metric"))
    if min(n0, n1) < 2:
        return res

    # 2.2 band power per channel ------------------------------------------
    print("\n2.2 Mu/beta band power by channel  (log scale)")
    sos = butter(4, [8, 30], btype="band", fs=fs, output="sos")
    f0 = sosfiltfilt(sos, t0, axis=-1)
    f1 = sosfiltfilt(sos, t1, axis=-1)
    p0 = np.log(np.var(f0, axis=2) + 1e-12)
    p1 = np.log(np.var(f1, axis=2) + 1e-12)

    print(f"      {'ch':<6}{'left-imag':>11}{'right-imag':>12}{'diff':>9}"
          f"{'Cohen d':>10}")
    ds = []
    for i in range(X.shape[1]):
        pooled = np.sqrt((p0[:, i].var() + p1[:, i].var()) / 2) + 1e-12
        d = (p0[:, i].mean() - p1[:, i].mean()) / pooled
        ds.append(d)
        print(f"      {names[i]:<6}{p0[:, i].mean():>11.3f}"
              f"{p1[:, i].mean():>12.3f}{p0[:, i].mean()-p1[:, i].mean():>9.3f}"
              f"{d:>10.2f}")

    # 2.3 lateralisation --------------------------------------------------
    print("\n2.3 Contralateral lateralisation")
    lat = m.get("motor_lateral")
    if lat and len(lat) == 2:
        c3, c4 = lat
        # imagining LEFT hand should REDUCE power at the RIGHT cortex (C4)
        left_c4 = p0[:, c4].mean()
        right_c4 = p1[:, c4].mean()
        left_c3 = p0[:, c3].mean()
        right_c3 = p1[:, c3].mean()
        ok = (left_c4 < right_c4) and (right_c3 < left_c3)
        res.append(_v(PASS if ok else WARN,
                      "ERD appears contralaterally",
                      f"{names[c4]}: left-imag {left_c4:.3f} vs right-imag "
                      f"{right_c4:.3f}  (expect left LOWER)\n"
                      f"{names[c3]}: right-imag {right_c3:.3f} vs left-imag "
                      f"{left_c3:.3f}  (expect right LOWER)\n"
                      "wrong direction can mean swapped electrodes or "
                      "swapped cue labels"))
    else:
        res.append(_v(WARN, "lateralisation skipped",
                      "montage has no motor_lateral pair"))

    # 2.4 overall effect size — THE GO/NO-GO GATE -------------------------
    print("\n2.4 Overall separability  (THE GO/NO-GO GATE)")
    best = max(abs(np.array(ds)))
    bi = int(np.argmax(np.abs(ds)))
    # multivariate distance across all motor channels
    mu0, mu1 = p0.mean(0), p1.mean(0)
    pooled_cov = (np.cov(p0.T) + np.cov(p1.T)) / 2
    try:
        maha = float(np.sqrt((mu0 - mu1) @ np.linalg.pinv(pooled_cov) @ (mu0 - mu1)))
    except Exception:
        maha = float("nan")
    tag = PASS if best > 1.5 else (WARN if best > 0.8 else FAIL)
    res.append(_v(tag, f"best single-channel Cohen d = {best:.2f} ({names[bi]})",
                  f"multivariate Mahalanobis distance = {maha:.2f}\n"
                  "d > 1.5 strongly separable -> proceed to layer 3\n"
                  "d 0.8-1.5 marginal -> more trials, check placement\n"
                  "d < 0.8 NOT separable -> fix layer 1 or reconsider paradigm"))

    # 2.5 statistical significance ---------------------------------------
    print("\n2.5 Is the difference statistically real?")
    t, pv = stats.ttest_ind(p0[:, bi], p1[:, bi])
    res.append(_v(PASS if pv < 0.05 else FAIL,
                  f"t-test on best channel: p = {pv:.4g}",
                  "p < 0.05 means the difference is unlikely to be chance"))

    # 2.6 idle separability ----------------------------------------------
    print("\n2.6 Can rest be told apart from imagery?")
    if len(idle) >= 5:
        fi = sosfiltfilt(sos, idle, axis=-1)
        pi = np.log(np.var(fi, axis=2) + 1e-12)
        act = np.concatenate([p0[:, bi], p1[:, bi]])
        pooled = np.sqrt((act.var() + pi[:, bi].var()) / 2) + 1e-12
        d_idle = abs(act.mean() - pi[:, bi].mean()) / pooled
        res.append(_v(PASS if d_idle > 0.8 else WARN,
                      f"idle vs active Cohen d = {d_idle:.2f}",
                      "needed so the system can output NO_ACTION; "
                      "low d means it cannot tell resting from trying"))
    else:
        res.append(_v(WARN, "idle test skipped", "no idle windows in recording"))
    return res


# =====================================================================
# LAYER 3 — MODEL
# =====================================================================
def data_layer3(user, n_splits=4, n_perm=200):
    """Can a classifier learn the difference, and is the score real?

    Covers: leakage-safe CV, significance vs chance, confidence intervals,
    pipeline comparison, learning curve, and cross-session stability.
    """
    print("\n" + "=" * 68)
    print("LAYER 3 · OFFLINE MODEL ACCURACY")
    print("=" * 68)
    res = []

    from ML.train_local import load_user_data
    from ML.evaluate import grouped_folds
    from ML.mi_advanced import PIPELINES

    d = load_user_data(user)
    X, y, g = d["X"], d["y"], d["groups"]
    fs = d["fs"]
    print(f"\n  {len(X)} trials, {len(np.unique(g))} blocks, "
          f"{len(d['files'])} session(s)")

    # 3.1 enough independent blocks ---------------------------------------
    print("\n3.1 Enough independent recording blocks for honest CV?")
    bpc = min(len(np.unique(g[y == c])) for c in (0, 1))
    res.append(_v(PASS if bpc >= 3 else FAIL,
                  f"{bpc} block(s) per class",
                  "need >=3 separate blocks per class, otherwise CV trains and\n"
                  "tests on single blocks and the score is meaningless\n"
                  "(observed: 100% and 26.6% from exactly this situation)"))

    # 3.2 grouped CV for every pipeline -----------------------------------
    print("\n3.2 Grouped cross-validation (no leakage)")
    print(f"      {'pipeline':<14}{'accuracy':>20}")
    scores = {}
    for name, fac in PIPELINES.items():
        accs = []
        for tr, te in grouped_folds(g, y, n_splits=n_splits):
            try:
                p = fac(sample_freq=fs); p.fit(X[tr], y[tr])
                accs.append(float(np.mean(p.predict(X[te]) == y[te])))
            except Exception:
                pass
        if accs:
            scores[name] = accs
            print(f"      {name:<14}{np.mean(accs):>12.1%} +/-{np.std(accs):<6.1%}")
    if not scores:
        return res + [_v(FAIL, "no pipeline trained", "check the recording")]

    best = max(scores, key=lambda k: np.mean(scores[k]))
    acc = float(np.mean(scores[best]))
    res.append(_v(PASS if acc > 0.65 else (WARN if acc > 0.55 else FAIL),
                  f"best pipeline {best} at {acc:.1%}",
                  "chance = 50%; >65% is the usable threshold"))

    # 3.3 significance vs chance ------------------------------------------
    print("\n3.3 Is the accuracy significantly above chance?")
    n_test = int(len(X) / n_splits)
    n_correct = int(round(acc * n_test))
    bt = stats.binomtest(n_correct, n_test, 0.5, alternative="greater")
    res.append(_v(PASS if bt.pvalue < 0.05 else FAIL,
                  f"binomial test p = {bt.pvalue:.4g}",
                  f"{n_correct}/{n_test} correct. p<0.05 means the result is\n"
                  "unlikely to come from a coin flip"))

    # 3.4 permutation test -------------------------------------------------
    print(f"\n3.4 Permutation test ({n_perm} label shuffles)")
    null = []
    rng = np.random.default_rng(0)
    fac = PIPELINES[best]
    for _ in range(n_perm):
        yp = rng.permutation(y)
        a = []
        for tr, te in grouped_folds(g, yp, n_splits=n_splits):
            try:
                p = fac(sample_freq=fs); p.fit(X[tr], yp[tr])
                a.append(float(np.mean(p.predict(X[te]) == yp[te])))
            except Exception:
                pass
        if a:
            null.append(np.mean(a))
    if null:
        pv = (np.sum(np.array(null) >= acc) + 1) / (len(null) + 1)
        res.append(_v(PASS if pv < 0.05 else FAIL,
                      f"permutation p = {pv:.4g}",
                      f"null distribution mean {np.mean(null):.1%}, "
                      f"95th pct {np.percentile(null,95):.1%}\n"
                      "this is the strictest test: it rebuilds the whole\n"
                      "pipeline on shuffled labels"))

    # 3.5 bootstrap CI ----------------------------------------------------
    print("\n3.5 Confidence interval (bootstrap)")
    a = np.array(scores[best])
    boot = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res.append(_v(PASS if lo > 0.5 else WARN,
                  f"{acc:.1%} (95% CI {lo:.1%} - {hi:.1%})",
                  "report the interval, not the point estimate;\n"
                  "if the lower bound touches 50% the result is not solid"))

    # 3.6 learning curve --------------------------------------------------
    print("\n3.6 Learning curve — is more data still helping?")
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(2, int(len(np.unique(g)) * frac))
        keep = np.isin(g, np.unique(g)[:k])
        a2 = []
        for tr, te in grouped_folds(g[keep], y[keep], n_splits=min(3, k)):
            try:
                p = fac(sample_freq=fs)
                p.fit(X[keep][tr], y[keep][tr])
                a2.append(float(np.mean(p.predict(X[keep][te]) == y[keep][te])))
            except Exception:
                pass
        if a2:
            print(f"      {int(frac*100):>3}% of blocks -> {np.mean(a2):.1%}")
    res.append(_v(PASS, "learning curve computed",
                  "still rising = record more; flat = at the ceiling"))

    # 3.7 cross-session -----------------------------------------------------
    print("\n3.7 Cross-session stability")
    if len(d["files"]) >= 2:
        res.append(_v(WARN, "manual check required",
                      "train on session 1, test on session 2.\n"
                      "Within ~10 points = electrodes are repeatable;\n"
                      "a big drop means recalibration is needed every session"))
    else:
        res.append(_v(WARN, "only one session",
                      "record on a second day to test repeatability"))
    return res


# =====================================================================
# LAYER 4 / 5 — protocols (require a live human)
# =====================================================================
def data_layer4():
    print("\n" + "=" * 68)
    print("LAYER 4 · ONLINE PERFORMANCE   (live session required)")
    print("=" * 68)
    print("""
4.1 Cued online accuracy
    Run:  python EEG/data_record.py --file realistic_ui --trials 20 --cue 6
    Metric: proportion of trials where the cued button was selected first
    PASS: >= 70%
    Catches: model works offline but not with feedback or fatigue

4.2 False positives per minute of IDLE      <-- the headline metric
    Protocol: wear the headset, do nothing for 5 minutes.
              Rest, read, look around, blink normally.
    Metric: commands emitted per minute
    PASS: < 0.5 / min
    Catches: threshold too low. A forced-argmax classifier measured 100%
             accuracy on attentive trials while firing on 141 of 141 idle
             windows - accuracy alone would have called that a success.

4.3 Time to selection
    Metric: seconds from cue to correct selection, including retries
    Report: median and 90th percentile
    Catches: technically accurate but unusably slow

4.4 Information Transfer Rate (Wolpaw ITR)
    Metric: bits/min, combining accuracy and speed
    Catches: a system that trades one for the other without net gain

4.5 Blink and artifact rejection
    Protocol: 20 deliberate blinks, 10 jaw clenches, 10 head turns during rest
    PASS: >= 90% flagged, 0 producing a command
    Catches: artifacts being read as intent

4.6 Error recovery
    Protocol: after a wrong selection, can the user correct within 2 attempts?
    Catches: a UI with no way back

4.7 Sustained performance (drift)
    Protocol: repeat 4.1 at minute 0, 10 and 20 of one session
    PASS: accuracy drop < 15 points
    Catches: electrode drying, fatigue, gel drift
""")


def data_layer5():
    print("\n" + "=" * 68)
    print("LAYER 5 · USABILITY AND PRODUCT   (participants required)")
    print("=" * 68)
    print("""
5.1 Task completion
    Task: select 5 specific actions in the game HUD in a given order
    Metrics: completion rate, errors, total time
    PASS: >= 80% completion unaided

5.2 Subjective workload
    Instrument: NASA-TLX, or a 1-5 fatigue scale after each block
    PASS: mental demand <= 3/5 after 10 minutes
    Catches: works for 2 minutes, exhausting after 10. No objective
             metric captures this, and for MI it is the usual limit.

5.3 Learning effect
    Protocol: same task on 3 separate days
    Expected: time-to-selection improves; BCI skill is trainable
    Catches: a system only the developer can drive

5.4 Multi-user validation
    Participants: 3-5, each with their own calibration
    Report: per-user accuracy and the spread
    Catches: a single-subject result presented as a general library.
             ~15-30% of people show little usable MI ("BCI illiteracy"),
             so a 3-subject study may legitimately include a failure.

5.5 Developer integration
    Protocol: someone unfamiliar builds a brain-controlled button using
              only the public API and the README
    PASS: working app without reading library source

5.6 Installation
    Run: pip install into a clean venv, then import bci_sdk
    PASS: imports and runs with no source checkout
    CURRENT STATUS: FAILS - the wheel omits EEG/ and pygame_lib/

5.7 Safety and comfort
    Checks: session length limits, electrode pressure, skin condition after
            removal, clear stop instruction
""")

CODE_LAYERS = {1: layer1, 2: layer2, 3: layer3, 4: layer4, 5: layer5}


def main():
    ap = argparse.ArgumentParser(
        description="Layered BCI tests. Bare = test the CODE; "
                    "--user/--recording = diagnose real DATA.")
    ap.add_argument("--layer", type=int, choices=sorted(CODE_LAYERS))
    ap.add_argument("--user", help="diagnose this user\'s latest recording")
    ap.add_argument("--recording", help="diagnose this .npz (glob allowed)")
    ap.add_argument("--code", action="store_true",
                    help="also run the code tests when diagnosing data")
    ap.add_argument("--perm", type=int, default=200,
                    help="label shuffles for the layer-3 permutation test")
    a = ap.parse_args()

    data_mode = bool(a.user or a.recording)
    want = [a.layer] if a.layer else sorted(CODE_LAYERS)

    if not data_mode or a.code:
        for n in want:
            CODE_LAYERS[n]()
        print("\n" + "=" * 70)
        print(f"  CODE: {ok} passed, {len(fail)} failed")
        if fail:
            print("  failed:", fail)
        print("=" * 70)

    if data_mode:
        pattern = a.recording or str(ROOT / "profiles" / a.user.lower()
                                     / "calib_*.npz")
        res = []
        if any(n in want for n in (1, 2)):
            X, labels, fs, montage, name = _load(pattern)
            print(f"\nrecording: {name}   {X.shape}  fs={fs}  "
                  f"montage={montage}")
            if 1 in want:
                res += data_layer1(X, labels, fs, montage)
            if 2 in want:
                res += data_layer2(X, labels, fs, montage)
        if 3 in want:
            if not a.user:
                print("\n  layer 3 needs --user (it trains on every session)")
            else:
                res += data_layer3(a.user, n_perm=a.perm)
        if 4 in want:
            data_layer4()
        if 5 in want:
            data_layer5()

        n_fail = sum(1 for r in res if r == FAIL)
        n_warn = sum(1 for r in res if r == WARN)
        print("\n" + "=" * 70)
        print(f"  DATA: {sum(1 for r in res if r == PASS)} pass, "
              f"{n_warn} warn, {n_fail} fail")
        if n_fail:
            print("  A data failure is a finding, not a bug. Re-record before "
                  "training.")
        print("=" * 70)

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
