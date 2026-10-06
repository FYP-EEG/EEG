"""
Author: Anson
Created on: 6/10/2026
Purpose: validate the real-time decision path against a KNOWN hardware signal.

    # OpenBCI GUI "Synthetic" data source (BrainFlow synthetic board)
    python tools/test_signal_check.py --board synthetic --decide

    # real Cyton, on-board square-wave generator
    python tools/test_signal_check.py --board cyton --serial-port COM4 --signal = --decide

WHAT THIS ANSWERS
-----------------
"Nothing happened" has five possible causes and they all look the same. A
generator whose output is known in advance separates them:

    frequency off by a fixed ratio -> sample rate is wrong
    detected on the wrong channel  -> channel mapping is wrong
    amplitude off by ~1e6          -> unit scaling is wrong
    nothing detected at all        -> windowing or streaming broken
    detected but no command        -> gates or thresholds
    command but no button          -> selection tree or callbacks

Only the last two are about the BCI. The rest are plumbing, and plumbing is
what a lab session is usually lost to.

THE TWO GENERATORS ARE NOT ALIKE  (both measured, not taken from the docs)
-------------------------------------------------------------------------
  synthetic  BrainFlow SYNTHETIC_BOARD, which is what the GUI's "Synthetic"
             source uses. SINE per channel, 5 Hz apart. Measured: ch0 4.88 Hz
             12 uV, ch1 10.01 Hz 28 uV, ch2 14.89 Hz 38 uV ... Its rate is
             250 Hz (not the 256 Hz older BrainFlow docs state) and it exposes
             16 EEG channels, not 8 -- channel lists written for the Cyton read
             the wrong rows.

  cyton      the ADS1299 on-board generator. SQUARE wave, ~1 Hz slow / ~2 Hz
             fast, 1855-1865 uVrms at 1x amplitude. That is ~19x the 100 uV
             artifact threshold, so --decide raises it for the test. Its
             fundamental is below the 8-30 Hz motor band, so it proves the
             acquisition chain, not the classifier.

The probe itself lives in bci_sdk/live.py, so the apps and this tool
share one implementation.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: returned when no sinusoid stands out from the noise floor
NO_ACTION = -1


from bci_sdk.live import SineProbe, SineProbeModel  # noqa: E402


#: serial command -> (description, expected fundamental Hz, expected uVrms)
CYTON_TEST_SIGNALS = {
    "0":  ("internal GND - measures the noise floor", None, None),
    "p":  ("DC signal", 0.0, None),
    "-":  ("square, 1x amplitude, slow pulse", 1.0, 1860.0),
    "=":  ("square, 1x amplitude, fast pulse", 2.0, 1860.0),
    "[":  ("square, 2x amplitude, slow pulse", 1.0, 3697.0),
    "]":  ("square, 2x amplitude, fast pulse", 2.0, 3697.0),
}

#: BrainFlow synthetic board: channel i carries a sine at (i+1)*5 Hz
SYNTHETIC_CHANNEL_HZ = {i: (i + 1) * 5.0 for i in range(16)}

def classify_waveform(window, sample_freq=250, channel=None, band=(0.5, 45.0)):
    """Tell a sine from a square from noise, using harmonic structure.

    An ideal square wave contains only ODD harmonics, at relative amplitudes
    1, 1/3, 1/5 ... A sine has none. Measuring the 3rd harmonic against the
    fundamental separates them cleanly:

        sine   -> h3/h1 near 0
        square -> h3/h1 near 0.33

    :return: dict(kind, fundamental, h3_ratio, h5_ratio, snr, channel)
    """
    probe = SineProbe(sample_freq=sample_freq, band=band, min_snr=0.0)
    d = probe.detect(window)
    f0, ch = d["freq"], d["channel"] if channel is None else channel
    out = {"kind": "noise", "fundamental": f0, "h3_ratio": 0.0,
           "h5_ratio": 0.0, "snr": d["snr"], "channel": ch}
    if f0 is None or d["snr"] < SineProbe.DEFAULT_MIN_SNR:
        return out

    x = np.asarray(window, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    sig = x[ch] - x[ch].mean()
    n = len(sig)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_freq)

    def amp_at(f):
        if f <= 0 or f >= freqs[-1]:
            return 0.0
        k = int(np.argmin(np.abs(freqs - f)))
        lo, hi = max(0, k - 1), min(len(spec), k + 2)
        return float(np.max(spec[lo:hi]))

    a1 = amp_at(f0) or 1e-30
    out["h3_ratio"] = amp_at(3 * f0) / a1
    out["h5_ratio"] = amp_at(5 * f0) / a1
    # 0.33 is the ideal square ratio; 0.15 sits well clear of a clean sine
    out["kind"] = "square" if out["h3_ratio"] > 0.15 else "sine"
    return out


PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"


def capture(board_name, serial_port, signal, seconds, window):
    """Stream from BrainFlow and return (data, fs, eeg_rows)."""
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

    BoardShim.disable_board_logger()
    ids = {"synthetic": BoardIds.SYNTHETIC_BOARD.value,
           "cyton": BoardIds.CYTON_BOARD.value,
           "cyton_daisy": BoardIds.CYTON_DAISY_BOARD.value}
    bid = ids[board_name]

    params = BrainFlowInputParams()
    if serial_port:
        params.serial_port = serial_port

    fs = BoardShim.get_sampling_rate(bid)
    rows = BoardShim.get_eeg_channels(bid)
    board = BoardShim(bid, params)
    board.prepare_session()

    if signal and board_name.startswith("cyton"):
        desc = CYTON_TEST_SIGNALS.get(signal, ("unknown", None, None))[0]
        print(f"  configuring board: {signal!r} -> {desc}")
        board.config_board(signal)
        time.sleep(0.5)

    board.start_stream()
    need = int(seconds * fs) + window
    t0 = time.time()
    while time.time() - t0 < seconds + 2.0:
        if board.get_board_data_count() >= need:
            break
        time.sleep(0.1)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    return data, fs, rows


def per_channel_report(data, fs, rows, window, expect_kind, expect_map,
                       tolerance):
    print(f"\n  {'ch':>3}{'row':>5}{'detected Hz':>13}{'expected':>10}"
          f"{'amp uV':>10}{'rms uV':>10}{'kind':>8}{'SNR':>10}  result")
    print("  " + "-" * 82)
    ok_all, seen = True, []
    for i, row in enumerate(rows):
        x = data[row][-window:]
        if len(x) < window:
            continue
        hi = min(120.0, fs / 2.0 - 5.0)
        c = classify_waveform(x[np.newaxis, :], sample_freq=fs,
                              band=(1.0, hi))
        probe = SineProbe(sample_freq=fs, band=(1.0, hi))
        d = probe.detect(x[np.newaxis, :])
        rms = float(np.std(x))
        exp = expect_map.get(i) if expect_map else None

        mark = PASS
        if exp is not None and exp > hi:
            mark, exp = "[skip]", None          # above the search band
        elif exp is not None:
            if d["freq"] is None or abs(d["freq"] - exp) > tolerance:
                mark, ok_all = FAIL, False
        elif not d["detected"]:
            mark = WARN
        if expect_kind and c["kind"] != expect_kind and d["detected"]:
            mark, ok_all = FAIL, False

        f_txt = f"{d['freq']:.2f}" if d["freq"] is not None else "-"
        e_txt = f"{exp:.1f}" if exp is not None else "-"
        print(f"  {i:>3}{row:>5}{f_txt:>13}{e_txt:>10}"
              f"{d['amplitude']:>10.1f}{rms:>10.1f}{c['kind']:>8}"
              f"{d['snr']:>10.0f}  {mark}")
        if d["detected"]:
            seen.append(d["freq"])
    return ok_all, seen


def run_decider(data, fs, rows, window, hop, artifact_threshold,
                probe_channels=None):
    """Replay the capture through the real decision path."""
    from bci_sdk.selection import SelectionTree
    from bci_sdk.runtime import RealtimeDecider, ArtifactDetector

    class Btn:
        def __init__(self, n):
            self.n, self.fired = n, 0

        def trigger(self):
            self.fired += 1

        def __repr__(self):
            return self.n

    buttons = [Btn(x) for x in ("inventory", "message", "shoot", "reload", "map")]
    tree = SelectionTree(buttons)
    model = SineProbeModel(freq_map={10.0: "LEFT", 15.0: "RIGHT"},
                           sample_freq=fs, channels=probe_channels)
    if probe_channels:
        print(f"  probe restricted to rows {probe_channels} "
              f"(10 Hz -> LEFT, 15 Hz -> RIGHT)")
    det = ArtifactDetector(channels=(0, 1), sample_freq=fs,
                           threshold_uv=artifact_threshold)
    clk = [0.0]
    dec = RealtimeDecider(model, tree, artifact=det, debounce_window=3,
                          refractory_ms=1200, clock=lambda: clk[0])

    eeg = data[rows, :]
    print(f"\n  replaying {eeg.shape[1]} samples as {window}/{hop} windows")
    print(f"  artifact threshold raised to {artifact_threshold:.0f} uV")
    print(f"\n  {'t(s)':>6}{'detected Hz':>13}{'command':>9}{'state':>13}  candidates")
    print("  " + "-" * 76)
    n = 0
    for s in range(0, eeg.shape[1] - window + 1, hop):
        r = dec.step(eeg[:, s:s + window])
        clk[0] += hop / fs * 1000.0
        n += 1
        d = model.last or {}
        f = d.get("freq")
        if n <= 14 or r["state"] == "DISPATCHED":
            print(f"  {s / fs:>6.1f}{(f'{f:.2f}' if f else '-'):>13}"
                  f"{str(r['command']):>9}{r['state']:>13}  {r['candidates']}")
    print(f"\n  {dec.report()}")
    print(f"  buttons fired: {[(b.n, b.fired) for b in buttons if b.fired]}")
    return dec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default="synthetic",
                    choices=["synthetic", "cyton", "cyton_daisy"])
    ap.add_argument("--serial-port", default=None)
    ap.add_argument("--signal", default=None,
                    help="Cyton test-signal command: " +
                         " ".join(repr(k) for k in CYTON_TEST_SIGNALS))
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--window", type=int, default=750)
    ap.add_argument("--hop", type=int, default=250)
    ap.add_argument("--tolerance", type=float, default=0.5)
    ap.add_argument("--decide", action="store_true",
                    help="also replay through the real-time decision path")
    a = ap.parse_args()

    print("=" * 86)
    print("TEST SIGNAL CHECK -- known input, known correct output")
    print("=" * 86)

    try:
        t_start = time.time()
        data, fs, rows = capture(a.board, a.serial_port, a.signal,
                                 a.seconds, a.window)
        elapsed = time.time() - t_start
    except Exception as e:
        print(f"\n  {FAIL} could not open the board: {type(e).__name__}: {e}")
        if a.board != "synthetic":
            print("        check the dongle, the serial port and that the "
                  "board is switched to PC")
        return 2

    print(f"\n  board {a.board}  fs={fs} Hz  eeg rows {rows}")
    print(f"  captured {data.shape[1]} samples "
          f"({data.shape[1] / fs:.1f} s)")

    if data.shape[1] < a.window:
        print(f"  {FAIL} fewer samples than one window -- streaming is broken")
        return 1

    # ---- sample-rate sanity, independent of the probe
    measured_fs = data.shape[1] / elapsed
    drift = abs(measured_fs - fs) / fs
    print(f"  {PASS if drift < 0.05 else FAIL} effective rate "
          f"{measured_fs:.1f} Hz vs {fs} Hz declared ({drift:+.1%})")
    if drift >= 0.05:
        print("        a rate error scales every detected frequency by the "
              "same factor -- fix this before trusting anything below")

    if a.board == "synthetic":
        expect_map = {i: SYNTHETIC_CHANNEL_HZ[i]
                      for i in range(min(len(rows), 16))}
        expect_kind = "sine"
        print("  expecting SINE per channel, 5 Hz apart (BrainFlow synthetic)")
    else:
        spec = CYTON_TEST_SIGNALS.get(a.signal or "", (None, None, None))
        expect_map = ({i: spec[1] for i in range(len(rows))}
                      if spec[1] else None)
        expect_kind = "square" if spec[1] else None
        if spec[0]:
            print(f"  expecting {spec[0]}")
            if spec[2]:
                print(f"  expecting about {spec[2]:.0f} uVrms per channel")

    ok, seen = per_channel_report(data, fs, rows, a.window, expect_kind,
                                  expect_map, a.tolerance)

    print()
    if expect_map:
        print(f"  {PASS if ok else FAIL} channel frequencies "
              f"{'all match' if ok else 'DO NOT match'} expectation")
        if not ok:
            print("        a constant ratio error means the sample rate is "
                  "wrong; one bad row means the channel map is wrong")
    elif seen:
        print(f"  {WARN} no expectation given; detected {len(seen)} "
              f"periodic channels")

    if a.decide:
        thr = 100.0
        if a.board != "synthetic" and a.signal in ("-", "=", "[", "]"):
            thr = 20000.0      # the square test signal dwarfs any blink
        pc = [1, 2] if a.board == "synthetic" else None
        run_decider(data, fs, rows, a.window, a.hop, thr, probe_channels=pc)

    print("\n" + "=" * 86)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
