"""
Author: Anson
Created on: 7/10/2026
Purpose: full dress rehearsal with no headset, no electrodes, no subject.

    python tools/dry_run.py
    python tools/dry_run.py --keep        # leave the generated profile on disk

WHY
---
Every stage of this system can be exercised on synthetic data, and the point
of doing so is NOT to find out whether motor imagery works - it cannot tell
you that. It is to find out whether anything is wired wrong, before a session
with a real person is spent discovering it.

The two kinds of failure are worth keeping separate in your head:

    PLUMBING   windowing, sample rate, channel order, units, gates, the tree,
               callbacks, file formats, the training path. All deterministic,
               all checkable here.
    SIGNAL     whether this person's C3/C4 separates. Needs a brain. Nothing
               in this file speaks to it, and any accuracy printed below is a
               property of the simulator, not of you.

Treat a green run as "the software is ready for a subject", never as
"the BCI works".

STAGES
------
  1  component tests            353 assertions, no data involved
  2  cued calibration           flashing-button session -> calib_*.npz
  3  recording diagnostics      the Layer 1-3 checks a real session gets
  4  local training             pipeline bake-off -> model.joblib
  5  real-time decision         the trained model drives actual buttons
  6  hardware loopback          a known signal end to end
"""

import argparse
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
for _sub in ("", "EEG"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS = []


def stage(n, title):
    print("\n" + "=" * 72)
    print(f"STAGE {n} - {title}")
    print("=" * 72)


def verdict(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# =====================================================================
def s1_component_tests():
    stage(1, "COMPONENT TESTS  (no data involved)")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "test_bci.py")],
                       capture_output=True, text=True,
                       env={**__import__("os").environ,
                            "SDL_VIDEODRIVER": "dummy"})
    line = [l for l in r.stdout.splitlines() if "CODE:" in l]
    print("  " + (line[-1].strip() if line else "no summary line"))
    verdict("component suite passes", r.returncode == 0)


# =====================================================================
def s2_calibration(profiles, subject="dryrun", trials=28, erd=0.75,
                   cue_s=3.2):
    stage(2, "CUED CALIBRATION  (the flow a real session uses)")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dr", ROOT / "EEG" / "data_record.py")
    dr = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(dr)
    except SystemExit:
        pass

    class ErdBackend:
        """Stands in for the board. Produces real ERD so the later stages
        have something to find - a flat noise source would make training
        look broken for the wrong reason."""
        name = "erd-sim"

        def __init__(self, fs=250, n_channels=8, seed=0):
            self.fs, self.n_channels = fs, n_channels
            self.rng = np.random.default_rng(seed)
            self.hand = None
            self._last = None

        def set_hand(self, h):
            self.hand = h

        def start(self):
            self._last = time.time()

        def stop(self):
            pass

        def info(self):
            return {"fs": self.fs, "n_channels": self.n_channels,
                    "backend": self.name}

        def read_available(self):
            now = time.time()
            n = int((now - self._last) * self.fs)
            if n <= 0:
                return None
            self._last += n / self.fs
            x = self.rng.normal(0, 8.0, (self.n_channels, n))
            if self.hand is not None:                 # contralateral ERD
                site = 4 if self.hand == 0 else 2     # left hand -> C4
                x[site] *= erd
            return x

    be = ErdBackend()
    cap = dr.SessionCapture(be).start()
    order = dr.build_hand_order(trials)
    print(f"  {trials} trials, balanced "
          f"{order.count(0)} LEFT / {order.count(1)} RIGHT")
    print(f"  (train_local needs 8 per class AFTER screening drops artifacts,"
          f" so {trials} is sized with headroom)")

    # The cue must be at least as long as the epoch window. Cueing for 0.3 s
    # and then cutting a 3 s epoch means 90% of every epoch is rest, which
    # trains a model on mostly nothing - it scored 44.8% that way, and the
    # failure looks exactly like "motor imagery does not work".
    print(f"  cue {cue_s:.1f}s per trial (>= the 3.0s epoch window), "
          f"~{trials * (cue_s + 0.1) + 3:.0f}s total")
    log = []
    for i, hand in enumerate(order):
        be.set_hand(hand)
        idx = cap.n_samples
        time.sleep(cue_s)
        hname, label = dr.HAND_LABELS[hand]
        log.append({"trial": i + 1, "cue_sample": idx, "cue_start_ms": 0,
                    "hand": hname, "label": label, "correct": 1})
        be.set_hand(None)
        time.sleep(0.05)
    time.sleep(3.1)                                   # let the last epoch fill
    stream = cap.stop()

    X, y, labels = dr.epoch_from_log(stream, be.fs, log, t0_ms=0)
    print(f"  captured {stream.shape[1] / be.fs:.1f}s -> {len(X)} epochs "
          f"of {X.shape[1:] if len(X) else '-'}")
    verdict("epochs cut from the stream", len(X) >= trials * 0.8,
            f"{len(X)}/{trials}")
    if not len(X):
        return None

    # split into blocks so grouped CV has something to hold out
    # array_split, not a fixed stride: a stride leaves a remainder, and a
    # 1-trial file is what Layer 2 then reads (the loader takes the newest
    # recording), reporting "target_0=1, target_1=0" on a session that is
    # actually fine.
    paths = []
    for sl in np.array_split(np.arange(len(X)), 3):
        if len(sl) < 4:
            continue
        paths.append(dr.save_session(X[sl], y[sl], labels[sl], subject, be.fs,
                                     "cyton8_motor", profiles / subject))
        time.sleep(1.05)                              # distinct timestamps
    print(f"  wrote {len(paths)} recording(s) to profiles/{subject}/")
    verdict("same file format as calibration", len(paths) >= 2)
    return paths[-1] if paths else None


# =====================================================================
def s3_diagnostics(profiles, subject):
    stage(3, "RECORDING DIAGNOSTICS  (what a real session gets)")
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_bci.py"),
         "--recording", str(profiles / subject / "calib_*.npz"), "--layer", "2"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "SDL_VIDEODRIVER": "dummy"})
    for l in r.stdout.splitlines():
        if any(k in l for k in ("Cohen", "GO/NO-GO", "balance", "DATA:",
                                "lateralis")):
            print("  " + l.strip())
    verdict("diagnostics run against the recording", "DATA:" in r.stdout)


# =====================================================================
def s4_training(profiles, subject):
    stage(4, "LOCAL TRAINING  (pipeline bake-off)")
    from ML.train_local import train_user
    res = train_user(subject, root=profiles, verbose=True)
    verdict("a model was trained and saved", res is not None)
    return res


# =====================================================================
def s5_realtime(profiles, subject):
    stage(5, "REAL-TIME DECISION  (trained model -> buttons)")
    from ML.local_model import LocalMIModel
    from bci_sdk import SelectionTree, RealtimeDecider, ArtifactDetector

    model = LocalMIModel.load(subject, root=profiles, confidence_threshold=0.55)
    print(f"  {model}")

    class Btn:
        def __init__(self, n, x):
            self.n, self.fired = n, 0
            self.rect = type("R", (), {"centerx": x})()

        def trigger(self):
            self.fired += 1
            print(f"      >>> BUTTON FIRED: {self.n}")

        def __repr__(self):
            return self.n

    buttons = [Btn(n, x) for n, x in
               zip(("rock", "paper", "scissors"), (100, 400, 700))]
    tree = SelectionTree(buttons)
    print(f"  {tree}")
    clk = [0.0]
    dec = RealtimeDecider(model, tree,
                          artifact=ArtifactDetector(sample_freq=250),
                          clock=lambda: clk[0], debounce_window=3,
                          refractory_ms=1200)

    rng = np.random.default_rng(9)
    for _ in range(18):
        w = rng.normal(0, 8.0, (8, 750))
        w[2] *= 0.80                                  # right-hand imagery
        dec.step(w)
        clk[0] += 1000
    rep = dec.report()
    print(f"  {rep}")
    verdict("windows produced commands", rep["dispatched"] > 0)
    verdict("a button was selected", sum(b.fired for b in buttons) > 0)


# =====================================================================
def s6_loopback():
    stage(6, "HARDWARE LOOPBACK  (known signal, known answer)")
    try:
        import brainflow  # noqa: F401
    except ImportError:
        print("  brainflow not installed - skipping")
        print("  install it and run:  python tools/test_signal_check.py "
              "--board synthetic --decide")
        verdict("loopback available", True, "(skipped, not a failure)")
        return
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "test_signal_check.py"),
         "--board", "synthetic", "--seconds", "6"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "SDL_VIDEODRIVER": "dummy"})
    for l in r.stdout.splitlines():
        if "effective rate" in l or "channel frequencies" in l:
            print("  " + l.strip())
    verdict("test signal reads back correctly", r.returncode == 0)


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="full dry run, no hardware")
    ap.add_argument("--keep", action="store_true",
                    help="leave the generated profile on disk")
    ap.add_argument("--subject", default="dryrun")
    ap.add_argument("--trials", type=int, default=28,
                    help="28 leaves headroom: train_local needs 8 per class "
                         "AFTER screening, and screening typically removes "
                         "10-20%")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="bci_dryrun_"))
    profiles = tmp / "profiles"
    print(__doc__.split("STAGES")[0].strip())
    print(f"\nworking in {profiles}")

    s1_component_tests()
    got = s2_calibration(profiles, a.subject, trials=a.trials)
    if got:
        s3_diagnostics(profiles, a.subject)
        if s4_training(profiles, a.subject):
            s5_realtime(profiles, a.subject)
    s6_loopback()

    print("\n" + "=" * 72)
    ok = sum(1 for _, v in RESULTS if v)
    for name, v in RESULTS:
        print(f"  {'OK  ' if v else 'FAIL'}  {name}")
    print(f"\n  {ok}/{len(RESULTS)} stages green")
    print("\n  This says the PLUMBING is sound. It says nothing about whether")
    print("  motor imagery works for you - that needs a brain, and it is")
    print("  decided at Layer 2 of your first real session.")
    print("=" * 72)

    if a.keep:
        print(f"\n  profile kept at {profiles}")
    else:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
