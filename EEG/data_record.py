"""
Author: Anson Li
Created on: 26/6/2026
Purpose: script for easier data recording
Location: project_dir/EEG/data_record.py

Edited on: 20/9/2026
Changes: added --file mode. Runs a script from example/, cues the user with a
         flashing button, and records EEG for the whole session.

TWO MODES
---------
1. Character mode (original, unchanged) - flashes typed/random characters:
       py EEG/data_record.py

2. Cued app mode (new) - runs a real app and cues button targets:
       py EEG/data_record.py --file rockpaperscissors
       py EEG/data_record.py --file rockpaperscissors --trials 20 --cue 5
       py EEG/data_record.py --file realistic_ui --dry-run     (no EEG needed)
       py EEG/data_record.py --list

WHY CUED RECORDING
------------------
To train a classifier you need to know WHICH target the user was attending to,
and WHEN. Watching someone use an app freely gives you EEG with no labels.

So the recorder drives the session: it picks a target, makes that button ripple,
and marks the EEG stream. Everything between the start and end marker is labelled
attention-on-that-button. The user's click is recorded too, which tells you
whether they actually complied with the cue - trials where they clicked the wrong
button should be dropped during analysis.

HOW IT DRIVES AN UNMODIFIED EXAMPLE SCRIPT
------------------------------------------
Example scripts are top-level code with their own `while run:` loop, so importing
one just runs it. Instead of rewriting every example, this hooks
`pygame.display.flip()` - the one call every frame makes - and executes the script
inside a namespace we own. From the hook we can reach the script's Button objects,
drive the cue, and draw an overlay on top of whatever it rendered.

Zero changes required in the example scripts.
"""

import argparse
import csv
import datetime as dt
import random
import threading
import time
import string
import sys
from pathlib import Path

import numpy as np

import pygame
#import constants for easier access to key events
from pygame.locals import *

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example"
sys.path.insert(0, str(ROOT / "pygame_lib"))


def _load_bci():
    """Import data_receive lazily.

    Kept out of module scope so --dry-run works on a machine with no brainflow
    installed, and so `--list` never touches the hardware stack.
    """
    import data_receive as BCI
    return BCI


# =====================================================================
#  CUED APP MODE
# =====================================================================
# =====================================================================
#  SHARED RECORDING FORMAT
# =====================================================================
# Calibration and cued-app recording must produce byte-identical file layouts,
# or ML/train_local.py can read only one of them. Rather than duplicate the
# writer, both call EEG/calibration_record.save(), which is the single
# definition of the format:
#
#     calib_<subject>_<stamp>.npz
#       X               (n_trials, n_channels, n_times) float32, microvolts
#       y               int64 class index
#       labels          "target_0" | "target_1" | "idle" | "artifact"
#       fs, montage, channel_labels, subject, created, target_freqs
#
# WHY THE LABEL IS A HAND AND NOT A BUTTON
# ----------------------------------------
# Motor-imagery classes are body parts, because the cortex is somatotopic:
# the left hand sits in right C4, the right hand in left C3. There is no
# cortical representation of "rock" or "inventory", so a button cannot be a
# class.
#
# With two hand classes and a binary selection tree, a button is reached by a
# SEQUENCE of decisions - rock is left-then-left, scissors is right. A trial
# cued as "rock" is therefore not one label but two, and saving it under a
# single label would mix two different imagined movements into one epoch.
# That is worse than having no label.
#
# So calibration cues one DECISION per trial - "imagine your LEFT hand" -
# while the app highlights the group that decision keeps. The user sees the
# real interface and the real consequence, so it remains a rehearsal of actual
# use, but every epoch carries exactly one hand label.
#
# Feet are deliberately not used. They are reserved for continuous movement
# control later; adding them now would turn a 2-class problem into a 3-class
# one, which measured 70.8% -> 63.2% per decision and would drop csp_lda and
# fbcsp out of the pipeline competition, since that CSP supports 2 classes.


def save_session(X, y, labels, subject, fs, montage_name, out_dir,
                 target_freqs=(0.0, 0.0)):
    """Write cued-app EEG in the SAME format calibration uses."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "calrec", Path(__file__).resolve().parent / "calibration_record.py")
    calrec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calrec)
    return calrec.save(X, y, labels, subject, fs, montage_name,
                       target_freqs, out_dir)


def epoch_from_log(stream, fs, log, t0_ms, window_s=3.0):
    """Cut a continuous recording into trials using the cue log.

    :param stream: (n_channels, n_samples) for the whole session
    :param t0_ms:  pygame tick at which stream sample 0 was captured
    :param log:    rows carrying cue_start_ms and a "label" of target_0/1
    :return: (X, y, labels), ready to hand to save_session
    """
    stream = np.asarray(stream, dtype=np.float64)
    n = int(window_s * fs)
    X, y, labels = [], [], []
    for row in log:
        lab = row.get("label")
        if lab not in ("target_0", "target_1"):
            continue
        if row.get("cue_sample") is not None:
            start = int(row["cue_sample"])          # exact, drift-free
        else:
            start = int((row["cue_start_ms"] - t0_ms) / 1000.0 * fs)
        if start < 0 or start + n > stream.shape[1]:
            continue                       # cue fell outside the recording
        X.append(stream[:, start:start + n])
        y.append(int(lab[-1]))
        labels.append(lab)
    return np.asarray(X), np.asarray(y), np.asarray(labels)

class SessionCapture:
    """Record the whole session continuously and index it by SAMPLE COUNT.

    Why not wall-clock: a cue happens at a pygame tick, but the EEG arrives in
    chunks whose timing drifts against the system clock. Converting ms to a
    sample index accumulates that drift, and a 3 s epoch cut 200 ms late is a
    3 s epoch of the wrong thing.

    Instead, every cue records how many samples had arrived at the moment it
    started. That index is exact by construction, because it comes from the
    same counter the data does.
    """

    def __init__(self, backend):
        self.backend = backend
        info = backend.info()
        self.fs = info["fs"]
        self.n_channels = info["n_channels"]
        self._chunks = []
        self._n = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def n_samples(self):
        with self._lock:
            return self._n

    def start(self):
        self.backend.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="SessionCapture")
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                chunk = self.backend.read_available()
            except Exception:
                chunk = None
            if chunk is not None and chunk.size:
                with self._lock:
                    self._chunks.append(np.asarray(chunk, dtype=np.float32))
                    self._n += chunk.shape[1]
            else:
                time.sleep(0.005)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self.backend.stop()
        except Exception:
            pass
        return self.data()

    def data(self):
        with self._lock:
            if not self._chunks:
                return np.zeros((self.n_channels, 0), dtype=np.float32)
            return np.concatenate(self._chunks, axis=1)


#: which hand presses which anchor button. Two classes only - feet are
#: reserved for continuous movement control later.
HAND_LABELS = {0: ("LEFT", "target_0"), 1: ("RIGHT", "target_1")}


def build_hand_order(n_trials, seed=None):
    """Balanced, shuffled hand sequence: equal LEFT and RIGHT, never a long run.

    A long run of one hand lets the user coast - they stop re-forming the
    imagery and the later trials in the run carry weaker ERD. Shuffling within
    balanced pairs caps the run length at two.
    """
    rng = random.Random(seed)
    order = []
    while len(order) < n_trials:
        pair = [0, 1]
        rng.shuffle(pair)
        order.extend(pair)
    return order[:n_trials]


class CuedRecorder:
    """Drives cue → wait → rest trials over an unmodified example script."""

    CUE = "CUE"
    REST = "REST"
    DONE = "DONE"

    def __init__(self, bci, trials=10, cue_s=5.0, rest_s=2.0, script="app",
                 mi=False, capture=None, anchors=None):
        """
        :param mi: calibration mode. Cues a HAND per trial instead of a button,
            because that is the only thing the cortex distinguishes. "Press
            rock with your left hand" and "press scissors with your left hand"
            are the same signal - left-hand imagery - so a button cannot be a
            class. The anchor button makes the instruction concrete without
            pretending it is a separate class.
        :param capture: SessionCapture, so each cue can record the sample index
            it began at and the stream can be epoched exactly afterwards.
        :param anchors: (left_button, right_button). Defaults to the leftmost
            and rightmost on screen, which keeps the instruction spatially
            congruent with the hand being imagined.
        """
        self.mi = mi
        self.capture = capture
        self.anchors = anchors
        self.hand_order = []
        self.bci = bci                    # None in --dry-run
        self.n_trials = trials
        self.cue_ms = int(cue_s * 1000)
        self.rest_ms = int(rest_s * 1000)
        self.script = script

        self.buttons = []
        self.order = []
        self.trial = 0
        self.phase = self.REST
        self.phase_end = 0
        self.target = None
        self.started = False
        self.done_at = None

        self.clicked = None               # last button clicked, for the overlay
        self.clicked_at = 0
        self.trial_click = None           # click recorded for the current trial
        self.log = []

        self.font_big = None
        self.font = None
        self.font_sm = None

    # ---------------------------------------------------------- discovery
    def find_buttons(self, namespace):
        """Pull Button instances out of the running script's globals."""
        from Button import Button
        found, seen = [], set()
        for value in namespace.values():
            if isinstance(value, Button) and id(value) not in seen:
                seen.add(id(value))
                found.append(value)
        # stable left-to-right order so trial labels mean something
        found.sort(key=lambda b: (b.rect.centerx, b.rect.centery))
        return found

    def _wrap_clicks(self):
        """Record clicks without touching the example script.

        The script calls button.click(); we wrap that method so the click is
        logged and the icon overlay is triggered, then the original runs as
        normal (including any func= the developer supplied).
        """
        for btn in self.buttons:
            if getattr(btn, "_dr_wrapped", False):
                continue
            original = btn.click

            def make(b, orig):
                def wrapped(*a, **k):
                    self.clicked = b
                    self.clicked_at = pygame.time.get_ticks()
                    if self.phase == self.CUE and self.trial_click is None:
                        self.trial_click = b
                    return orig(*a, **k)
                return wrapped

            btn.click = make(btn, original)
            btn._dr_wrapped = True

    # ------------------------------------------------------------ markers
    def _marker(self, name, phase):
        if self.bci is None:
            return
        try:
            self.bci.put_marker(name, phase=phase)
        except Exception as exc:
            print(f"  [marker failed: {exc}]")

    # ------------------------------------------------------- trial engine
    def _pick_anchors(self):
        """Leftmost and rightmost button, by screen position."""
        if self.anchors:
            return self.anchors
        ordered = sorted(self.buttons,
                         key=lambda b: getattr(getattr(b, "rect", None),
                                               "centerx", 0))
        return ordered[0], ordered[-1]

    def _build_order(self):
        """Balanced, shuffled target sequence - every button appears equally."""
        if self.mi:
            self.hand_order = build_hand_order(self.n_trials)
            left, right = self._pick_anchors()
            return [(left if h == 0 else right) for h in self.hand_order]
        order = []
        while len(order) < self.n_trials:
            batch = list(self.buttons)
            random.shuffle(batch)
            order.extend(batch)
        return order[:self.n_trials]

    def _start_cue(self, now):
        self.trial += 1
        self.target = self.order[self.trial - 1]
        self.trial_click = None
        self.phase = self.CUE
        self.phase_end = now + self.cue_ms
        self.cue_start = now

        for b in self.buttons:
            b.stop_flash()
        # loop=True so the ripple runs for the whole cue, however long it is
        self.target.flash(loop=True, waves=2, pulse_hz=0.5)

        self.cue_sample = (self.capture.n_samples
                           if self.capture is not None else None)
        if self.mi:
            hand, _lab = HAND_LABELS[self.hand_order[self.trial - 1]]
            self.cue_hand = self.hand_order[self.trial - 1]
            self._marker(f"hand_{hand}", "start")
            print(f"  trial {self.trial}/{self.n_trials}: imagine PRESSING "
                  f"'{self.target.name}' with your {hand} hand")
        else:
            self.cue_hand = None
            self._marker(self.target.name, "start")
            print(f"  trial {self.trial}/{self.n_trials}: "
                  f"LOOK AT '{self.target.name}'")

    def _end_cue(self, now):
        self._marker(self.target.name, "end")
        self.target.stop_flash()

        hit = self.trial_click
        ok = hit is self.target
        verdict = "correct" if ok else (f"clicked {hit.name}" if hit else "no click")
        print(f"           -> {verdict}")

        row = {
            "trial": self.trial,
            "target": self.target.name,
            "target_id": self.target.id,
            "cue_start_ms": self.cue_start,
            "cue_end_ms": now,
            "clicked": hit.name if hit else "",
            "correct": int(ok),
        }
        if self.mi:
            hand, label = HAND_LABELS[self.cue_hand]
            row["hand"] = hand
            row["label"] = label
            row["cue_sample"] = self.cue_sample
        self.log.append(row)

        self.phase = self.REST
        self.phase_end = now + self.rest_ms

    def update(self, now):
        if self.phase == self.DONE:
            if self.mi and now - getattr(self, "done_at", now) > 2000:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            return

        if not self.started:
            self.started = True
            self.order = self._build_order()
            self.phase = self.REST
            self.phase_end = now + self.rest_ms
            return

        if self.phase == self.CUE:
            # self-heal: the example script may restart its own non-looping
            # flash on click, which would end our cue ripple early
            if not self.target.is_flashing:
                self.target.flash(loop=True, waves=2, pulse_hz=0.5)
            if now >= self.phase_end:
                self._end_cue(now)

        elif self.phase == self.REST and now >= self.phase_end:
            if self.trial >= self.n_trials:
                self.phase = self.DONE
                self.done_at = now
                if self.mi:
                    # Calibration ends itself. Leaving the window open after
                    # the last cue means the saved file depends on when
                    # somebody happens to click the X, and a session that is
                    # never closed is a session never written to disk.
                    print("\n  all trials complete - saving in 2s")
                else:
                    print("\n  all trials complete - close the window "
                          "to finish")
            else:
                self._start_cue(now)

    # --------------------------------------------------------- rendering
    def _fonts(self):
        if self.font is None:
            self.font_big = pygame.font.SysFont("dejavusans,arial", 30, bold=True)
            self.font = pygame.font.SysFont("dejavusans,arial", 19, bold=True)
            self.font_sm = pygame.font.SysFont("dejavusans,arial", 14)

    def draw(self, win):
        """Overlay drawn on top of whatever the example rendered this frame."""
        self._fonts()
        w, h = win.get_size()

        # --- clicked icon at screen centre -----------------------------
        # Shown for 1.2 s after any click, so the user gets confirmation that
        # the selection registered.
        if self.clicked is not None:
            age = pygame.time.get_ticks() - self.clicked_at
            if age < 1200:
                side = int(min(w, h) * 0.22)
                icon = pygame.transform.smoothscale(self.clicked.img, (side, side))
                if age > 900:                       # fade the last 300 ms
                    icon = icon.copy()
                    icon.set_alpha(int(255 * (1200 - age) / 300))
                win.blit(icon, icon.get_rect(center=(w // 2, h // 2 - 30)))
                cap = self.font.render(self.clicked.name.upper(), True, (250, 214, 110))
                win.blit(cap, cap.get_rect(center=(w // 2, h // 2 + side // 2 - 10)))
            else:
                self.clicked = None

        # --- status bar -------------------------------------------------
        bar = pygame.Surface((w, 58), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 170))
        win.blit(bar, (0, 0))

        if self.phase == self.CUE:
            msg, col = f"LOOK AT:  {self.target.name.upper()}", (250, 214, 110)
        elif self.phase == self.DONE:
            msg, col = "COMPLETE - close the window", (120, 220, 160)
        else:
            msg, col = "rest...", (170, 178, 198)
        win.blit(self.font_big.render(msg, True, col), (18, 10))

        right = f"trial {min(self.trial, self.n_trials)}/{self.n_trials}"
        r = self.font.render(right, True, (220, 226, 240))
        win.blit(r, r.get_rect(topright=(w - 18, 16)))

        if self.phase in (self.CUE, self.REST) and self.started:
            left = max(0, (self.phase_end - pygame.time.get_ticks()) / 1000.0)
            t = self.font_sm.render(f"{left:4.1f}s", True, (150, 158, 180))
            win.blit(t, t.get_rect(topright=(w - 18, 38)))

        if self.mi and self.phase == self.CUE and self.cue_hand is not None:
            hand, _ = HAND_LABELS[self.cue_hand]
            self.hint = (f"imagine PRESSING {self.target.name} "
                         f"with your {hand} hand")
        rec = "REC" if self.bci is not None else "DRY-RUN (no EEG)"
        reccol = (220, 90, 90) if self.bci is not None else (140, 146, 164)
        win.blit(self.font_sm.render(rec, True, reccol), (18, 40))

        # --- progress bar ----------------------------------------------
        if self.started and self.phase != self.DONE:
            total = self.cue_ms if self.phase == self.CUE else self.rest_ms
            left_ms = max(0, self.phase_end - pygame.time.get_ticks())
            frac = 1.0 - (left_ms / total if total else 1)
            pygame.draw.rect(win, (40, 44, 58), pygame.Rect(0, 56, w, 4))
            pygame.draw.rect(win, col, pygame.Rect(0, 56, int(w * frac), 4))

    # ------------------------------------------------------------- output
    def save_log(self, out_dir=None):
        if not self.log:
            return None
        out_dir = Path(out_dir or (ROOT / "dataset"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"cue_log_{self.script}_{stamp}.csv"
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(self.log[0].keys()))
            wr.writeheader()
            wr.writerows(self.log)
        return path


def run_example(args):
    """Execute example/<name>.py with the cue recorder attached."""
    name = args.file
    if not name.endswith(".py"):
        name += ".py"
    script = EXAMPLE_DIR / name
    if not script.exists():
        print(f"No such script: {script}")
        print(f"Available: {', '.join(sorted(p.stem for p in EXAMPLE_DIR.glob('*.py') if p.stem != '__init__'))}")
        return 1

    BCI = capture = None
    if args.mi:
        # Calibration mode owns the stream itself, through StreamBridge's
        # BrainFlowBackend, because it must keep every sample to epoch
        # afterwards. The LSL path used for marker-only recording hands
        # samples onward and keeps nothing.
        if not args.subject:
            print("--mi needs --subject")
            return 1
        if args.dry_run:
            from stream_bridge import SyntheticBackend
            backend = SyntheticBackend(fs=250, n_channels=8)
            print("DRY-RUN: synthetic backend, no hardware touched")
        else:
            from stream_bridge import BrainFlowBackend
            # serial_port must be a string: BrainFlow rejects None with a
            # bare GENERAL_ERROR:17 that says nothing about the cause. The
            # synthetic board (-1) needs no port at all.
            backend = BrainFlowBackend(board_id=args.board_id,
                                       serial_port=args.serial_port or "",
                                       n_channels=8)
            print(f"Recording started (board_id={args.board_id})")
        capture = SessionCapture(backend).start()
    elif not args.dry_run:
        BCI = _load_bci()
        BCI.start(BID=args.board_id, port=args.serial_port, plot_domain="f")
        print(f"Recording started (board_id={args.board_id})")

    rec = CuedRecorder(BCI, trials=args.trials, cue_s=args.cue,
                       rest_s=args.rest, script=script.stem,
                       mi=args.mi, capture=capture)
    if args.mi and args.cue < 3.0:
        print(f"  WARNING: cue {args.cue}s is shorter than the 3.0s epoch "
              f"window, so most of every epoch would be rest. Use --cue 4 "
              f"or more.")

    print(f"\nRunning {script.name}")
    print(f"  {args.trials} trials · {args.cue}s cue · {args.rest}s rest")
    print("  the flashing button is your target - try to select it\n")

    namespace = {"__name__": "__main__", "__file__": str(script)}
    original_flip = pygame.display.flip

    def hooked_flip(*a, **k):
        """Runs once per frame, after the script has drawn, before presenting."""
        win = pygame.display.get_surface()
        if win is not None:
            if not rec.buttons:
                rec.buttons = rec.find_buttons(namespace)
                if rec.buttons:
                    rec._wrap_clicks()
                    print(f"  found {len(rec.buttons)} buttons: "
                          f"{', '.join(b.name for b in rec.buttons)}\n")
            if rec.buttons:
                rec.update(pygame.time.get_ticks())
                rec.draw(win)
        return original_flip(*a, **k)

    """
    Added by Anson
    Date: 20/9/2026
    Purpose: run from whichever directory actually contains assets/.

    The examples load images with the relative path "assets/", so the correct
    working directory depends on where assets/ lives - NOT on where the script
    lives. Running `py example/rockpaperscissors.py` from the project root means
    assets/ is expected at the root, so chdir-ing into example/ broke it.

    Look for assets/ in the likely places and use the first hit. Falling back to
    the current directory keeps the behaviour identical to launching the script
    by hand.
    """
    import os
    prev_cwd = os.getcwd()
    for candidate in (Path(prev_cwd), ROOT, EXAMPLE_DIR):
        if (candidate / "assets").is_dir():
            run_dir = candidate
            break
    else:
        run_dir = Path(prev_cwd)
    if run_dir != Path(prev_cwd):
        print(f"  assets found in {run_dir}")
    os.chdir(run_dir)

    failed = None
    try:
        pygame.display.flip = hooked_flip
        exec(compile(script.read_text(), str(script), "exec"), namespace)
    except SystemExit:
        pass
    except FileNotFoundError as exc:
        # examples load assets with relative paths; a missing asset is the
        # script's problem, not the recorder's - say so plainly
        searched = ", ".join(str(c) for c in (Path(prev_cwd), ROOT, EXAMPLE_DIR))
        failed = (f"{script.name} could not find a file it needs:\n"
                  f"    {exc}\n"
                  f"  Looked for an assets/ folder in: {searched}\n"
                  f"  Run data_record.py from the same folder you would run\n"
                  f"  the example from, or move assets/ next to the script.")
    except Exception as exc:
        failed = f"{script.name} raised {type(exc).__name__}: {exc}"
    finally:
        pygame.display.flip = original_flip
        os.chdir(prev_cwd)
        if failed:
            print(f"\n  SCRIPT ERROR\n  {failed}")
        path = rec.save_log()
        if path:
            hits = sum(r["correct"] for r in rec.log)
            print(f"\n  {hits}/{len(rec.log)} trials matched the cue")
            print(f"  trial log -> {path}")
        if BCI is not None:
            BCI.end()
            print("  EEG recording stopped")
        if capture is not None:
            stream = capture.stop()
            print(f"  captured {stream.shape[1] / capture.fs:.1f}s "
                  f"({stream.shape[1]} samples)")
            X, y, labels = epoch_from_log(stream, capture.fs, rec.log,
                                          t0_ms=0)
            if len(X) == 0:
                print("  no usable epochs - was the cue shorter than 3 s?")
            else:
                out = Path(args.out) if args.out else (
                    ROOT / "profiles" / args.subject.lower())
                npz = save_session(X, y, labels, args.subject, capture.fs,
                                   args.montage, out)
                n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
                print(f"  {len(X)} epochs  (LEFT {n0} / RIGHT {n1})")
                print(f"  calibration -> {npz}")
                print(f"  next:  python ML/train_local.py {args.subject}")
        try:
            pygame.quit()
        except Exception:
            pass
    return 0


# =====================================================================
#  CHARACTER MODE (original behaviour, unchanged)
# =====================================================================
def main():
    import matplotlib.pyplot as plt
    BCI = _load_bci()

    #init fields
    ##game
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    win = pygame.display.set_mode((800,600))
    width, height = win.get_size()

    size = int(width*0.1)
    center = (width/2, height/2)
    my_font = pygame.font.SysFont('Comic Sans MS', size)

    show = ""
    count = 0
    show_text = False
    allow_input = True
    buffer = False
    buffer_time = 1
    show_time = 2
    cycle_start = 0
    buffer_start = 0
    ##random character list
    ###length = 0 to manual input
    length = 0
    arr = random.choices(string.ascii_letters + string.digits, k=length)

    ##game loop
    run = True
    win.fill((0,0,0))

    BCI.start(BID=-2,port=None, plot_domain="f")
    #start live plot in non blocking mode
    ani = BCI.plot_data(block=False)

    win.fill((0,0,0))
    #for logging, start_time set only when end
    #phase = start/end
    #type = buffer/cycle
    def log(phase, type, current_time, count, start_time=-1):
        if phase == "start":
            print(f"Starting {count} {type} at {current_time//1000}s")
        elif phase == "end" and start_time != -1:
            print(f"Finished {count} {type} at {current_time//1000}s in {(current_time-start_time)//1000}s")

    """
    Edit by Anson
    Date: 18/7/2026
    Changes: wrap entire game flow in try and finally, only set run=False inside
    """
    try:
        if arr:
            print(arr)
            allow_input = False
            # send start marker of show
            show = arr.pop(0)
            BCI.put_marker(show, phase="start")
            count += 1
            show_text = True

            cycle_start = pygame.time.get_ticks()
            log("start", "cycle", cycle_start, count)
            display_time = cycle_start + (show_time * 1000)

        while run:
            current_time = pygame.time.get_ticks()
            #send gui events so chart update without freezing
            plt.pause(0.001)
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    run = False
                elif e.type == KEYDOWN and not arr:
                    #if key pressed is not enter, key is alphanumeric, input is allowed and arr is empty
                    if e.key != K_RETURN and e.unicode.isalnum() and allow_input:
                        show += e.unicode
                        print(f"\rKeyboard: {show: <20}", end='', flush=True)

                    elif e.key == K_BACKSPACE and allow_input:
                        show = show[:-1]
                        print(f"\rKeyboard: {show: <20}", end='', flush=True)

                    elif e.key in (K_RETURN, K_KP_ENTER) and show and allow_input:
                        print()
                        #send start marker of show
                        BCI.put_marker(show, phase="start")
                        allow_input = False
                        count += 1
                        show_text = True

                        cycle_start = current_time
                        log("start", "cycle", current_time=current_time, count=count)
                        display_time = current_time + show_time*1000

            if length != 0 and not arr and not show_text and not buffer:
                run = False

            #showing text but passed display_time(2s)            
            if show_text and current_time >= display_time:
                #send end marker of show
                BCI.put_marker(show, phase="end")
                print(show)
                log("end", "cycle",current_time=current_time, count=count, start_time=cycle_start)
                show_text = False
                buffer = True
                buffer_start = current_time
                display_time = current_time + buffer_time*1000

            elif buffer and current_time >= display_time:
                buffer = False
                allow_input = True
                show = ""
                log("end", "buffer", current_time=current_time, count=count, start_time=buffer_start)
                print("Allow input")
                if arr:
                    show = arr.pop(0)
                    BCI.put_marker(show, phase="start")
                    count += 1
                    show_text = True
                    cycle_start = current_time
                    log("start", "cycle", current_time, count)
                    display_time = current_time + (show_time * 1000)

            if show_text:
                text_surface = my_font.render(show, False, (255, 255, 255))
                textRect = text_surface.get_rect(center=center)
                win.blit(text_surface, textRect)
            else:
                win.fill((0, 0, 0))

            #update full display surface to screen
            pygame.display.flip()
            clock.tick(60)
    finally:
        BCI.end()
        pygame.quit()


def cli():
    ap = argparse.ArgumentParser(
        description="Record EEG while running an example app with cued targets")
    ap.add_argument("--file", default=None,
                    help="script in example/ to run, e.g. rockpaperscissors")
    ap.add_argument("--trials", type=int, default=10, help="number of cued trials")
    ap.add_argument("--cue", type=float, default=5.0, help="seconds per cue")
    ap.add_argument("--rest", type=float, default=2.0, help="seconds between cues")
    ap.add_argument("--board-id", type=int, default=-2,
                    help="-2 streaming board (OpenBCI GUI) · 0 Cyton · -1 synthetic")
    ap.add_argument("--serial-port", default=None, help="e.g. COM4")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the UI and cues without recording EEG")
    ap.add_argument("--list", action="store_true", help="list example scripts")
    ap.add_argument("--mi", action="store_true",
                    help="CALIBRATION mode: cue a hand per trial and write a "
                         "calib_*.npz that ML/train_local.py can train on")
    ap.add_argument("--subject", default=None,
                    help="subject name for --mi (required)")
    ap.add_argument("--montage", default="cyton8_motor")
    ap.add_argument("--out", default=None,
                    help="output dir for --mi (default profiles/<subject>)")
    args = ap.parse_args()

    if args.list:
        print("Scripts in example/:")
        for p in sorted(EXAMPLE_DIR.glob("*.py")):
            if p.stem != "__init__":
                print("  ", p.stem)
        return 0

    if args.file:
        return run_example(args)
    main()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
