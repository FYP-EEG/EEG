"""
Author: Anson Li
Created on: 26/6/2026
Purpose: script for easier data recording
Location: project_dir/EEG/data_record.py

Edited on: 20/9/2026
Changes: added --file mode. Runs a script from example/, cues the user with a
         flashing button, and records EEG for the whole session.

Edited by Anson
Date: 13/10/2026
Purpose: add an isolated, experimental --button-mi mode. It labels the actual
         app buttons directly and never changes the existing --mi hand mode.

Edited by Anson
Date: 13/10/2026
Purpose: prevent intermittent REST stalls by using a monotonic phase clock and
         advancing from event.get(), display.flip(), or display.update().

Edited by Anson
Date: 19/10/2026
Purpose: add reusable --intent-mi labels, task-set profiles, developer action
         mappings, optional idle trials and generic non-button action targets.

MODES
-----
1. Character mode (original, unchanged) - flashes typed/random characters:
       py EEG/data_record.py

2. Cued app mode - runs a real app and cues button targets:
       py EEG/data_record.py --file rockpaperscissors
       py EEG/data_record.py --file rockpaperscissors --trials 20 --cue 5
       py EEG/data_record.py --file realistic_ui --dry-run     (no EEG needed)
       py EEG/data_record.py --list

3. Existing LEFT/RIGHT MI calibration (unchanged):
       py EEG/data_record.py --file rockpaperscissors --mi --subject anson

4. Legacy direct-button MI calibration (separate files/model):
       py EEG/data_record.py --file rockpaperscissors --button-mi \
            --button-hand right --subject anson --trials 90 --cue 4

5. Recommended reusable motor-intent calibration:
       py EEG/data_record.py --file rockpaperscissors --intent-mi \
            --subject anson --trials 90 --idle-trials 15 --cue 4

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

Legacy examples need no changes because pygame hooks remain available. The main
demos expose an explicit __bci_recorder_frame__ callback for more reliable timing.
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
for _path in (ROOT, ROOT / "pygame_lib"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


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
# THREE ISOLATED MI LABEL CONTRACTS
# ---------------------------------
# --mi is the established path. Its classes remain target_0/target_1 =
# LEFT/RIGHT hand, and buttons are reached through the existing selector.
# Nothing in the experimental mode below changes that contract.
#
# --button-mi is an opt-in experiment. It stores button::NNN::<name> and asks
# the participant to imagine the complete spatial reach-and-press action for
# that exact on-screen button. This does NOT imply that EEG reads the semantic
# idea of "rock" or "paper". If all imagined movements are physically identical,
# the classes will probably remain near chance. Different screen positions give
# the imagined movements the only plausible motor distinction.
#
# Direct-button recordings are isolated under
# profiles/<user>/button_mi/<app>/ and are trained by ML/train_button_mi.py.
#
# --intent-mi is the recommended scalable contract. It stores physical labels
# such as intent::left_index or intent::both_feet independently of Rock, jump,
# inventory, or any other app action. Recordings/models are keyed by user and
# task set under profiles/<user>/motor_intent/, so developers can remap a
# trained task set to new callbacks without retraining.
#
# ML/train_local.py never sees either experimental data directory.


_BUTTON_LABEL_FN = None
_MOTOR_INTENT_API = None


def _load_motor_intent_api():
    """Preload reusable intent helpers before any timed cue begins."""
    global _MOTOR_INTENT_API
    if _MOTOR_INTENT_API is None:
        from ML import motor_intent
        _MOTOR_INTENT_API = motor_intent
    return _MOTOR_INTENT_API


def _load_button_label_fn():
    """Load the direct-label helper outside any timed cue.

    Importing ML.button_mi for the first time took ~0.9 s on Linux and nearly
    4 s on one Windows machine. It used to happen inside _start_cue(), after the
    CUE clock had already started, so trial 1 lost most of its visible time.
    """
    global _BUTTON_LABEL_FN
    if _BUTTON_LABEL_FN is None:
        from ML.button_mi import make_button_label
        _BUTTON_LABEL_FN = make_button_label
    return _BUTTON_LABEL_FN


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


def epoch_button_mi_from_log(stream, fs, log, t0_ms, window_s=3.0):
    """Cut direct-button trials without accepting LEFT/RIGHT labels.

    Kept separate from epoch_from_log() so changing the experimental label set
    cannot alter the established two-class MI calibration path.
    """
    from ML.button_mi import BUTTON_LABEL_PREFIX

    stream = np.asarray(stream, dtype=np.float64)
    n = int(window_s * fs)
    X, y, labels = [], [], []
    for row in log:
        label = str(row.get("label", ""))
        if not label.startswith(BUTTON_LABEL_PREFIX):
            continue
        if row.get("cue_sample") is not None:
            start = int(row["cue_sample"])
        else:
            start = int((row["cue_start_ms"] - t0_ms) / 1000.0 * fs)
        if start < 0 or start + n > stream.shape[1]:
            continue
        X.append(stream[:, start:start + n])
        y.append(int(row["button_index"]))
        labels.append(label)
    return np.asarray(X), np.asarray(y), np.asarray(labels)


def epoch_intent_mi_from_log(stream, fs, log, t0_ms, window_s=3.0):
    """Cut reusable motor-intent trials without accepting other MI labels."""
    api = _load_motor_intent_api()
    stream = np.asarray(stream, dtype=np.float64)
    n = int(window_s * fs)
    X, y, labels = [], [], []
    for row in log:
        label = str(row.get("label", ""))
        if not label.startswith(api.INTENT_LABEL_PREFIX):
            continue
        api.parse_intent_label(label)
        if row.get("cue_sample") is not None:
            start = int(row["cue_sample"])
        else:
            start = int((row["cue_start_ms"] - t0_ms) / 1000.0 * fs)
        if start < 0 or start + n > stream.shape[1]:
            continue
        X.append(stream[:, start:start + n])
        y.append(int(row["class_index"]))
        labels.append(label)
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
        """Atomic snapshot used to mark a cue without blocking the UI thread.

        ``_n`` is a Python integer and assignment is atomic under the GIL. The
        old lock-taking getter could be starved by the capture thread repeatedly
        reacquiring the lock; on Windows this blocked the first cue for 3-4
        seconds while EEG capture continued. A lock-free snapshot is the right
        boundary semantics here: a concurrently arriving chunk belongs either
        just before or just after the cue, and the integer is never torn.
        """
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
                # Always yield after a successful read. BrainFlow buffers data,
                # so this loses no samples; it prevents the acquisition thread
                # from immediately reacquiring _lock and starving the UI.
                time.sleep(0.001)
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
                 mi=False, button_mi=False, intent_mi=False, intent_map=None,
                 idle_trials=0, button_hand="RIGHT", capture=None,
                 anchors=None, clock=None):
        """
        :param mi: established calibration mode; labels LEFT/RIGHT hand only.
        :param button_mi: legacy isolated experiment with button-name labels.
        :param intent_mi: recommended reusable mode; labels physical intentions
            independently of application action names.
        :param intent_map: action-name -> motor-intent mapping. It can be set
            after the app exposes BCI_INTENT_MAP.
        :param idle_trials: explicit no-movement trials. Required when only one
            motor intention is trained and recommended for false-positive data.
        :param button_hand: fixed hand for the legacy direct-button experiment.
        :param capture: SessionCapture, so each cue can record the sample index
            it began at and the stream can be epoched exactly afterwards.
        :param anchors: (left_button, right_button) for the existing --mi mode.
        :param clock: monotonic milliseconds provider, injectable for tests.
            Phase timing deliberately does not use pygame.time.get_ticks(): that
            counter can restart when an app reinitialises pygame/display, which
            previously left REST waiting for an obsolete deadline.
        """
        if sum(bool(v) for v in (mi, button_mi, intent_mi)) > 1:
            raise ValueError("--mi, --button-mi and --intent-mi are separate modes")
        self.mi = bool(mi)
        self.button_mi = bool(button_mi)
        self.intent_mi = bool(intent_mi)
        self.capture_mode = self.mi or self.button_mi or self.intent_mi
        self.button_hand = str(button_hand).upper()
        self.capture = capture
        self.anchors = anchors
        self.hand_order = []
        self.action_trials = int(trials)
        self.idle_trials = int(idle_trials) if self.intent_mi else 0
        self.intent_map = None
        self.intent_values = ()
        self.intent_class = {}
        if intent_map is not None:
            self.set_intent_map(intent_map)
        self._clock = clock or (lambda: time.monotonic() * 1000.0)
        self.bci = bci                    # None in --dry-run
        self.n_trials = self.action_trials + self.idle_trials
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

    def set_intent_map(self, mapping):
        api = _load_motor_intent_api()
        parsed = api.parse_intent_map(mapping)
        self.intent_map = parsed
        self.intent_values = api.canonical_intents(parsed.values())
        classes = list(self.intent_values)
        if self.idle_trials:
            classes.append(api.IDLE)
        self.intent_class = {name: i for i, name in enumerate(classes)}
        assessment = api.assess_intent_set(self.intent_values)
        print(f"  motor-intent task set: {', '.join(self.intent_values)}")
        print(f"  difficulty: {assessment['difficulty']} · "
              f"profile {assessment['task_set']}")
        for warning in assessment["warnings"]:
            print(f"  WARNING: {warning}")
        if len(self.intent_values) == 1 and not self.idle_trials:
            raise ValueError(
                "one-intent recording requires --idle-trials 10 or more")
        return parsed

    # ---------------------------------------------------------- discovery
    def find_buttons(self, namespace):
        """Find pygame Buttons or explicit BCI_RECORD_TARGETS action tiles."""
        provided = namespace.get("BCI_RECORD_TARGETS")
        if provided:
            found = list(provided)
            required = ("name", "id", "rect", "flash", "stop_flash")
            for target in found:
                missing = [a for a in required if not hasattr(target, a)]
                if missing:
                    raise ValueError(
                        f"BCI_RECORD_TARGETS item {target!r} lacks {missing}")
        else:
            from Button import Button
            found, seen = [], set()
            for value in namespace.values():
                if isinstance(value, Button) and id(value) not in seen:
                    seen.add(id(value))
                    found.append(value)
        # Stable screen order for logging and balanced cue construction.
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

    def _button_label(self, button):
        """Direct-mode label in stable left-to-right screen order."""
        make_button_label = _load_button_label_fn()
        return make_button_label(self.buttons.index(button), button.name)

    def _build_order(self):
        """Balanced, shuffled target sequence - every button appears equally."""
        if self.mi:
            self.hand_order = build_hand_order(self.n_trials)
            left, right = self._pick_anchors()
            return [(left if h == 0 else right) for h in self.hand_order]
        if self.button_mi:
            names = [str(b.name) for b in self.buttons]
            if len(names) != len(set(names)):
                raise ValueError("--button-mi requires unique Button.name values")
        if self.intent_mi:
            if not self.intent_map:
                raise ValueError(
                    "--intent-mi needs BCI_INTENT_MAP in the example or "
                    "--intent-map action=intent,...")
            by_name = {str(b.name): b for b in self.buttons}
            missing = set(self.intent_map) - set(by_name)
            if missing:
                raise ValueError(
                    f"intent map references missing actions: {sorted(missing)}")
            action_targets = [by_name[name] for name in self.intent_map]
            order = []
            while len(order) < self.action_trials:
                batch = list(action_targets)
                random.shuffle(batch)
                order.extend(batch)
            order = order[:self.action_trials] + [None] * self.idle_trials
            random.shuffle(order)
            return order
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
        self.phase_end = self._clock() + self.cue_ms
        self.cue_start = now

        for b in self.buttons:
            b.stop_flash()
        # None is an explicit idle/no-movement class in reusable intent mode.
        if self.target is not None:
            self.target.flash(loop=True, waves=2, pulse_hz=0.5)

        self.cue_sample = (self.capture.n_samples
                           if self.capture is not None else None)
        if self.mi:
            hand, _lab = HAND_LABELS[self.hand_order[self.trial - 1]]
            self.cue_hand = self.hand_order[self.trial - 1]
            self._marker(f"hand_{hand}", "start")
            print(f"  trial {self.trial}/{self.n_trials}: imagine PRESSING "
                  f"'{self.target.name}' with your {hand} hand")
        elif self.button_mi:
            self.cue_hand = None
            self._marker(self._button_label(self.target), "start")
            print(f"  trial {self.trial}/{self.n_trials}: imagine REACHING to "
                  f"and PRESSING '{self.target.name}' with your "
                  f"{self.button_hand} hand - do not move")
        elif self.intent_mi:
            api = _load_motor_intent_api()
            self.cue_hand = None
            self.cue_intent = (api.IDLE if self.target is None
                               else self.intent_map[self.target.name])
            label = api.make_intent_label(self.cue_intent)
            self._marker(label, "start")
            action = None if self.target is None else self.target.name
            instruction = api.intent_instruction(self.cue_intent, action)
            print(f"  trial {self.trial}/{self.n_trials}: {instruction}")
        else:
            self.cue_hand = None
            self._marker(self.target.name, "start")
            print(f"  trial {self.trial}/{self.n_trials}: "
                  f"LOOK AT '{self.target.name}'")

    def _end_cue(self, now):
        if self.intent_mi:
            api = _load_motor_intent_api()
            marker_name = api.make_intent_label(self.cue_intent)
        elif self.button_mi:
            marker_name = self._button_label(self.target)
        else:
            marker_name = self.target.name
        self._marker(marker_name, "end")
        if self.target is not None:
            self.target.stop_flash()

        hit = self.trial_click
        ok = hit is self.target
        if self.button_mi or self.intent_mi:
            # Mental compliance is not observable. A real click/movement would
            # contaminate the imagery epoch with muscle activity.
            verdict = "imagery recorded (no physical movement required)"
        else:
            verdict = "correct" if ok else (f"clicked {hit.name}" if hit else "no click")
        print(f"           -> {verdict}")

        row = {
            "trial": self.trial,
            "target": self.target.name if self.target is not None else "idle",
            "target_id": self.target.id if self.target is not None else -1,
            "cue_start_ms": self.cue_start,
            "cue_end_ms": now,
            "clicked": hit.name if hit else "",
            "correct": "" if (self.button_mi or self.intent_mi) else int(ok),
        }
        if self.mi:
            hand, label = HAND_LABELS[self.cue_hand]
            row["hand"] = hand
            row["label"] = label
            row["cue_sample"] = self.cue_sample
        elif self.button_mi:
            row["button_hand"] = self.button_hand
            row["button_index"] = self.buttons.index(self.target)
            row["label"] = self._button_label(self.target)
            row["cue_sample"] = self.cue_sample
            row["compliance"] = "not_observable"
        elif self.intent_mi:
            row["intent"] = self.cue_intent
            row["class_index"] = self.intent_class[self.cue_intent]
            row["label"] = api.make_intent_label(self.cue_intent)
            row["cue_sample"] = self.cue_sample
            row["compliance"] = "not_observable"
        self.log.append(row)

        self.phase = self.REST
        self.phase_end = self._clock() + self.rest_ms
        print(f"           rest {self.rest_ms / 1000.0:.1f}s")

    def update(self, now=None):
        """Advance the trial engine using a monotonic clock.

        ``now`` remains the pygame tick used only in the CSV log. Deadlines use
        time.monotonic(), so a display reinitialisation or pygame timer reset
        cannot strand the recorder in REST.
        """
        if now is None:
            now = pygame.time.get_ticks()
        clock_now = self._clock()

        if self.phase == self.DONE:
            if (self.capture_mode and
                    clock_now - getattr(self, "done_at", clock_now) > 2000):
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            return

        if not self.started:
            self.started = True
            self.order = self._build_order()
            self.phase = self.REST
            self.phase_end = clock_now + self.rest_ms
            print(f"  initial rest {self.rest_ms / 1000.0:.1f}s")
            return

        if self.phase == self.CUE:
            # self-heal: the example script may restart its own non-looping
            # flash on click, which would end our cue ripple early
            if self.target is not None and not self.target.is_flashing:
                self.target.flash(loop=True, waves=2, pulse_hz=0.5)
            if clock_now >= self.phase_end:
                self._end_cue(now)

        elif self.phase == self.REST and clock_now >= self.phase_end:
            if self.trial >= self.n_trials:
                self.phase = self.DONE
                self.done_at = clock_now
                if self.capture_mode:
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
            if self.intent_mi:
                api = _load_motor_intent_api()
                if self.cue_intent == api.IDLE:
                    msg = "REST / NO MOVEMENT"
                else:
                    msg = f"IMAGINE:  {api.intent_display_name(self.cue_intent)}"
            elif self.button_mi:
                msg = f"IMAGINE PRESSING:  {self.target.name.upper()}"
            else:
                msg = f"LOOK AT:  {self.target.name.upper()}"
            col = (250, 214, 110)
        elif self.phase == self.DONE:
            msg, col = "COMPLETE - saving", (120, 220, 160)
        else:
            msg, col = "rest...", (170, 178, 198)
        win.blit(self.font_big.render(msg, True, col), (18, 10))

        # Never display "trial 0". Before the first cue this is preparation,
        # not a trial; between cues report what completed and what comes next.
        # Calling the preparation interval trial 0 made it look as though it
        # overlapped trial 1 even though no EEG epoch is cut from that interval.
        if self.phase == self.CUE:
            right = f"trial {self.trial}/{self.n_trials}"
        elif self.phase == self.DONE:
            right = f"complete {self.n_trials}/{self.n_trials}"
        elif self.trial == 0:
            right = f"get ready · next 1/{self.n_trials}"
        elif self.trial < self.n_trials:
            right = (f"rest · done {self.trial}/{self.n_trials} · "
                     f"next {self.trial + 1}/{self.n_trials}")
        else:
            right = f"rest · done {self.n_trials}/{self.n_trials}"
        r = self.font_sm.render(right, True, (220, 226, 240))
        win.blit(r, r.get_rect(topright=(w - 18, 18)))

        if self.phase in (self.CUE, self.REST) and self.started:
            left = max(0, (self.phase_end - self._clock()) / 1000.0)
            t = self.font_sm.render(f"{left:4.1f}s", True, (150, 158, 180))
            win.blit(t, t.get_rect(topright=(w - 18, 38)))

        if self.mi and self.phase == self.CUE and self.cue_hand is not None:
            hand, _ = HAND_LABELS[self.cue_hand]
            self.hint = (f"imagine PRESSING {self.target.name} "
                         f"with your {hand} hand")
        elif self.button_mi and self.phase == self.CUE:
            hint = (f"feel your {self.button_hand} hand reach to its screen "
                    f"position and press - keep the real hand still")
            hs = self.font_sm.render(hint, True, (220, 226, 240))
            win.blit(hs, hs.get_rect(center=(w // 2, 47)))
        elif self.intent_mi and self.phase == self.CUE:
            api = _load_motor_intent_api()
            action = None if self.target is None else self.target.name
            hint = api.intent_instruction(self.cue_intent, action)
            # Keep the overlay compact on 800 px demo windows.
            if len(hint) > 92:
                hint = hint[:89] + "..."
            hs = self.font_sm.render(hint, True, (220, 226, 240))
            win.blit(hs, hs.get_rect(center=(w // 2, 47)))
        rec = "REC" if self.capture is not None or self.bci is not None else "DRY-RUN (no EEG)"
        reccol = ((220, 90, 90) if self.capture is not None or self.bci is not None
                  else (140, 146, 164))
        win.blit(self.font_sm.render(rec, True, reccol), (18, 40))

        # --- progress bar ----------------------------------------------
        if self.started and self.phase != self.DONE:
            total = self.cue_ms if self.phase == self.CUE else self.rest_ms
            left_ms = max(0, self.phase_end - self._clock())
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

    script_source = script.read_text()
    explicit_integration = "__bci_recorder_frame__" in script_source

    if args.button_mi or args.intent_mi:
        preload_started = time.perf_counter()
        if args.button_mi:
            _load_button_label_fn()
        else:
            _load_motor_intent_api()
        if args.debug_timing:
            print(f"[REC DEBUG] preloaded MI label module in "
                  f"{(time.perf_counter() - preload_started) * 1000.0:.1f}ms",
                  flush=True)

    BCI = capture = None
    capture_mode = args.mi or args.button_mi or args.intent_mi
    if capture_mode:
        # Calibration modes own the stream through StreamBridge's
        # BrainFlowBackend, because they must keep every sample to epoch
        # afterwards. Direct-button data is saved in an isolated directory.
        if not args.subject:
            print("--mi/--button-mi/--intent-mi needs --subject")
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
                       mi=args.mi, button_mi=args.button_mi,
                       intent_mi=args.intent_mi, intent_map=args.intent_map,
                       idle_trials=args.idle_trials,
                       button_hand=args.button_hand, capture=capture)
    if capture_mode and args.cue < 3.0:
        print(f"  WARNING: cue {args.cue}s is shorter than the 3.0s epoch "
              f"window, so most of every epoch would be rest. Use --cue 4 "
              f"or more.")

    print(f"\nRunning {script.name}")
    if args.intent_mi:
        print(f"  {args.trials} action trials + {args.idle_trials} idle trials · "
              f"{args.cue}s cue · {args.rest}s rest")
    else:
        print(f"  {args.trials} trials · {args.cue}s cue · {args.rest}s rest")
    print("  recorder integration: " +
          ("explicit app frame hook" if explicit_integration
           else "legacy pygame fallback hooks"))
    if args.button_mi:
        print("  imagine the complete reach-and-press action; do NOT physically click\n")
    elif args.intent_mi:
        print("  follow the physical motor-intent cue; do NOT physically move\n")
    else:
        print("  the flashing button is your target - try to select it\n")

    namespace = {"__name__": "__main__", "__file__": str(script)}
    original_flip = pygame.display.flip
    original_update = pygame.display.update
    original_event_get = pygame.event.get
    last_step_ms = [-1e18]
    last_callback_ms = [time.monotonic() * 1000.0]
    last_debug_ms = [-1e18]
    callback_counts = {"explicit": 0, "flip": 0, "update": 0, "event": 0}
    explicit_hook_used = [False]
    buttons_prewarmed = [False]

    def prewarm_first_cue(win):
        """Pay pygame/SDL's first animated-alpha-draw cost before REST starts.

        On one Windows setup the first call to Button._draw_flash() blocked the
        render thread for ~4 seconds. The recorder clock correctly entered CUE,
        but the user could not see those first four seconds. Rendering one
        off-screen flash per button here moves any lazy SDL/font initialisation
        before the timed protocol.
        """
        if buttons_prewarmed[0] or not rec.buttons:
            return
        started = time.perf_counter()
        warm = pygame.Surface(win.get_size(), pygame.SRCALPHA)
        for button in rec.buttons:
            button.flash(loop=True, waves=2, pulse_hz=0.5)
            try:
                button.draw(warm)
            except TypeError:
                # Generic BCI_RECORD_TARGETS may require app-owned font args.
                # Their drawing is already exercised by the app each frame.
                pass
            button.stop_flash()
        rec._fonts()
        rec.font_big.render("IMAGINE PRESSING", True, (250, 214, 110))
        rec.font_sm.render(
            "feel your RIGHT hand reach to its screen position and press",
            True, (220, 226, 240))
        buttons_prewarmed[0] = True
        elapsed = (time.perf_counter() - started) * 1000.0
        if args.debug_timing:
            print(f"[REC DEBUG] prewarmed first cue draw in {elapsed:.1f}ms",
                  flush=True)

    def debug_line(source, wall_ms, force=False):
        if not args.debug_timing:
            return
        if not force and wall_ms - last_debug_ms[0] < 1000.0:
            return
        last_debug_ms[0] = wall_ms
        left = (max(0.0, rec.phase_end - rec._clock()) / 1000.0
                if rec.started and rec.phase != rec.DONE else 0.0)
        samples = capture.n_samples if capture is not None else -1
        thread_alive = bool(capture is not None and capture._thread is not None
                            and capture._thread.is_alive())
        display = pygame.display.get_surface()
        print(
            f"[REC DEBUG] source={source:<8} phase={rec.phase:<4} "
            f"trial={rec.trial}/{rec.n_trials} left={left:5.2f}s "
            f"pygame_ms={pygame.time.get_ticks()} samples={samples} "
            f"capture_thread={'alive' if thread_alive else 'n/a'} "
            f"display={'yes' if display is not None else 'no'} "
            f"calls={dict(callback_counts)}",
            flush=True,
        )

    def recorder_step(draw=False, source="unknown"):
        """Advance independently of an app's chosen display-present function.

        The old recorder advanced only inside pygame.display.flip(). Apps that
        use display.update(), or skip a presentation while resting, could leave
        the state machine parked in REST forever. Event pumping now advances it
        too; flip/update are both hooked for the overlay.
        """
        wall_ms = time.monotonic() * 1000.0
        callback_counts[source] = callback_counts.get(source, 0) + 1
        last_callback_ms[0] = wall_ms

        win = pygame.display.get_surface()
        if win is None:
            debug_line(source, wall_ms)
            return
        if not rec.buttons:
            rec.buttons = rec.find_buttons(namespace)
            if rec.buttons:
                rec._wrap_clicks()
                print(f"  found {len(rec.buttons)} buttons: "
                      f"{', '.join(b.name for b in rec.buttons)}\n")
        if not rec.buttons:
            debug_line(source, wall_ms)
            return
        if rec.intent_mi and not rec.intent_map:
            default_map = namespace.get("BCI_INTENT_MAP")
            if not default_map:
                raise ValueError(
                    "--intent-mi needs BCI_INTENT_MAP in the example or "
                    "--intent-map action=intent,...")
            rec.set_intent_map(default_map)

        prewarm_first_cue(win)

        # event.get() and display.flip()/update() usually happen in the same
        # frame. Advance at most once within 2 ms, but draw on every present.
        update_ms = 0.0
        if wall_ms - last_step_ms[0] >= 2.0:
            before = (rec.phase, rec.trial)
            update_started = time.perf_counter()
            try:
                rec.update(pygame.time.get_ticks())
            except Exception:
                if args.debug_timing:
                    import traceback
                    print("[REC DEBUG] exception inside rec.update():",
                          flush=True)
                    traceback.print_exc()
                raise
            update_ms = (time.perf_counter() - update_started) * 1000.0
            last_step_ms[0] = wall_ms
            after = (rec.phase, rec.trial)
            if args.debug_timing and after != before:
                print(f"[REC DEBUG] TRANSITION {before[0]} {before[1]} -> "
                      f"{after[0]} {after[1]} at monotonic_ms={wall_ms:.0f}",
                      flush=True)
        draw_ms = 0.0
        if draw:
            draw_started = time.perf_counter()
            rec.draw(win)
            draw_ms = (time.perf_counter() - draw_started) * 1000.0
        if args.debug_timing and max(update_ms, draw_ms) >= 250.0:
            print(f"[REC DEBUG] SLOW RECORDER update={update_ms:.1f}ms "
                  f"overlay_draw={draw_ms:.1f}ms phase={rec.phase} "
                  f"trial={rec.trial}", flush=True)
        debug_line(source, wall_ms)

    def hooked_flip(*a, **k):
        recorder_step(draw=not explicit_hook_used[0], source="flip")
        return original_flip(*a, **k)

    def hooked_update(*a, **k):
        recorder_step(draw=not explicit_hook_used[0], source="update")
        return original_update(*a, **k)

    def hooked_event_get(*a, **k):
        events = original_event_get(*a, **k)
        recorder_step(draw=False, source="event")
        return events

    def explicit_recorder_frame():
        """Stable integration point used by the two main example apps.

        Monkey-patching pygame works for legacy examples, but some pygame/SDL
        builds cache display functions internally. The main demos call this
        hook themselves once per frame, after drawing and before presenting.
        """
        explicit_hook_used[0] = True
        recorder_step(draw=True, source="explicit")

    # The executed script sees these globals without importing data_record.py.
    namespace["__bci_recorder_frame__"] = explicit_recorder_frame
    namespace["__bci_debug_timing__"] = bool(args.debug_timing)

    debug_stop = threading.Event()
    debug_thread = None

    def debug_watchdog():
        """Report when the app stops calling every recorder integration path."""
        while not debug_stop.wait(1.0):
            now_ms = time.monotonic() * 1000.0
            silent_s = (now_ms - last_callback_ms[0]) / 1000.0
            if silent_s >= 2.0:
                samples = capture.n_samples if capture is not None else -1
                thread_alive = bool(
                    capture is not None and capture._thread is not None
                    and capture._thread.is_alive())
                left = (max(0.0, rec.phase_end - rec._clock()) / 1000.0
                        if rec.started and rec.phase != rec.DONE else 0.0)
                print(
                    f"[REC WATCHDOG] NO FRAME CALLBACK for {silent_s:.1f}s | "
                    f"phase={rec.phase} trial={rec.trial}/{rec.n_trials} "
                    f"left={left:.2f}s samples={samples} "
                    f"capture_thread={'alive' if thread_alive else 'n/a'} | "
                    "the pygame app loop/display is stalled",
                    flush=True,
                )

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
    if args.debug_timing:
        print("[REC DEBUG] build=2026-10-13-debug-2", flush=True)
        print(f"[REC DEBUG] python={sys.version.split()[0]} "
              f"pygame={pygame.version.ver} platform={sys.platform}", flush=True)
        print(f"[REC DEBUG] script={script} integration="
              f"{'explicit' if explicit_integration else 'legacy'} "
              f"cue={args.cue}s rest={args.rest}s trials={args.trials}",
              flush=True)
        print(f"[REC DEBUG] cwd={run_dir} capture_mode={capture_mode} "
              f"dry_run={args.dry_run}", flush=True)
        debug_thread = threading.Thread(
            target=debug_watchdog, daemon=True, name="RecorderDebugWatchdog")
        debug_thread.start()

    try:
        if not explicit_integration:
            # Legacy examples have no recorder callback, so retain the original
            # monkey-patch fallback. The two main demos deliberately avoid it:
            # replacing pygame's C-backed presentation functions caused stale
            # frames on some Windows/SDL builds (REST stayed at 0.0, then the
            # next cue appeared several seconds late).
            pygame.display.flip = hooked_flip
            pygame.display.update = hooked_update
            pygame.event.get = hooked_event_get
        exec(compile(script_source, str(script), "exec"), namespace)
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
        if args.debug_timing:
            import traceback
            print("[REC DEBUG] unhandled app/recorder exception:", flush=True)
            traceback.print_exc()
        failed = f"{script.name} raised {type(exc).__name__}: {exc}"
    finally:
        debug_stop.set()
        if debug_thread is not None:
            debug_thread.join(timeout=2.0)
        pygame.display.flip = original_flip
        pygame.display.update = original_update
        pygame.event.get = original_event_get
        os.chdir(prev_cwd)
        if failed:
            print(f"\n  SCRIPT ERROR\n  {failed}")
        path = rec.save_log()
        if path:
            if args.button_mi:
                print(f"\n  {len(rec.log)} button-imagery trials recorded; "
                      "mental compliance is not observable")
            elif args.intent_mi:
                print(f"\n  {len(rec.log)} motor-intent trials recorded; "
                      "mental compliance is not observable")
            else:
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
            if args.button_mi:
                X, y, labels = epoch_button_mi_from_log(
                    stream, capture.fs, rec.log, t0_ms=0)
            elif args.intent_mi:
                X, y, labels = epoch_intent_mi_from_log(
                    stream, capture.fs, rec.log, t0_ms=0)
            else:
                X, y, labels = epoch_from_log(stream, capture.fs, rec.log,
                                              t0_ms=0)
            if len(X) == 0:
                print("  no usable epochs - was the cue shorter than 3 s?")
            else:
                if args.out:
                    out = Path(args.out)
                elif args.button_mi:
                    from ML.button_mi import button_profile_dir
                    out = button_profile_dir(args.subject, script.stem)
                elif args.intent_mi:
                    api = _load_motor_intent_api()
                    out = api.intent_profile_dir(args.subject, rec.intent_values)
                else:
                    out = ROOT / "profiles" / args.subject.lower()
                npz = save_session(X, y, labels, args.subject, capture.fs,
                                   args.montage, out)
                if args.button_mi:
                    counts = [(b.name, int((y == i).sum()))
                              for i, b in enumerate(rec.buttons)]
                    print(f"  {len(X)} direct-button epochs  (" +
                          " / ".join(f"{name} {n}" for name, n in counts) + ")")
                    print(f"  button calibration -> {npz}")
                    print(f"  next:  python ML/train_button_mi.py "
                          f"{args.subject} --app {script.stem}")
                elif args.intent_mi:
                    counts = [(name, int((y == index).sum()))
                              for name, index in rec.intent_class.items()]
                    print(f"  {len(X)} reusable motor-intent epochs  (" +
                          " / ".join(f"{name} {n}" for name, n in counts) + ")")
                    print(f"  intent calibration -> {npz}")
                    joined = ",".join(rec.intent_values)
                    print(f"  next:  python ML/train_intent_mi.py "
                          f"{args.subject} --intents {joined}")
                else:
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
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--mi", action="store_true",
                       help="existing calibration mode: LEFT/RIGHT hand labels; "
                            "writes data for ML/train_local.py")
    modes.add_argument("--button-mi", action="store_true",
                       help="legacy experimental button-labelled imagery mode")
    modes.add_argument("--intent-mi", "--finger-mi", dest="intent_mi",
                       action="store_true",
                       help="recommended reusable physical motor-intent mode")
    ap.add_argument("--intent-map", "--finger-map", dest="intent_map",
                    default=None,
                    help="override action=intent pairs; otherwise use the "
                         "example's BCI_INTENT_MAP")
    ap.add_argument("--idle-trials", type=int, default=10,
                    help="no-movement trials added in --intent-mi (default 10)")
    ap.add_argument("--button-hand", choices=["left", "right"], default="right",
                    help="fixed imagined hand for legacy --button-mi")
    ap.add_argument("--subject", default=None,
                    help="subject name for any MI calibration mode")
    ap.add_argument("--montage", default="cyton8_motor")
    ap.add_argument("--out", default=None,
                    help="output override; default keeps all MI modes isolated")
    ap.add_argument("--debug-timing", action="store_true",
                    help="print a 1 Hz recorder heartbeat, phase transitions, "
                         "frame-hook activity and capture-thread state")
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
