"""
Author: Anson
Created on: 6/10/2026
Purpose: the real-time decision path -- window in, button action out.

    raw window -> artifact check -> local model -> confidence gate
               -> consecutive agreement -> refractory -> SelectionTree -> button

Everything runs on this machine. Nothing in this module opens a network
connection; the model is a file on disk loaded at start-up.

THE GATES, AND WHY EACH ONE IS HERE
-----------------------------------
Measured on this project's own data, a forced argmax scored 100% on attentive
trials while firing on 141 of 141 idle windows. The same classifier behind the
gates below fired on 0 of 105. Accuracy is not the metric that decides whether
a BCI is usable; false activations per minute at rest is.

  artifact lockout   a blink is 221.6 uV raw. It must be detected on RAW data in
                     its own 0.5-8 Hz band: the 8-30 Hz motor filter removes it
                     almost entirely (221.6 uV -> 60.2 uV, under the 100 uV
                     threshold), so checking after filtering finds 0 of 32.
  confidence gate    inside LocalMIModel; below threshold the window is simply
                     discarded and the user tries again. Rejection costs time,
                     not an error.
  consecutive run    3 windows must agree. A majority vote (3 of 5) still fires
                     on noise; an unbroken run does not.
  refractory         1.2 s minimum between dispatches, so one sustained effort
                     is one command rather than ten.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.local_model import LocalMIModel, NO_ACTION        # noqa: E402
from bci_sdk.selection import SelectionTree, LEFT, RIGHT  # noqa: E402

# states the decider can report for a window
BLINK = "BLINK"
LOCKOUT = "LOCKOUT"
BUILDING = "BUILDING"
REFRACTORY = "REFRACTORY"
REJECTED = "REJECTED"        # confidence gate said no -- the reject option
DISPATCHED = "DISPATCHED"


class ArtifactDetector:
    """Blink / EMG rejection on RAW frontal channels.

    Runs before any motor-band filtering, because that filter destroys the very
    signal this is looking for.
    """

    def __init__(self, channels=(0, 1), sample_freq=250, band=(0.5, 8.0),
                 threshold_uv=100.0):
        self.channels = list(channels)
        self.threshold_uv = float(threshold_uv)
        self.sos = butter(4, list(band), btype="band", fs=sample_freq,
                          output="sos")

    def peak_uv(self, window):
        x = np.asarray(window, dtype=np.float64)
        idx = [c for c in self.channels if c < x.shape[0]]
        if not idx:
            return 0.0
        f = sosfiltfilt(self.sos, x[idx], axis=-1)
        return float(np.max(np.abs(f)))

    def is_artifact(self, window):
        return self.peak_uv(window) > self.threshold_uv


class RealtimeDecider:
    """Turns a stream of windows into button selections."""

    def __init__(self, model, tree, artifact=None, debounce_window=3,
                 artifact_lockout_ms=800, refractory_ms=1200,
                 on_state=None, clock=None):
        """
        :param model: LocalMIModel
        :param tree: SelectionTree
        :param clock: () -> milliseconds. Injected so tests need no sleeps.
        """
        self.model = model
        self.tree = tree
        self.artifact = artifact
        self.debounce_window = int(debounce_window)
        self.artifact_lockout_ms = float(artifact_lockout_ms)
        self.refractory_ms = float(refractory_ms)
        self.on_state = on_state
        self._clock = clock or (lambda: time.monotonic() * 1000.0)

        self._run_label = None
        self._run_len = 0
        self._locked_until = -1e18
        self._last_dispatch = -1e18
        self.last_confidence = 0.0
        self.stats = {"windows": 0, "rejected": 0, "blink": 0, "locked_out": 0,
                      "building": 0, "refractory_suppressed": 0,
                      "dispatched": 0, "selections": 0}

    # ----------------------------------------------------------------- misc
    @property
    def is_locked_out(self):
        return self._clock() < self._locked_until

    def _reset_run(self):
        self._run_label, self._run_len = None, 0

    def _emit(self, state, command=None, selected=None):
        if self.on_state:
            self.on_state(state, command, selected)
        return {"state": state, "command": command, "selected": selected,
                "confidence": self.last_confidence,
                "candidates": self.tree.candidate_buttons()}

    def reset(self):
        self._reset_run()
        self._locked_until = -1e18
        self._last_dispatch = -1e18
        self.tree.reset()

    # ----------------------------------------------------------------- step
    def step(self, window):
        """Process exactly one (channels, samples) window.

        :return: dict with state, command, selected, confidence, candidates
        """
        self.stats["windows"] += 1
        now = self._clock()

        if self.is_locked_out:
            self.stats["locked_out"] += 1
            self._reset_run()
            return self._emit(LOCKOUT)

        if self.artifact is not None and self.artifact.is_artifact(window):
            self.stats["blink"] += 1
            self._locked_until = now + self.artifact_lockout_ms
            self._reset_run()
            return self._emit(BLINK)

        command, conf = self.model.predict_command(window)
        self.last_confidence = conf

        if command is None:
            self.stats["rejected"] += 1
            self._reset_run()
            return self._emit(REJECTED)

        if command == self._run_label:
            self._run_len += 1
        else:
            self._run_label, self._run_len = command, 1

        if self._run_len < self.debounce_window:
            self.stats["building"] += 1
            return self._emit(BUILDING, command)

        if now - self._last_dispatch < self.refractory_ms:
            self.stats["refractory_suppressed"] += 1
            return self._emit(REFRACTORY, command)

        self._last_dispatch = now
        self._reset_run()
        self.stats["dispatched"] += 1
        selected = self.tree.decide(command)
        if selected is not None:
            self.stats["selections"] += 1
        return self._emit(DISPATCHED, command, selected)

    # ------------------------------------------------------------ live loop
    def run(self, bridge, max_seconds=None, stop=None):
        """Drive the decider from a StreamBridge until stopped.

        The bridge owns the thread and the ring buffer; this only consumes
        whatever complete windows it has produced.
        """
        t0 = time.monotonic()
        while True:
            if stop is not None and stop():
                break
            if max_seconds is not None and time.monotonic() - t0 > max_seconds:
                break
            win = bridge.wait_for_window(timeout=1.0)
            if win is None:
                continue
            yield self.step(win)

    # ----------------------------------------------------------------- info
    def report(self):
        w = max(1, self.stats["windows"])
        return {**self.stats,
                "reject_rate": self.stats["rejected"] / w,
                "dispatch_rate": self.stats["dispatched"] / w}


# ------------------------------------------------------------------ factory
def build_runtime(user, buttons, root=None, montage=None, on_select=None,
                  on_change=None, on_state=None, confidence_threshold=None,
                  **decider_kw):
    """Load this user's local model and wire it to a tree over `buttons`.

        rt = build_runtime("anson", [inventory, message, shoot, reload, map_])
        rt.step(window)

    Raises FileNotFoundError when the user has no model and no starter model is
    bundled, rather than running an untrained system.
    """
    model = LocalMIModel.load(user, root=root,
                              confidence_threshold=confidence_threshold)

    art_channels = (0, 1)
    name = montage or model.montage
    if name:
        try:
            from ML.montage import get
            art_channels = tuple(get(name).get("artifact", (0, 1)))
        except Exception:
            pass

    tree = SelectionTree(buttons, on_select=on_select, on_change=on_change)
    detector = ArtifactDetector(channels=art_channels,
                                sample_freq=model.sample_freq)
    return RealtimeDecider(model, tree, artifact=detector, on_state=on_state,
                           **decider_kw)
