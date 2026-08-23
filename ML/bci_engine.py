"""
Author: Brian
Created on: 10/08/2026
Purpose: key middleware between the UDP/OpenBCI data stream and the game UI controller

Edited 16/08/2026 -- implements instructions.md Gap 4 (idle state & distraction handling):
  1. FIXED filter bypass: previously called cca_ssvep_detection() directly, so the
     1-45 Hz pre-filter never ran on the live path -- unlike the validated notebook.
     Now routes through predict_proba(), which filters.
  2. Confidence thresholding -> NO_ACTION instead of a forced argmax that types
     garbage 100% of the time the user is idle.
  3. Debounce now requires N CONSECUTIVE agreeing windows (a run), not a majority
     vote over a mixed window. A majority vote still fires on 3-of-5 noise.
  4. Artifact lockout: a blink pauses classifier output for 500-1000 ms.
  5. Refractory period after a dispatch, so one long gaze != ten repeated keys.
"""

import time
from collections import deque

import numpy as np

from ML.hybrid_classifier import (HybridSSVEPClassifier, MotorImageryClassifier,
                                  NO_ACTION)


class BCIEngine:
    def __init__(self, mi_model_path=None, mode='SSVEP', debounce_window=3,
                 confidence_threshold=0.15, montage="cyton8_ssvep",
                 target_freqs=(15.0, 20.0), sample_freq=250,
                 artifact_lockout_ms=800, refractory_ms=1200,
                 on_state_change=None):
        """
        :param mode: 'SSVEP', 'MI', or 'HYBRID'
        :param debounce_window: number of CONSECUTIVE agreeing windows required
        :param artifact_lockout_ms: output paused this long after a blink
        :param refractory_ms: minimum gap between two dispatched commands
        """
        self.mode = mode
        self.debounce_window = debounce_window
        self.artifact_lockout_ms = artifact_lockout_ms
        self.refractory_ms = refractory_ms
        self.on_state_change = on_state_change

        self.ssvep_clf = HybridSSVEPClassifier(
            target_freqs=target_freqs, sample_freq=sample_freq,
            montage=montage, confidence_threshold=confidence_threshold)
        self.mi_clf = MotorImageryClassifier(model_path=mi_model_path,
                                             sample_freq=sample_freq)

        self.history = deque(maxlen=max(8, debounce_window * 2))
        self._run_label = None
        self._run_len = 0
        self._locked_until = 0.0
        self._last_dispatch = 0.0
        #: confidence behind the most recent classification (exposed to the SDK)
        self.last_confidence = 0.0

        # telemetry -- the metrics that actually matter for a GUI (see accuracy_strategy.md)
        self.stats = {"windows": 0, "no_action": 0, "blink": 0,
                      "locked_out": 0, "dispatched": 0, "refractory_suppressed": 0}

    # ------------------------------------------------------------------ profile
    @classmethod
    def from_profile(cls, profile_path, mode="SSVEP", **overrides):
        """
        Build an engine from a calibration profile produced by ML/calibration.py.

        This is how per-subject calibration (Step 3) reaches the runtime: the
        confidence threshold is the one fitted to THIS user's idle recording,
        not a simulation-derived default.
        """
        import json
        with open(profile_path) as fh:
            p = json.load(fh)
        kwargs = dict(
            mode=mode,
            montage=p.get("montage", "cyton8_ssvep"),
            target_freqs=tuple(p.get("target_freqs", (15.0, 20.0))),
            sample_freq=p.get("sample_freq", 250),
            confidence_threshold=p.get("confidence_threshold", 0.15),
            debounce_window=p.get("debounce_window", 3),
        )
        kwargs.update(overrides)
        eng = cls(**kwargs)
        eng.profile = p
        return eng

    # ------------------------------------------------------------------- timing
    def _now_ms(self):
        return time.time() * 1000.0

    @property
    def is_locked_out(self):
        return self._now_ms() < self._locked_until

    def _reset_run(self):
        self._run_label, self._run_len = None, 0

    # ------------------------------------------------------------------ process
    def process_frame(self, raw_trial_data):
        """
        Feed one sliding window of shape (channels, n_times).

        :return: (decision, is_triggered)
                 decision is a command string ('SSVEP_0', 'MI_1', ...) when
                 is_triggered is True; otherwise it is a status string
                 ('NO_ACTION', 'BLINK', 'LOCKOUT', 'BUILDING', 'REFRACTORY').
        """
        self.stats["windows"] += 1
        now = self._now_ms()

        # ---- 1. artifact lockout (Gap 4, item 3)
        if self.is_locked_out:
            self.stats["locked_out"] += 1
            self._reset_run()
            return self._emit("LOCKOUT", False)

        raw_pred, confidence = self._classify(raw_trial_data)
        self.last_confidence = float(confidence)

        if raw_pred == "BLINK":
            self.stats["blink"] += 1
            self._locked_until = now + self.artifact_lockout_ms
            self._reset_run()
            return self._emit("BLINK", False)

        # ---- 2. confidence gate (Gap 4, item 1)
        if raw_pred is None:
            self.stats["no_action"] += 1
            self._reset_run()
            return self._emit("NO_ACTION", False)

        # ---- 3. consecutive-agreement debounce (Gap 4, item 2)
        self.history.append((raw_pred, confidence))
        if raw_pred == self._run_label:
            self._run_len += 1
        else:
            self._run_label, self._run_len = raw_pred, 1

        if self._run_len < self.debounce_window:
            return self._emit("BUILDING", False)

        # ---- 4. refractory period
        if now - self._last_dispatch < self.refractory_ms:
            self.stats["refractory_suppressed"] += 1
            return self._emit("REFRACTORY", False)

        self._last_dispatch = now
        self._reset_run()
        self.stats["dispatched"] += 1
        return self._emit(raw_pred, True)

    def _emit(self, state, triggered):
        if self.on_state_change:
            self.on_state_change(state, triggered)
        return state, triggered

    # ----------------------------------------------------------------- classify
    def _classify(self, raw_trial_data):
        """:return: (command_string | 'BLINK' | None, confidence)"""
        if self.mode == 'SSVEP':
            label, conf, _ = self.ssvep_clf.predict_proba(raw_trial_data)
            if label == "BLINK":
                return "BLINK", conf
            if label == NO_ACTION:
                return None, conf
            return f"SSVEP_{label}", conf

        if self.mode == 'MI':
            label, conf = self.mi_clf.predict_proba(raw_trial_data)
            if label == NO_ACTION:
                return None, conf
            return f"MI_{label}", conf

        # HYBRID: SSVEP decides selection, MI is advisory only.
        # MI is 61.4% cross-subject, so it must never override a confident SSVEP.
        s_label, s_conf, _ = self.ssvep_clf.predict_proba(raw_trial_data)
        if s_label == "BLINK":
            return "BLINK", s_conf
        if s_label != NO_ACTION:
            return f"SSVEP_{s_label}", s_conf
        m_label, m_conf = self.mi_clf.predict_proba(raw_trial_data)
        if m_label != NO_ACTION:
            return f"MI_{m_label}", m_conf
        return None, max(s_conf, m_conf)

    # -------------------------------------------------------------- diagnostics
    def report(self):
        w = max(1, self.stats["windows"])
        return {**self.stats,
                "no_action_rate": self.stats["no_action"] / w,
                "dispatch_rate": self.stats["dispatched"] / w}

    def reset(self):
        self.last_confidence = 0.0
        self.history.clear()
        self._reset_run()
        self._locked_until = 0.0
        self._last_dispatch = 0.0
