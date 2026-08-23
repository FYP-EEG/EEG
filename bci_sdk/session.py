"""
BCISession -- the one object a developer needs to get brain commands.

Wires backend -> StreamBridge -> BCIEngine and delivers decoded commands to
callbacks, hiding the ring buffer, window cadence, debouncing and confidence
thresholding behind a small surface.

    import bci_sdk

    session = bci_sdk.BCISession(source="sim")
    session.on_command(lambda c: print(c.name, c.confidence))
    session.start()
    while running:
        session.poll()          # non-blocking, call once per frame
    session.stop()
"""

import glob
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ML.bci_engine import BCIEngine
from stream_bridge import (StreamBridge, SyntheticBackend, ReplayBackend,
                           BrainFlowBackend)

from .errors import CalibrationError, HardwareError

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Command:
    """A decoded user intent handed to your callback.

    :param name:       'SSVEP_0', 'SSVEP_1', 'MI_0', ...
    :param index:      0-based command index (0 = first target frequency)
    :param confidence: classifier score behind the decision
    :param state:      raw engine state string
    :param timestamp:  time.time() at dispatch
    """
    name: str
    index: int
    confidence: float = 0.0
    state: str = ""
    timestamp: float = field(default_factory=time.time)

    def __str__(self):
        return f"{self.name}(conf={self.confidence:.3f})"


class BCISession:
    """High-level façade over acquisition + decoding."""

    #: engine states that are informational, not commands
    IDLE_STATES = ("NO_ACTION", "BUILDING", "REFRACTORY", "LOCKOUT")

    def __init__(self, source="sim", profile=None, montage="cyton8_ssvep",
                 target_freqs=(15.0, 20.0), sample_freq=250, mode="SSVEP",
                 window=750, hop=250, debounce=3, confidence_threshold=0.15,
                 serial_port="COM4", board_id=0, replay_npz=None,
                 fallback_to_sim=True, model=None):
        """
        :param source: 'sim' | 'replay' | 'hardware'
        :param profile: path to a calibration profile json (Step 3). Overrides
                        target_freqs / threshold / montage with fitted values.
        :param fallback_to_sim: if hardware is unavailable, silently use synthetic
                        data instead of raising. Set False to fail loudly.
        :param model: path to a TRCA model saved with TRCAClassifier.save().
                        Trained offline (Colab on public data, or on your own
                        recording) and used here at runtime. Falls back to the
                        training-free FBCCA classifier when omitted.
        """
        self.source = source
        self.profile_path = profile
        self.profile = None
        self._command_cbs = []
        self._state_cbs = []
        self._raw_cbs = []
        self._running = False
        self.last_state = "-"
        self.last_command = None
        self.model = None
        self.model_path = None

        if profile:
            p = Path(profile)
            if not p.exists():
                hits = sorted(glob.glob(str(profile)))
                if not hits:
                    raise CalibrationError(f"No profile found at {profile!r}")
                p = Path(hits[-1])
            try:
                self.profile = json.loads(p.read_text())
            except Exception as exc:
                raise CalibrationError(f"Could not read profile {p}: {exc}") from exc
            montage = self.profile.get("montage", montage)
            target_freqs = tuple(self.profile.get("target_freqs", target_freqs))
            sample_freq = self.profile.get("sample_freq", sample_freq)
            confidence_threshold = self.profile.get("confidence_threshold",
                                                    confidence_threshold)
            debounce = self.profile.get("debounce_window", debounce)

        self.target_freqs = tuple(target_freqs)
        self.sample_freq = sample_freq

        self.engine = BCIEngine(
            mode=mode, debounce_window=debounce,
            confidence_threshold=confidence_threshold,
            montage=montage, target_freqs=self.target_freqs,
            sample_freq=sample_freq)

        self.model_path = model
        if model:
            self._load_model(model)

        backend = self._make_backend(source, serial_port, board_id, replay_npz,
                                     fallback_to_sim)
        self.bridge = StreamBridge(backend=backend, window=window, hop=hop)

    # ---------------------------------------------------------------- model
    def _load_model(self, model):
        """Swap the engine's SSVEP classifier for a pre-trained TRCA model."""
        from ML.trca import TRCAClassifier
        p = Path(model)
        if not p.exists():
            hits = sorted(glob.glob(str(model)))
            if not hits:
                raise CalibrationError(f"No model file found at {model!r}")
            p = Path(hits[-1])
        try:
            trained = TRCAClassifier.load(p)
        except Exception as exc:
            raise CalibrationError(f"Could not load model {p}: {exc}") from exc

        if tuple(trained.target_freqs) != tuple(self.target_freqs):
            raise CalibrationError(
                f"Model was trained for {trained.target_freqs} Hz but this session "
                f"uses {list(self.target_freqs)} Hz. Retrain, or match the "
                f"stimulus frequencies.")

        fbcca = self.engine.ssvep_clf
        trained.channels = trained.channels or fbcca.ssvep_channels

        class _TRCAAdapter:
            """Gives TRCA the HybridSSVEPClassifier interface the engine expects."""
            def __init__(self, trca, fb):
                self._t, self._fb = trca, fb
                self.target_freqs = trca.target_freqs
                self.ssvep_channels = fb.ssvep_channels
                self.artifact_channels = fb.artifact_channels
                self.confidence_threshold = trca.confidence_threshold
            def artifact_detection(self, d, is_raw=True):
                return self._fb.artifact_detection(d, is_raw=is_raw)
            def apply_filter(self, d, sos=None):
                return self._fb.apply_filter(d, sos)
            def predict_proba(self, raw, auto_filter=True):
                if self.artifact_detection(raw, is_raw=auto_filter):
                    import numpy as _np
                    return "BLINK", 1.0, _np.zeros(len(self.target_freqs))
                data = self.apply_filter(raw) if auto_filter else raw
                return self._t.predict_proba(data)

        self.engine.ssvep_clf = _TRCAAdapter(trained, fbcca)
        self.model = trained
        return trained

    # ------------------------------------------------------------- backends
    def _make_backend(self, source, serial_port, board_id, replay_npz, fallback):
        if source == "hardware":
            try:
                return BrainFlowBackend(board_id=board_id, serial_port=serial_port)
            except Exception as exc:
                if not fallback:
                    raise HardwareError(
                        f"Could not open the board: {exc}. Pass "
                        f"fallback_to_sim=True to degrade to synthetic data.") from exc
                self.source = "sim (hardware unavailable)"
                return SyntheticBackend(seed=1, target_freqs=self.target_freqs)

        if source == "replay":
            pattern = replay_npz or str(ROOT / "dataset" / "calib_*.npz")
            hits = sorted(glob.glob(pattern))
            if not hits:
                if not fallback:
                    raise CalibrationError(f"No recording matched {pattern!r}")
                self.source = "sim (no recording found)"
                return SyntheticBackend(seed=1, target_freqs=self.target_freqs)
            return ReplayBackend(hits[-1], loop=True)

        return SyntheticBackend(seed=1, target_freqs=self.target_freqs)

    # ------------------------------------------------------------ callbacks
    def on_command(self, fn):
        """Register callback(Command) for confirmed selections. Returns fn."""
        self._command_cbs.append(fn)
        return fn

    def on_state(self, fn):
        """Register callback(state:str) for every engine state, including idle."""
        self._state_cbs.append(fn)
        return fn

    def on_raw_window(self, fn):
        """Register callback(ndarray) for each raw EEG window. For plotting/logging."""
        self._raw_cbs.append(fn)
        return fn

    # ------------------------------------------------------------ lifecycle
    def start(self):
        self.bridge.start()
        self._running = True
        return self

    def stop(self):
        self._running = False
        self.bridge.stop()
        return self

    @property
    def is_running(self):
        return self._running and self.bridge.is_running

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ----------------------------------------------------------------- poll
    def poll(self):
        """Process at most one window. Non-blocking. Returns a Command or None.

        Call this once per render frame. Returning None is the normal case --
        windows arrive about once per `hop` samples, not once per frame.
        """
        if not self._running:
            return None
        window = self.bridge.poll()
        if window is None:
            return None

        for cb in self._raw_cbs:
            try:
                cb(window)
            except Exception:
                pass

        state, triggered = self.engine.process_frame(window)
        self.last_state = state
        for cb in self._state_cbs:
            try:
                cb(state)
            except Exception:
                pass

        if not triggered:
            return None

        idx = -1
        if isinstance(state, str) and "_" in state:
            tail = state.rsplit("_", 1)[-1]
            if tail.isdigit():
                idx = int(tail)
        cmd = Command(name=state, index=idx,
                      confidence=float(getattr(self.engine, "last_confidence", 0.0)),
                      state=state)
        self.last_command = cmd
        for cb in self._command_cbs:
            try:
                cb(cmd)
            except Exception:
                pass
        return cmd

    def run(self, seconds=None, fps=60):
        """Blocking convenience loop for scripts and tests (not for UIs)."""
        self.start()
        end = None if seconds is None else time.time() + seconds
        try:
            while self.is_running and (end is None or time.time() < end):
                self.poll()
                time.sleep(1.0 / fps)
        finally:
            self.stop()
        return self

    # -------------------------------------------------------------- reports
    def report(self):
        r = {"source": self.source, "target_freqs": list(self.target_freqs),
             "calibrated": self.profile is not None,
             "model": self.model_path,
             "classifier": type(self.engine.ssvep_clf).__name__}
        r.update({f"engine_{k}": v for k, v in self.engine.report().items()})
        r.update({f"stream_{k}": v for k, v in self.bridge.report().items()})
        return r

    def print_report(self):
        print(f"\nBCISession ({self.source})")
        print(f"  targets       : {list(self.target_freqs)} Hz")
        print(f"  calibrated    : {'yes' if self.profile else 'no (default threshold)'}")
        print(f"  classifier    : {'trained TRCA' if self.model_path else 'FBCCA (training-free)'}")
        self.bridge.print_report()
        e = self.engine.report()
        print(f"  commands sent : {e['dispatched']}")
        print(f"  idle rate     : {e['no_action_rate']:.1%}")
