"""
Author: Anson
Created on: 6/10/2026
Purpose: drop a live BCI into an existing pygame app in a few lines.

    bci = LiveBCI(buttons, source="sine")        # plumbing check, no EEG
    bci = LiveBCI(buttons, source="cyton", user="anson", serial_port="COM4")

    while running:                                # once per frame, non-blocking
        r = bci.poll()
        if r and r["selected"]:
            ...

WHAT "source" MEANS
-------------------
  sine     BrainFlow's synthetic board, the same generator the OpenBCI GUI's
           "Synthetic" data source uses: a sine per channel, 5 Hz apart. The
           classifier is SineProbeModel, which maps 10 Hz -> LEFT and
           15 Hz -> RIGHT. Nothing about motor imagery is exercised.
  replay   a recorded .npz played back at wall-clock rate
  cyton    the real board, with this user's trained model

  THE SINE PROVES THE PLUMBING, MOTOR IMAGERY PROVES THE CLASSIFIER.
  A pure sine contains no event-related desynchronisation, so feeding it to a
  CSP+LDA trained on motor imagery produces a meaningless answer. What the sine
  DOES establish, end to end and in real time, is that a signal arrives, gets
  windowed, reaches the classifier, produces a decision, survives the gates and
  moves a button. That is five of the six things that can be broken.

TWO SELECTION SCHEMES
---------------------
  tree   both classes select; the candidate set halves each decision.
         3 buttons -> at most 2 decisions, 5 buttons -> at most 3.
  scan   LEFT advances a highlight, RIGHT confirms. One class carries no
         information about WHICH button is wanted, so it costs more decisions
         (measured: 3.0 vs 2.4 for five buttons, 47 s vs 34 s per selection at
         0.614 per-decision accuracy). Supported because an existing UI may
         already be built around it.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("", "EEG"):
    _p = str(ROOT / _sub) if _sub else str(ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bci_sdk.selection import SelectionTree, LEFT, RIGHT       # noqa: E402
from bci_sdk.runtime import RealtimeDecider, ArtifactDetector  # noqa: E402

NO_ACTION = -1


# =====================================================================
#  SINE STAND-IN CLASSIFIER  (plumbing check, not a BCI)
# =====================================================================
class SineProbe:
    """Find the dominant sinusoid in a window.

    Frequency is refined by parabolic interpolation on the log spectrum, so
    resolution beats the FFT bin width: at 750 samples and 250 Hz the bins are
    0.333 Hz apart and the measured worst-case error is 0.008 Hz.
    """

    #: Threshold set by measuring the null, not by guessing. For WHITE NOISE,
    #: best-of-8-channels peak/median power is already large: 50th pct 11.3,
    #: 99th 19.1, max 27.0 over 3000 windows. So an intuitive 3.0 flags 100% of
    #: noise. At 30 the false-alarm rate is 0/3000, while a 2 uV sine against
    #: 3 uV noise still scores 53.6 at its 5th percentile.
    DEFAULT_MIN_SNR = 30.0

    def __init__(self, sample_freq=250, band=(1.0, 45.0), min_snr=None,
                 channels=None):
        self.fs = int(sample_freq)
        self.band = band
        self.min_snr = self.DEFAULT_MIN_SNR if min_snr is None else min_snr
        self.channels = channels

    def detect(self, window):
        x = np.asarray(window, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        chans = self.channels if self.channels is not None else range(x.shape[0])
        n = x.shape[1]
        win = np.hanning(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        mask = (freqs >= self.band[0]) & (freqs <= self.band[1])
        best = {"detected": False, "freq": None, "amplitude": 0.0,
                "snr": 0.0, "channel": None}
        if not mask.any():
            return best

        for ch in chans:
            if ch >= x.shape[0]:
                continue
            sig = x[ch] - x[ch].mean()
            spec = np.abs(np.fft.rfft(sig * win))
            power = spec ** 2
            band_p = power[mask]
            k_rel = int(np.argmax(band_p))
            k = int(np.flatnonzero(mask)[k_rel])

            others = band_p.copy()
            others[max(0, k_rel - 2):k_rel + 3] = np.nan
            floor = np.nanmedian(others) if np.isfinite(others).any() else 0.0
            snr = float(power[k] / floor) if floor > 0 else float("inf")

            f_hat = float(freqs[k])
            if 0 < k < len(spec) - 1:
                a, b, c = (np.log(power[k - 1] + 1e-30),
                           np.log(power[k] + 1e-30),
                           np.log(power[k + 1] + 1e-30))
                den = a - 2 * b + c
                if den != 0:
                    f_hat = float((k + float(np.clip(0.5 * (a - c) / den,
                                                     -0.5, 0.5)))
                                  * self.fs / n)
            if snr > best["snr"]:
                best = {"detected": bool(snr >= self.min_snr), "freq": f_hat,
                        "amplitude": float(2.0 * spec[k] / win.sum()),
                        "snr": snr, "channel": int(ch)}
        return best

    def reconstruct(self, window, freq=None, channel=None):
        """Least-squares fit at the detected frequency: recovers amplitude AND
        phase, so the output can be overlaid on the input sample by sample."""
        x = np.asarray(window, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if freq is None or channel is None:
            d = self.detect(window)
            freq = d["freq"] if freq is None else freq
            channel = d["channel"] if channel is None else channel
        if freq is None or channel is None:
            return None
        sig = x[channel]
        t = np.arange(len(sig)) / self.fs
        A = np.column_stack([np.sin(2 * np.pi * freq * t),
                             np.cos(2 * np.pi * freq * t), np.ones(len(sig))])
        coef, *_ = np.linalg.lstsq(A, sig, rcond=None)
        wave = A @ coef
        resid = sig - wave
        ss = float(np.sum((sig - sig.mean()) ** 2))
        return {"freq": float(freq),
                "amplitude": float(np.hypot(coef[0], coef[1])),
                "phase": float(np.arctan2(coef[1], coef[0])),
                "wave": wave,
                "residual_rms": float(np.sqrt(np.mean(resid ** 2))),
                "r2": 1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else 0.0}


class SineProbeModel:
    """Duck-types LocalMIModel so the real decision path runs on sine input."""

    def __init__(self, freq_map=None, tolerance_hz=1.0, min_snr=None,
                 sample_freq=250, band=(1.0, 45.0), montage=None,
                 report_frequency=False, channels=None):
        self.freq_map = dict(freq_map or {10.0: LEFT, 15.0: RIGHT})
        self.tolerance_hz = float(tolerance_hz)
        self.report_frequency = bool(report_frequency)
        self.sample_freq = int(sample_freq)
        self.montage = montage
        self.probe = SineProbe(sample_freq=sample_freq, band=band,
                               min_snr=min_snr, channels=channels)
        self.last = None

    def predict_proba(self, window):
        d = self.probe.detect(window)
        self.last = d
        if not d["detected"]:
            return NO_ACTION, 0.0
        return d["freq"], float(min(1.0, d["snr"] / 20.0))

    def predict_command(self, window):
        freq, conf = self.predict_proba(window)
        if freq == NO_ACTION:
            return None, conf
        if self.report_frequency:
            return f"SINE_{freq:.1f}", conf
        for nominal, name in self.freq_map.items():
            if abs(freq - nominal) <= self.tolerance_hz:
                return name, conf
        return None, conf

    def describe(self):
        return {"source": "sine-probe", "pipeline": "SineProbe",
                "freq_map": self.freq_map, "sample_freq": self.sample_freq}

    def __repr__(self):
        m = ", ".join(f"{k:g}Hz->{v}" for k, v in self.freq_map.items())
        return f"<SineProbeModel {m} +/-{self.tolerance_hz:g}Hz>"


# =====================================================================
#  SCANNING  (alternative to the tree, for UIs already built around it)
# =====================================================================
class ScanSelector:
    """LEFT advances the highlight, RIGHT confirms it."""

    def __init__(self, buttons, on_select=None, on_change=None):
        self.buttons = list(buttons)
        self.on_select = on_select
        self.on_change = on_change
        self.index = 0
        self.stats = {"decisions": 0, "selections": 0}

    @property
    def candidates(self):
        return [self.index]

    def candidate_buttons(self):
        return [self.buttons[self.index]]

    def reset(self):
        self.index = 0
        if self.on_change:
            self.on_change(self.candidate_buttons())

    def decide(self, command):
        self.stats["decisions"] += 1
        if command == LEFT:
            self.index = (self.index + 1) % len(self.buttons)
            if self.on_change:
                self.on_change(self.candidate_buttons())
            return None
        b = self.buttons[self.index]
        self.stats["selections"] += 1
        if self.on_select:
            self.on_select(b)
        return b


# =====================================================================
#  THE GLUE
# =====================================================================
class LiveBCI:
    """Stream -> window -> classify -> gate -> select, polled from a game loop."""

    def __init__(self, buttons, source="sine", user=None, scheme="tree",
                 serial_port=None, recording=None, sample_freq=250,
                 window=750, hop=250, on_select=None, on_change=None,
                 freq_map=None, probe_channels=(1, 2), verbose=True,
                 **decider_kw):
        self.verbose = verbose
        self.source = source
        self.sample_freq = sample_freq

        # ---- classifier
        if source in ("sine", "synthetic"):
            self.model = SineProbeModel(freq_map=freq_map,
                                        sample_freq=sample_freq,
                                        channels=list(probe_channels))
            artifact_uv = 1e9          # a test sine is not an artifact
        else:
            from ML.local_model import LocalMIModel
            # user=None is legitimate: a first-time user has no personal model
            # and must fall back to the bundled starter. LocalMIModel.load()
            # already does personal -> starter -> FileNotFoundError, so the
            # only thing a missing name costs is the personal lookup.
            self.model = LocalMIModel.load(user or "_anonymous")
            artifact_uv = 100.0

        # ---- selector
        sel = SelectionTree if scheme == "tree" else ScanSelector
        self.selector = sel(buttons, on_select=on_select, on_change=on_change)

        self.decider = RealtimeDecider(
            self.model, self.selector,
            artifact=ArtifactDetector(sample_freq=sample_freq,
                                      threshold_uv=artifact_uv),
            **decider_kw)

        # ---- stream
        self.bridge = self._make_bridge(source, serial_port, recording,
                                        sample_freq, window, hop)
        if self.verbose:
            print(f"[LiveBCI] source={source} scheme={scheme} "
                  f"model={self.model}")

    def _make_bridge(self, source, serial_port, recording, fs, window, hop):
        from stream_bridge import StreamBridge, ReplayBackend, BrainFlowBackend
        if source in ("sine", "synthetic"):
            backend = BrainFlowBackend(board_id=-1, serial_port="",
                                       n_channels=8)
        elif source == "replay":
            backend = ReplayBackend(recording, loop=True, hop=hop)
        else:
            backend = BrainFlowBackend(board_id=0, serial_port=serial_port,
                                       n_channels=8)
        return StreamBridge(backend=backend, window=window, hop=hop)

    def start(self):
        self.bridge.start()
        return self

    def stop(self):
        try:
            self.bridge.stop()
        except Exception:
            pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def poll(self):
        """Call once per frame. Returns a result dict, or None if no window
        is ready yet. Never blocks, so the frame rate is unaffected."""
        win = self.bridge.poll()
        if win is None:
            return None
        r = self.decider.step(win)
        if self.verbose and r["state"] in ("DISPATCHED", "BLINK"):
            d = getattr(self.model, "last", None) or {}
            extra = f" {d['freq']:.2f} Hz" if d.get("freq") else ""
            print(f"  [{r['state']}]{extra} -> {r['command']} "
                  f"conf={r['confidence']:.2f} "
                  f"candidates={r['candidates']}")
        return r

    def report(self):
        return self.decider.report()
