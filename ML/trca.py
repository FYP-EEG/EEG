"""
Author: Anson
Created on: 23/8/2026
Purpose: Step 6 -- Task-Related Component Analysis (TRCA) + Ensemble TRCA for SSVEP,
         with an FBCCA fallback and leakage-safe evaluation.

WHY TRCA
--------
FBCCA (Step 2) is *training-free*: it correlates the EEG against synthetic sine/cosine
references. That is robust but throws away everything specific to the user -- their
head geometry, electrode placement, and the phase of their evoked response.

TRCA instead learns, from the subject's own calibration data, the spatial filter w that
MAXIMISES REPRODUCIBILITY across repetitions of the same stimulus:

    maximise   w' S w / (w' Q w)

    S = sum of BETWEEN-TRIAL covariances (signal that repeats every trial)
    Q = covariance of all concatenated trials (total variance)

Solving the generalised eigenvalue problem S w = lambda Q w gives filters that amplify
the reproducible evoked response and suppress non-reproducible background (including
alpha, which is *not* phase-locked to the stimulus). Classification then correlates a
test trial against each class's TRCA-filtered template.

Filter-bank TRCA (FBTRCA) repeats this per sub-band and combines with the same
1/f-style weights used by FBCCA. Ensemble TRCA concatenates all classes' filters.

REQUIREMENTS AND HONESTY
------------------------
TRCA needs subject-specific training data -- which is exactly why this step had to wait
for Step 3. It is NOT a drop-in improvement:
  * with too few trials the covariance estimate is unstable and TRCA underperforms FBCCA
  * it is sensitive to electrode shift, so it should be recalibrated per session
`HybridTRCAClassifier` therefore keeps FBCCA available and can fall back automatically.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

from ML.hybrid_classifier import HybridSSVEPClassifier, NO_ACTION


# ------------------------------------------------------------------ core TRCA
def trca_weights(trials, reg=0.05):
    """Compute the leading TRCA spatial filter.

    :param trials: (n_trials, n_channels, n_times) -- repetitions of ONE stimulus
    :param reg:    shrinkage toward the identity; essential with few trials
    :return: (n_channels,) unit-norm spatial filter
    """
    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim != 3:
        raise ValueError(f"expected (trials, channels, times), got {trials.shape}")
    n_trials, n_ch, n_t = trials.shape
    if n_trials < 2:
        raise ValueError("TRCA needs at least 2 trials of the same stimulus")

    # centre each trial per channel
    X = trials - trials.mean(axis=2, keepdims=True)

    # S: sum of cross-trial covariances (reproducible component)
    S = np.zeros((n_ch, n_ch))
    for i in range(n_trials):
        for j in range(n_trials):
            if i == j:
                continue
            S += X[i] @ X[j].T
    # UX: all trials concatenated in time -> total covariance
    UX = X.transpose(1, 0, 2).reshape(n_ch, -1)
    Q = UX @ UX.T

    # shrinkage regularisation -- without this Q is often near-singular for 8 channels
    if reg > 0:
        Q = (1 - reg) * Q + reg * np.trace(Q) / n_ch * np.eye(n_ch)

    try:
        vals, vecs = eigh(S, Q)
    except np.linalg.LinAlgError:
        vals, vecs = eigh(S, Q + 1e-6 * np.eye(n_ch) * np.trace(Q) / n_ch)

    w = vecs[:, np.argmax(vals)]
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def corr2(a, b):
    """Pearson correlation between two 1-D signals, NaN-safe."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da < 1e-12 or db < 1e-12:
        return 0.0
    r = float(a @ b / (da * db))
    return 0.0 if np.isnan(r) else r


# --------------------------------------------------------------- classifier
class TRCAClassifier:
    """Filter-bank (ensemble) TRCA. Requires per-subject training data."""

    def __init__(self, target_freqs=(15.0, 20.0), sample_freq=250,
                 sub_band_starts=(12.0, 22.0, 32.0), highcut=88.0,
                 band_weights=(1.0, 0.65, 0.45), channels=None,
                 ensemble=True, reg=0.05, confidence_threshold=0.20):
        self.target_freqs = list(target_freqs)
        self.sample_freq = sample_freq
        self.sub_band_starts = list(sub_band_starts)
        self.highcut = highcut
        self.band_weights = list(band_weights)
        self.channels = list(channels) if channels is not None else None
        self.ensemble = ensemble
        self.reg = reg
        self.confidence_threshold = confidence_threshold

        self._band_sos = [butter(4, [lo, highcut], btype='band',
                                 fs=sample_freq, output='sos')
                          for lo in self.sub_band_starts]
        self.templates_ = None      # [band][class] -> (n_ch, n_times)
        self.filters_ = None        # [band][class] -> (n_ch,)
        self.is_fitted = False
        self.n_train_ = 0

    # ------------------------------------------------------------- internals
    def _sel(self, X):
        return X if self.channels is None else X[..., self.channels, :]

    def _band(self, X, b):
        return sosfiltfilt(self._band_sos[b], X, axis=-1)

    # -------------------------------------------------------------- fitting
    def fit(self, X, y):
        """
        :param X: (n_trials, n_channels, n_times)
        :param y: (n_trials,) class indices into target_freqs
        """
        X = self._sel(np.asarray(X, dtype=np.float64))
        y = np.asarray(y)
        classes = list(range(len(self.target_freqs)))

        for c in classes:
            if np.sum(y == c) < 2:
                raise ValueError(
                    f"class {c} has {int(np.sum(y == c))} trials; TRCA needs >= 2. "
                    f"Record more calibration data for this target.")

        self.templates_, self.filters_ = [], []
        for b in range(len(self._band_sos)):
            Xb = self._band(X, b)
            t_b, f_b = [], []
            for c in classes:
                Xc = Xb[y == c]
                t_b.append(Xc.mean(axis=0))              # class template
                f_b.append(trca_weights(Xc, reg=self.reg))
            self.templates_.append(t_b)
            self.filters_.append(f_b)

        self.is_fitted = True
        self.n_train_ = len(y)
        return self

    # ------------------------------------------------------------- scoring
    def scores(self, trial):
        """Weighted filter-bank correlation score per class."""
        if not self.is_fitted:
            raise RuntimeError("TRCAClassifier is not fitted")
        trial = self._sel(np.asarray(trial, dtype=np.float64))
        n_classes = len(self.target_freqs)
        out = np.zeros(n_classes)

        for b, w_band in enumerate(self.band_weights):
            Xb = self._band(trial, b)
            for c in range(n_classes):
                if self.ensemble:
                    # ensemble TRCA: project with ALL classes' filters, then correlate
                    W = np.stack(self.filters_[b], axis=1)     # (n_ch, n_classes)
                    r = corr2((W.T @ Xb).ravel(),
                              (W.T @ self.templates_[b][c]).ravel())
                else:
                    w = self.filters_[b][c]
                    r = corr2(w @ Xb, w @ self.templates_[b][c])
                out[c] += w_band * np.sign(r) * (r ** 2)
        return out

    def predict_proba(self, trial):
        """:return: (label|NO_ACTION, confidence, scores)"""
        s = self.scores(trial)
        best = int(np.argmax(s))
        conf = float(s[best])
        if conf < self.confidence_threshold:
            return NO_ACTION, conf, s
        return best, conf, s

    def predict(self, trial):
        return self.predict_proba(trial)[0]

    # -------------------------------------------------------------- persistence
    def save(self, path):
        """Save a fitted model so it can be shipped to another machine.

        Without this, a model trained in Colab on public data (or on your own
        recording) could never be used by the runtime -- which is the whole point
        of training it. Stores the learned filters/templates plus every
        hyper-parameter needed to reproduce the transform.
        """
        import joblib
        if not self.is_fitted:
            raise RuntimeError("refusing to save an unfitted TRCAClassifier")
        joblib.dump({
            "kind": "TRCAClassifier", "version": 1,
            "target_freqs": self.target_freqs, "sample_freq": self.sample_freq,
            "sub_band_starts": self.sub_band_starts, "highcut": self.highcut,
            "band_weights": self.band_weights, "channels": self.channels,
            "ensemble": self.ensemble, "reg": self.reg,
            "confidence_threshold": self.confidence_threshold,
            "templates": self.templates_, "filters": self.filters_,
            "n_train": self.n_train_,
        }, path)
        return path

    @classmethod
    def load(cls, path):
        """Load a model saved by save()."""
        import joblib
        d = joblib.load(path)
        if d.get("kind") != "TRCAClassifier":
            raise ValueError(f"{path} is not a TRCAClassifier model file")
        obj = cls(target_freqs=d["target_freqs"], sample_freq=d["sample_freq"],
                  sub_band_starts=d["sub_band_starts"], highcut=d["highcut"],
                  band_weights=d["band_weights"], channels=d["channels"],
                  ensemble=d["ensemble"], reg=d["reg"],
                  confidence_threshold=d["confidence_threshold"])
        obj.templates_ = d["templates"]
        obj.filters_ = d["filters"]
        obj.n_train_ = d.get("n_train", 0)
        obj.is_fitted = True
        return obj

    def calibrate_threshold(self, idle_trials, percentile=95.0):
        """Fit the NO_ACTION gate from the subject's own idle windows."""
        peaks = [float(np.max(self.scores(w))) for w in idle_trials]
        if not peaks:
            raise ValueError("no idle trials supplied")
        self.confidence_threshold = float(np.percentile(peaks, percentile))
        return self.confidence_threshold


# ------------------------------------------------------- hybrid front-end
class HybridTRCAClassifier:
    """TRCA when trained, FBCCA otherwise -- same API as HybridSSVEPClassifier.

    Lets the engine use the better method per subject without any call-site change,
    and degrades safely when no calibration exists.
    """

    def __init__(self, target_freqs=(15.0, 20.0), sample_freq=250,
                 montage="cyton8_ssvep", artifact_threshold=100.0,
                 confidence_threshold=0.15, prefer="auto", **kw):
        self.fbcca = HybridSSVEPClassifier(
            target_freqs=target_freqs, sample_freq=sample_freq, montage=montage,
            artifact_threshold=artifact_threshold,
            confidence_threshold=confidence_threshold, **kw)
        self.trca = TRCAClassifier(
            target_freqs=target_freqs, sample_freq=sample_freq,
            channels=self.fbcca.ssvep_channels)
        self.prefer = prefer            # 'auto' | 'trca' | 'fbcca'
        self.target_freqs = list(target_freqs)

    # artifact handling is shared -- always raw-band (Step 3 fix)
    def artifact_detection(self, trial, is_raw=True):
        return self.fbcca.artifact_detection(trial, is_raw=is_raw)

    def apply_filter(self, data, sos=None):
        return self.fbcca.apply_filter(data, sos)

    @property
    def active(self):
        if self.prefer == "fbcca":
            return "fbcca"
        if self.prefer == "trca":
            return "trca"
        return "trca" if self.trca.is_fitted else "fbcca"

    def fit(self, X, y, idle=None, percentile=95.0):
        self.trca.fit(X, y)
        if idle is not None and len(idle):
            self.trca.calibrate_threshold(idle, percentile=percentile)
        return self

    def save(self, path):
        """Save the fitted TRCA half of this hybrid."""
        return self.trca.save(path)

    def load_trca(self, path):
        """Load a pre-trained TRCA model; switches `active` to 'trca'."""
        self.trca = TRCAClassifier.load(path)
        return self

    def predict_proba(self, raw_trial, auto_filter=True):
        if self.artifact_detection(raw_trial, is_raw=auto_filter):
            return "BLINK", 1.0, np.zeros(len(self.target_freqs))
        if self.active == "trca":
            data = self.apply_filter(raw_trial) if auto_filter else raw_trial
            return self.trca.predict_proba(data)
        return self.fbcca.predict_proba(raw_trial, auto_filter=auto_filter)

    def predict(self, raw_trial, auto_filter=True):
        label, _, _ = self.predict_proba(raw_trial, auto_filter=auto_filter)
        if label == "BLINK":
            return 2
        return label
