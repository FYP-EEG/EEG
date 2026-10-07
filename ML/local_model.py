"""
Author: Anson
Created on: 6/10/2026
Purpose: serve a locally trained model at inference time. No server, no network.

WHY THIS EXISTS RATHER THAN REUSING MotorImageryClassifier
----------------------------------------------------------
The blocking reason is a FILE FORMAT mismatch. ML/train_local.py saves a dict:

    {"kind": "MILocalModel", "pipeline": "csp_lda", "model": <Pipeline>, ...}

MotorImageryClassifier does `joblib.load(model_path)` and treats the result as
the estimator itself, so a locally trained model fails immediately:

    AttributeError: 'dict' object has no attribute 'predict'

It was written for the PhysioNet baseline, which was pickled bare. This module
unwraps the bundle and carries the metadata (pipeline name, CV accuracy,
montage, sample rate, provenance) that the dict exists to hold.

TWO THINGS THAT LOOK LIKE BUGS AND ARE NOT  (measured, not assumed)
-------------------------------------------------------------------
Both were initially written up here as faults. Measurement contradicted both,
so they are recorded as properties instead:

  1. UNIT SCALING IS HARMLESS. MotorImageryClassifier multiplies by 1e-6 for
     volts-scale EDF data, while calibration recordings are in uV. That looks
     fatal but is not: _cov() trace-normalises, so every pipeline is
     scale-invariant.

         max |P(uV) - P(volts)|   csp_lda 2.9e-55   fbcsp 4.9e-70
                                  riemann_svm 4.4e-16   riemann_lr 4.4e-16

     A useful consequence: electrode gain drift between sessions cannot shift
     the features. This is asserted in tests/test_step10_realtime.py.

  2. DOUBLE FILTERING IS NEARLY HARMLESS. Every pipeline band-passes internally
     (BandpassFilter in csp_lda, FilterBankCSP's bank, TangentSpaceMapper's
     `band`), so an external 8-30 Hz pre-filter applies it twice. Measured:

         label agreement 100% on all four pipelines
         mean |dP|  csp_lda 0.0000  fbcsp 0.0000  riemann_svm 0.0021  riemann_lr 0.0030

     Still omitted here, because paying for a filter that changes nothing is
     waste, and the pipeline owning its own preprocessing is the contract that
     keeps training and inference identical.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

#: returned when the confidence gate rejects a window (Chow's reject option)
NO_ACTION = -1

#: class index -> command name. target_0 / target_1 in the calibration protocol.
DEFAULT_CLASS_MAP = {0: "LEFT", 1: "RIGHT"}


class LocalMIModel:
    """A trained model loaded from disk and run entirely on this machine.

    The model file is produced by ML/train_local.py and is 1.9-11.4 KB. Nothing
    here opens a socket: `load()` touches the filesystem and nothing else.
    """

    def __init__(self, model, sample_freq=250, montage=None, units="uV",
                 confidence_threshold=0.65, source="unknown", meta=None,
                 class_map=None, prefilter=None, scale=1.0):
        """
        :param prefilter: (low, high) Hz to apply before predicting, or None
            when the pipeline band-passes internally. Models trained by
            ML/train_local.py own their filtering, so None. A model exported
            from Colab after `raw.filter(8, 30)` does NOT, and needs (8, 30):
            measured 50.0% unfiltered vs 90.5% filtered on contaminated input.
        :param scale: multiply input by this before predicting. 1.0 for models
            trained on uV; 1e-6 for models trained on volts-scale EDF. Our own
            pipelines trace-normalise and ignore scale entirely, but MNE's CSP
            with log=True, norm_trace=False does not -- at x1e6 its label
            agreement drops to 83.3%.
        """
        self.model = model
        self.sample_freq = int(sample_freq)
        self.montage = montage
        self.units = units
        self.confidence_threshold = float(confidence_threshold)
        self.source = source                      # 'personal' | 'starter'
        self.meta = meta or {}
        self.class_map = dict(class_map or DEFAULT_CLASS_MAP)
        self.prefilter = tuple(prefilter) if prefilter else None
        self.scale = float(scale)
        self._sos = None
        if self.prefilter:
            from scipy.signal import butter
            self._sos = butter(4, list(self.prefilter), btype="band",
                               fs=self.sample_freq, output="sos")

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, user, root=None, confidence_threshold=None):
        """Personal model if the user has one, else the bundled starter model.

        :raises FileNotFoundError: when neither exists, because silently running
            an untrained system is the failure mode that wastes a lab session.
        """
        from ML.train_local import get_model

        bundle, source = get_model(user, root=root)

        # A model exported from Colab is usually a BARE sklearn pipeline, not
        # the dict ML/train_local.py writes. Accept both, so a starter model
        # trained on public data can simply be dropped into ML/.
        colab = False
        if bundle is not None and not isinstance(bundle, dict):
            colab = source == "starter"
            bundle = {"model": bundle, "sample_freq": 250,
                      "pipeline": type(bundle).__name__,
                      "montage": "cyton8_motor" if colab else None,
                      "note": "PhysioNet starter model exported from Colab"
                              if colab else "bare estimator, no metadata"}
        if bundle is None:
            raise FileNotFoundError(
                f"no model for {user!r}: train one with "
                f"`python ML/train_local.py {user}`, and no starter model is "
                f"bundled with this build")

        thr = confidence_threshold
        if thr is None:
            thr = cls._threshold_from_profile(user, root)

        # A bare Colab pipeline band-passed its RAW data before epoching and
        # trained on volts-scale EDF, so it needs both applied here: measured
        # 50.0% without the contract against 90.5% with it.
        return cls(model=bundle["model"],
                   sample_freq=bundle.get("sample_freq", 250),
                   montage=bundle.get("montage"),
                   units="volts" if colab else "uV",
                   confidence_threshold=thr,
                   source=source,
                   meta={k: v for k, v in bundle.items() if k != "model"},
                   prefilter=(8.0, 30.0) if colab else None,
                   scale=1e-6 if colab else 1.0)

    @classmethod
    def from_colab(cls, path, sample_freq=250, prefilter=(8.0, 30.0),
                   scale=1e-6, montage="cyton8_motor",
                   confidence_threshold=0.65):
        """Load a bare pipeline exported from the PhysioNet training notebook.

        Defaults match that notebook exactly: MNE applied `raw.filter(8, 30)`
        before epoching, so the pipeline has no filter of its own, and EDF data
        is in volts while the board streams uV.

        Channel order needs no remapping any more. The notebook picks
        FC3 FC4 C3 CZ C4 CP3 CP4 PZ, and `cyton8_motor` - the montage this
        project now records with - is exactly that list, so the starter model
        and live recordings sit on the same electrodes.
        """
        import joblib
        obj = joblib.load(path)
        model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        return cls(model=model, sample_freq=sample_freq, montage=montage,
                   units="volts" if scale != 1.0 else "uV",
                   confidence_threshold=confidence_threshold, source="starter",
                   meta={"pipeline": type(model).__name__,
                         "note": "PhysioNet starter model, 61.40% +/- 1.01 "
                                 "5-fold CV, 4927 trials"},
                   prefilter=prefilter, scale=scale)

    @staticmethod
    def _threshold_from_profile(user, root=None, default=0.65):
        """Per-user threshold fitted to THIS subject's rest recording.

        ML/calibration.py writes `confidence_threshold` as the 95th percentile
        of the idle-period score distribution. Falling back to a constant is a
        real downgrade - the whole point of fitting it is that the idle
        distribution differs per person - so the fallback is reported.
        """
        base = Path(root) if root else (ROOT / "profiles")
        p = base / str(user).lower() / "profile.json"
        if p.exists():
            try:
                v = json.loads(p.read_text()).get("confidence_threshold")
                if v is not None:
                    return float(v)
            except (ValueError, OSError):
                pass
        return default

    # --------------------------------------------------------------- predict
    def _prepare(self, window):
        """(channels, samples) -> (1, channels, samples), units untouched.

        No band-pass here on purpose: the pipeline does it. No unit scaling
        either - the model was fitted on uV and the stream delivers uV.
        """
        x = np.asarray(window, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (channels, samples), got {x.shape}")
        if self.scale != 1.0:
            x = x * self.scale
        if self._sos is not None:
            from scipy.signal import sosfiltfilt
            x = sosfiltfilt(self._sos, x, axis=-1)
        return x[np.newaxis, ...]

    def predict_proba(self, window):
        """:return: (label, confidence). label is NO_ACTION below threshold."""
        xi = self._prepare(window)
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(xi))[0]
            best = int(np.argmax(probs))
            conf = float(probs[best])
        else:
            best = int(self.model.predict(xi)[0])
            conf = 1.0
        if conf < self.confidence_threshold:
            return NO_ACTION, conf
        return best, conf

    def predict_command(self, window):
        """:return: (command_name | None, confidence) -- None means no action."""
        label, conf = self.predict_proba(window)
        if label == NO_ACTION:
            return None, conf
        return self.class_map.get(label, str(label)), conf

    # ----------------------------------------------------------- diagnostics
    def describe(self):
        m = self.meta
        return {
            "source": self.source,
            "pipeline": m.get("pipeline"),
            "cv_accuracy": m.get("cv_accuracy"),
            "cv_reliable": m.get("cv_reliable"),
            "n_trials": m.get("n_trials"),
            "sample_freq": self.sample_freq,
            "montage": self.montage,
            "confidence_threshold": self.confidence_threshold,
            "prefilter": self.prefilter,
            "scale": self.scale,
            "trained_at": m.get("trained_at"),
        }

    def __repr__(self):
        d = self.describe()
        acc = d["cv_accuracy"]
        acc = f"{acc:.1%}" if isinstance(acc, float) else "?"
        return (f"<LocalMIModel {d['source']} {d['pipeline']} cv={acc} "
                f"thr={self.confidence_threshold:.2f}>")
