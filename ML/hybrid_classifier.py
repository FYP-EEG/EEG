"""
Author: Brian
Created on: 10/08/2026
Purpose: Unified BCI Machine Learning Classifiers Module (SSVEP + Motor Imagery)

Edited 16/08/2026:
  1. Targets default to 15/20 Hz -- 10 Hz collided with the occipital alpha band and
     caused the 10 Hz prediction bias seen at 73.10% in the Colab run. Both new
     targets are exact 60 Hz monitor divisors (4 and 3 frames) for jitter-free flicker.
  2. Sub-bands now start ABOVE alpha (12/22/32 Hz) instead of 8/18/28 Hz.
  3. predict_proba() returns (label, confidence, scores) so the engine can reject
     low-confidence windows. argmax alone can never output "no command".
  4. NO_ACTION class added -- the single most important change for a GUI that must
     tolerate a distracted or resting user.
  5. Channel indices come from ML.montage, not hard-coded 64-ch integers.
  6. MotorImageryClassifier now applies the 8-30 Hz filter and unit scaling that the
     training pipeline used, and exposes predict_proba.
"""

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.cross_decomposition import CCA

from ML import montage as montage_mod

#: returned when no target is confidently detected
NO_ACTION = -1


class HybridSSVEPClassifier:
    def __init__(self, target_freqs=(15.0, 20.0), sample_freq=250,
                 artifact_threshold=100.0, lowcut=6.0, highcut=88.0, order=4,
                 montage="cyton8_ssvep", ssvep_channels=None, artifact_channels=None,
                 confidence_threshold=0.15, sub_band_starts=(12.0, 22.0, 32.0),
                 band_weights=(1.0, 0.65, 0.45), num_harmonics=4,
                 artifact_band=(0.5, 8.0)):
        """
        :param target_freqs: stimulus frequencies in Hz. Default 15/20 avoids the
                             8-13 Hz alpha band that biased the old 10/12 design.
        :param montage:      name in ML.montage; supplies channel indices.
        :param confidence_threshold: minimum FBCCA score to emit a command at all.
                             Below this the classifier returns NO_ACTION.
        """
        self.target_freqs = list(target_freqs)
        self.sample_freq = sample_freq
        self.artifact_threshold = artifact_threshold
        self.confidence_threshold = confidence_threshold
        self.sub_band_starts = list(sub_band_starts)
        self.band_weights = list(band_weights)
        self.num_harmonics = num_harmonics
        self.highcut = highcut
        self.montage_name = montage

        m = montage_mod.get(montage)
        self.ssvep_channels = list(ssvep_channels) if ssvep_channels is not None else list(m["ssvep"])
        self.artifact_channels = list(artifact_channels) if artifact_channels is not None else list(m["artifact"])

        self.sos = butter(order, [lowcut, highcut], btype='band',
                          fs=sample_freq, output='sos')
        # Artifact filter. Blinks are 0.5-3 Hz slow waves; the SSVEP highpass
        # (6 Hz) DELETES them, so detecting artifacts on SSVEP-filtered data can
        # never fire. Found by the Step 3 calibration run: 0/32 blink windows
        # were caught. Artifacts must be judged on their own low band.
        self.artifact_sos = butter(2, list(artifact_band), btype='band',
                                   fs=sample_freq, output='sos')

        # pre-build sub-band filters once instead of per-trial (was a real cost)
        self._band_sos = [
            butter(4, [lo, highcut], btype='band', fs=sample_freq, output='sos')
            for lo in self.sub_band_starts
        ]

    # ---------------------------------------------------------------- filtering
    def apply_filter(self, data, sos=None):
        """2D EEG (Channels, Time_points) Bandpass Filtering"""
        return sosfiltfilt(sos if sos is not None else self.sos, data, axis=-1)

    # ------------------------------------------------------------------- scores
    def _reference(self, freq, n_times):
        t = np.linspace(0, n_times / self.sample_freq, n_times, endpoint=False)
        ref = np.zeros((n_times, 2 * self.num_harmonics))
        for h in range(1, self.num_harmonics + 1):
            ref[:, 2 * (h - 1)] = np.sin(2 * np.pi * h * freq * t)
            ref[:, 2 * (h - 1) + 1] = np.cos(2 * np.pi * h * freq * t)
        return ref

    def fbcca_scores(self, trial_data):
        """Filter-Bank CCA. Returns the per-class score vector (NOT an argmax).

        Keeping the scores is what makes confidence thresholding and NO_ACTION
        possible; the previous version discarded them inside np.argmax.
        """
        n_times = trial_data.shape[1]
        scores = np.zeros(len(self.target_freqs))

        # filter each sub-band once, reuse across all classes
        banded = [sosfiltfilt(sos, trial_data, axis=-1)[self.ssvep_channels, :].T
                  for sos in self._band_sos]

        for ci, freq in enumerate(self.target_freqs):
            ref = self._reference(freq, n_times)
            total = 0.0
            for seg, w in zip(banded, self.band_weights):
                try:
                    cca = CCA(n_components=1)
                    cca.fit(seg, ref)
                    xs, ys = cca.transform(seg, ref)
                    rho = np.corrcoef(xs.T, ys.T)[0, 1]
                    if np.isnan(rho):
                        rho = 0.0
                except Exception:
                    rho = 0.0
                total += (rho ** 2) * w
            scores[ci] = total
        return scores

    def cca_ssvep_detection(self, trial_data, num_harmonics=None, num_bands=None):
        """Back-compat: returns argmax only. Prefer predict_proba()."""
        return int(np.argmax(self.fbcca_scores(trial_data)))

    # ---------------------------------------------------------------- artifacts
    def artifact_detection(self, trial_data, is_raw=True):
        """Peak detection on frontal channels (blinks / eye movement).

        :param is_raw: True if `trial_data` has NOT been through the SSVEP
                       bandpass. Blinks live at 0.5-3 Hz, below the 6 Hz SSVEP
                       highpass, so they must be measured on raw or low-band data.
        """
        chans = [c for c in self.artifact_channels if c < trial_data.shape[0]]
        if not chans:
            return False
        if is_raw:
            band = sosfiltfilt(self.artifact_sos, trial_data[chans, :], axis=-1)
        else:
            band = trial_data[chans, :]
        return bool(np.max(np.abs(band)) > self.artifact_threshold)

    # ------------------------------------------------------------------ predict
    def predict_proba(self, raw_trial_data, auto_filter=True):
        """
        Confidence-aware prediction.

        :return: (label, confidence, scores)
                 label = class index, or NO_ACTION (-1) when nothing is confident,
                         or 'BLINK' when a frontal artifact dominates.
                 confidence = winning FBCCA score
                 scores = full per-class score vector
        """
        montage_mod.validate(self.montage_name, raw_trial_data)

        # Artifact check runs on the RAW signal, before the SSVEP highpass.
        if self.artifact_detection(raw_trial_data, is_raw=auto_filter):
            return "BLINK", 1.0, np.zeros(len(self.target_freqs))

        data = self.apply_filter(raw_trial_data) if auto_filter else raw_trial_data
        scores = self.fbcca_scores(data)
        best = int(np.argmax(scores))
        conf = float(scores[best])

        if conf < self.confidence_threshold:
            return NO_ACTION, conf, scores
        return best, conf, scores

    def predict(self, raw_trial_data, auto_filter=True):
        """Back-compat wrapper. 0/1 = target, 2 = blink, -1 = NO_ACTION."""
        label, _, _ = self.predict_proba(raw_trial_data, auto_filter=auto_filter)
        if label == "BLINK":
            return 2
        return label

    # -------------------------------------------------------------- calibration
    def calibrate_threshold(self, idle_windows, percentile=95.0, auto_filter=True):
        """
        Set confidence_threshold from the user's own IDLE data.

        Feed windows recorded while the user rests / reads / looks away. The
        threshold is placed above `percentile` of idle scores, so idle windows
        fall through to NO_ACTION. This is the per-subject adaptation that public
        benchmark data cannot give you.
        """
        peaks = []
        for w in idle_windows:
            data = self.apply_filter(w) if auto_filter else w
            if self.artifact_detection(data):
                continue
            peaks.append(float(np.max(self.fbcca_scores(data))))
        if not peaks:
            raise ValueError("No usable idle windows (all rejected as artifacts).")
        self.confidence_threshold = float(np.percentile(peaks, percentile))
        return self.confidence_threshold


class MotorImageryClassifier:
    """
    NOTE: cross-subject CV accuracy is 61.4% (near chance). Recommended to keep MI
    OUT of the live demo path until per-subject calibration data exists.

    Fixes applied: the training pipeline filtered 8-30 Hz and worked in VOLTS
    (MNE EDF). BrainFlow delivers MICROVOLTS and the old predict() applied no
    filter at all, so live features were off by ~1e6 and out of band.
    """

    def __init__(self, model_path=None, sample_freq=250, input_units="uV",
                 bandpass=(8.0, 30.0), confidence_threshold=0.65):
        self.pipeline = joblib.load(model_path) if model_path else None
        self.sample_freq = sample_freq
        self.input_units = input_units
        self.confidence_threshold = confidence_threshold
        self.sos = butter(4, list(bandpass), btype='band',
                          fs=sample_freq, output='sos') if bandpass else None

    def _prepare(self, raw_trial_data, auto_filter=True):
        x = np.asarray(raw_trial_data, dtype=np.float64)
        if self.input_units == "uV":
            x = x * 1e-6                      # match the volts-scale training data
        if auto_filter and self.sos is not None:
            x = sosfiltfilt(self.sos, x, axis=-1)
        return np.expand_dims(x, axis=0)

    def predict_proba(self, raw_trial_data, auto_filter=True):
        """:return: (label, confidence) -- label may be NO_ACTION."""
        if self.pipeline is None:
            return NO_ACTION, 0.0
        xi = self._prepare(raw_trial_data, auto_filter)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(xi)[0]
            best = int(np.argmax(probs))
            conf = float(probs[best])
            if conf < self.confidence_threshold:
                return NO_ACTION, conf
            return best, conf
        return int(self.pipeline.predict(xi)[0]), 1.0

    def predict(self, raw_trial_data, auto_filter=True):
        label, _ = self.predict_proba(raw_trial_data, auto_filter)
        return label
