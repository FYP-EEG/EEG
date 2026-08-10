"""
Author: Brian
Created on: 10/08/2026
Purpose: Unified BCI Machine Learning Classifiers Module (SSVEP + Motor Imagery)
"""

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.cross_decomposition import CCA

class HybridSSVEPClassifier:
    def __init__(self, target_freqs=[10.0, 12.0], sample_freq=250, artifact_threshold=100.0, lowcut=1.0, highcut=45.0, order=4, ssvep_channels=[54, 55, 60, 61, 62], artifact_channels=[0, 1]):
        self.target_freqs = target_freqs
        self.sample_freq = sample_freq
        self.artifact_threshold = artifact_threshold
        self.ssvep_channels = ssvep_channels
        self.artifact_channels = artifact_channels
        self.sos = butter(order, [lowcut, highcut], btype='band', fs=sample_freq, output='sos')

    def apply_filter(self, data, sos=None):
        """2D EEG (Channels, Time_points) Bandpass Filtering"""
        if sos is None:
            sos = self.sos
        return sosfiltfilt(sos, data, axis=-1)

    def cca_ssvep_detection(self, trial_data):
        """SSVEP Detection using Canonical Correlation Analysis (CCA)"""
        # Extract only the relevant occipital channels for CCA
        eeg_segment = trial_data[self.ssvep_channels, :].T
        n_times = eeg_segment.shape[0]
        r_max = -1
        detected_class = 0

        t = np.linspace(0, n_times / self.sample_freq, n_times, endpoint=False)

        for class_idx, freq in enumerate(self.target_freqs):
            ref_signals = np.zeros((n_times, 4))
            ref_signals[:, 0] = np.sin(2 * np.pi * freq * t)
            ref_signals[:, 1] = np.cos(2 * np.pi * freq * t)
            ref_signals[:, 2] = np.sin(2 * np.pi * 2 * freq * t)
            ref_signals[:, 3] = np.cos(2 * np.pi * 2 * freq * t)

            cca = CCA(n_components=1)
            cca.fit(eeg_segment, ref_signals)
            x_score, y_score = cca.transform(eeg_segment, ref_signals)

            r = np.corrcoef(x_score.T, y_score.T)[0, 1]
            if r > r_max:
                r_max = r
                detected_class = class_idx

        return detected_class

    def artifact_detection(self, trial_data):
        """Artifact Detection using Time-Domain Peak Detection"""
        for ch in self.artifact_channels:
            peak = np.max(np.abs(trial_data[ch, :]))
            if peak > self.artifact_threshold:
                return True
        return False

    def predict(self, raw_trial_data, auto_filter=True):
        """
        API for predicting the class of a single trial of EEG data.
        :param raw_trial_data: shape (Channels, Time_points) 
        :param auto_filter: Whether to automatically apply the bandpass filter (default is True)
        :return: Predicted class index (0: 10Hz, 1: 12Hz, 2: Blink)
        """
        trial_data = self.apply_filter(raw_trial_data) if auto_filter else raw_trial_data
        
        if self.artifact_detection(trial_data):
            return 2  

        return self.cca_ssvep_detection(trial_data)


class MotorImageryClassifier:
    def __init__(self, model_path=None):
        if model_path:
            self.pipeline = joblib.load(model_path)
        else:
            self.pipeline = None

    def predict(self, raw_trial_data, auto_filter=True):
        """
        API for single trial Motor Imagery inference.
        """
        x_input = np.expand_dims(raw_trial_data, axis=0) 
        
        if self.pipeline:
            pred = self.pipeline.predict(x_input)[0]
            return pred
        return 0