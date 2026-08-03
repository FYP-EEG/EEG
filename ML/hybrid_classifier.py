"""
Author: Brian
Created on: 03/08/2026
Purpose: moduling the ML package to include the hybrid classifier
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.cross_decomposition import CCA

class HybridSSVEPClassifier:
    def __init__(self, target_freqs=[10.0, 12.0], sample_freq=250, artifact_threshold=100.0, lowcut=1.0, highcut=45.0, order=4):
        self.target_freqs = target_freqs
        self.sample_freq = sample_freq
        self.artifact_threshold = artifact_threshold
        self.sos = butter(order, [lowcut, highcut], btype='band', fs=sample_freq, output='sos')

    # Filtering
    def apply_filter(self, data, sos):
        """2D EEG (Channels, Time_points) """
        return sosfiltfilt(sos, data, axis=-1)

    # SSVEP Detection using Canonical Correlation Analysis (CCA)
    def cca_ssvep_detection(self, trial_data):
        eeg_segment = trial_data[[6, 7], :].T
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

    # Artifact Detection using Time-Domain Peak Detection
    def artifact_detection(self, trial_data):
        peak_ch0 = np.max(np.abs(trial_data[0, :]))
        peak_ch1 = np.max(np.abs(trial_data[1, :]))
        return (peak_ch0 > self.artifact_threshold) or (peak_ch1 > self.artifact_threshold)

    def predict(self, raw_trial_data, auto_filter=True):
        """
        API for predicting the class of a single trial of EEG data.
        :param raw_trial_data: shape (Channels, Time_points) 
        :param auto_filter: Whether to automatically apply the bandpass filter (default is True)
        :return: Predicted class index (0: 10Hz, 1: 12Hz, 2: Blink)
        """
        # 1. bypass filter
        trial_data = self.apply_filter(raw_trial_data) if auto_filter else raw_trial_data
        
        # 2. Blink Artifact
        if self.artifact_detection(trial_data):
            return 2  # Blink Trigger
        # 3. SSVEP Detection
        return self.cca_ssvep_detection(trial_data)