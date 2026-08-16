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

    def cca_ssvep_detection(self, trial_data, num_harmonics=4, num_bands=3):
        """Upgraded to Filter-Bank CCA (FBCCA) to eliminate 1/f biological noise"""
        n_times = trial_data.shape[1]
        t = np.linspace(0, n_times / self.sample_freq, n_times, endpoint=False)

        weights = [1.0, 0.65, 0.45] 
        class_scores = np.zeros(len(self.target_freqs))
        
        for class_idx, freq in enumerate(self.target_freqs):
            ref_signals = np.zeros((n_times, 2 * num_harmonics))
            for h in range(1, num_harmonics + 1):
                ref_signals[:, 2*(h-1)] = np.sin(2 * np.pi * h * freq * t)
                ref_signals[:, 2*(h-1)+1] = np.cos(2 * np.pi * h * freq * t)
            
            total_rho = 0
            
            for band in range(num_bands):
                low_f = 8.0 + (band * 10.0) 
                high_f = 88.0 
                
                sos = butter(4, [low_f, high_f], btype='band', fs=self.sample_freq, output='sos')
                filtered_data = sosfiltfilt(sos, trial_data, axis=-1)

                eeg_segment = filtered_data[self.ssvep_channels, :].T
                
                cca = CCA(n_components=1)
                cca.fit(eeg_segment, ref_signals)
                x_score, y_score = cca.transform(eeg_segment, ref_signals)
                

                rho = np.corrcoef(x_score.T, y_score.T)[0, 1]

                total_rho += (rho ** 2) * weights[band]
                
            class_scores[class_idx] = total_rho
            
        return np.argmax(class_scores)

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
        # 1. bypass filter
        trial_data = self.apply_filter(raw_trial_data, self.sos) if auto_filter else raw_trial_data
        
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