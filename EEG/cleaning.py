"""
Author: Anson Li
Created on: 26/6/2026
Purpose: script for cleaning EEG data
Location: project_dir/EEG/cleaning.py
"""
import pandas as pd
import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes

def clean(data, eeg_chann, sampling_rate, amplitude_threshold_uV=3.0, use_spectral_gating=False):
    for count, channel in enumerate(eeg_chann):
        # 1. Environmental noise removal (50Hz notch) & Bandpass (1-40Hz)
        DataFilter.remove_environmental_noise(data[channel], sampling_rate, NoiseTypes.FIFTY.value)
        DataFilter.perform_bandpass(data[channel], sampling_rate, 1, 40, 8, FilterTypes.BUTTERWORTH, 0)
        
        # 2. Amplitude Filtering (Spectral Gating in uV)
        # Note: Set use_spectral_gating=True only if isolating dominant peaks (e.g. SSVEP/Alpha)
        if use_spectral_gating and amplitude_threshold_uV > 0:
            N = len(data[channel])
            fft_complex = np.fft.rfft(data[channel])
            amplitudes = np.abs(fft_complex)
            
            # Convert unnormalized FFT magnitude to Peak Amplitude in microvolts (uV)
            amplitudes_uV = amplitudes * (2.0 / N)
            amplitudes_uV[0] = amplitudes[0] / N
            if N % 2 == 0:
                amplitudes_uV[-1] = amplitudes[-1] / N
            
            # Zero out frequencies below threshold in uV
            fft_complex[amplitudes_uV < amplitude_threshold_uV] = 0
            
            # Convert back to time domain
            data[channel] = np.fft.irfft(fft_complex, n=N)
        
    df = pd.DataFrame(np.transpose(data))
    return df