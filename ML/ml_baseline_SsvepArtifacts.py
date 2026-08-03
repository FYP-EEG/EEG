"""
Author: Brian
Created on: 20/7/2026
Purpose: Hybrid SSVEP + Artifacts Pipeline Baseline
Ideology: Test both CCA frequency matching and time-domain peak detection using simulated data
Edited on: 
1. 03/8/2026 (Adding Bandpass Filter)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import butter, sosfiltfilt

print("Initializing simulated 8-channel EEG Data")
# experiment data set
n_trials = 90    
n_channels = 8       
sample_freq = 250          
trial_duration = 3   
n_times = sample_freq * trial_duration 

# Channel 0, 1 (Fp1, Fp2) -> Forehead, responsible for capturing blinks (Artifacts)
# Channel 6, 7 (O1, O2) -> Back of the head, responsible for capturing visual resonance (SSVEP)
target_frequencies = [10.0, 12.0] # 10Hz and 12Hz SSVEP targets

# 0: Looking at 10Hz button | 1: Looking at 12Hz button | 2: Blinking (Artifact)
y_true = np.array([0, 1, 2] * (n_trials // 3))

# random noise
X = np.random.randn(n_trials, n_channels, n_times) * 2.0
time_vector = np.linspace(0, trial_duration, n_times)

# Feature Injection
for i in range(n_trials):
    if y_true[i] in [0, 1]:
        freq = target_frequencies[y_true[i]]
        X[i, 6, :] += np.sin(2 * np.pi * freq * time_vector) * 1.8
        X[i, 7, :] += np.sin(2 * np.pi * freq * time_vector) * 1.8
    elif y_true[i] == 2:
        X[i, 0, 240:260] += 150.0 
        X[i, 1, 240:260] += 150.0

print(f"Simulated Data Shape: {X.shape} (Trials, Channels, Time_points)")

#2. Algorithm Components
#fitering
def create_bandpass_filter(lowcut=1.0, highcut=45.0, fs=250, order=4):
    """butter SOS bandpass filter"""
    return butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')

def apply_filter(data, sos):
    """2D EEG (Channels, Time_points) """
    return sosfiltfilt(sos, data, axis=-1)


#SSVEP Detection using Canonical Correlation Analysis (CCA)
def cca_ssvep_detection(trial_data, target_freqs, sample_freq):
    eeg_segment = trial_data[[6,7], :].T
    n_times = eeg_segment.shape[0]
    r_max = -1
    detected_class = 0

    t = np.linspace(0, n_times / sample_freq, n_times, endpoint=False)

    for class_idx, freq in enumerate(target_freqs):
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
def artifact_detection(trial_data, threshold=100.0):
    peak_ch0 = np.max(np.abs(trial_data[0, :]))
    peak_ch1 = np.max(np.abs(trial_data[1, :]))
    return (peak_ch0 > threshold) or (peak_ch1 > threshold)

#3. Hybird Decoing 
print("\nRunning Hybrid Decoding Pipeline (Non-blocking Engine simulation)...")
y_pred = []
sos_filter = create_bandpass_filter(lowcut=1.0, highcut=45.0, fs=sample_freq)

for i in range(n_trials):
    raw_trial_data = X[i, :, :]
    filtered_trial_data = apply_filter(raw_trial_data, sos_filter)

    if artifact_detection(filtered_trial_data, threshold=100.0):
        y_pred.append(2)
    else:
        class_idx = cca_ssvep_detection(filtered_trial_data, target_frequencies, sample_freq)
        y_pred.append(class_idx)

y_pred = np.array(y_pred)

#4. Report Generation and Visualization
print("\n" + "="*40)
print("    Hybrid Pipeline Baseline Report   ")
print("="*40)
print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.2%}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['10Hz Button', '12Hz Button', 'Blink Trigger']))
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Simulated EEG Trial Signals (Hybrid Visualization)", fontsize=14, fontweight='bold')

idx_10hz = np.where(y_true == 0)[0][0]
idx_12hz = np.where(y_true == 1)[0][0]
idx_blink = np.where(y_true == 2)[0][0]

axes[0].plot(time_vector, X[idx_10hz, 6, :], label="Ch6 (O1) - 10Hz Signal", color='tab:blue')
axes[0].set_title("Class 0: Looking at 10Hz Button (Occipital Resonance)")
axes[0].legend(loc="upper right")
axes[0].set_ylabel("Amplitude (uV)")


axes[1].plot(time_vector, X[idx_12hz, 7, :], label="Ch7 (O2) - 12Hz Signal", color='tab:green')
axes[1].set_title("Class 1: Looking at 12Hz Button (Occipital Resonance)")
axes[1].legend(loc="upper right")
axes[1].set_ylabel("Amplitude (uV)")

axes[2].plot(time_vector, X[idx_blink, 0, :], label="Ch0 (Fp1) - Blink Artifact", color='tab:red', linewidth=1.5)
axes[2].axhline(y=100.0, color='gray', linestyle='--', label="Artifact Threshold (100uV)")
axes[2].axhline(y=-100.0, color='gray', linestyle='--')
axes[2].set_title("Class 2: Blink Trigger (Forehead High-Voltage Peak)")
axes[2].legend(loc="upper right")
axes[2].set_ylabel("Amplitude (uV)")
axes[2].set_xlabel("Time (Seconds)")

plt.tight_layout()

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['10Hz', '12Hz', 'Blink'], 
            yticklabels=['10Hz', '12Hz', 'Blink'])
plt.title("Hybrid Pipeline Confusion Matrix", fontsize=12, fontweight='bold')
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

plt.show()
