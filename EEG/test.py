import matplotlib.pyplot as plt
import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations

FS = 250
seconds = 30  # Longer data helps examine 0.3 Hz
t = np.arange(FS * seconds) / FS

# Desired EEG-ish components: should mostly remain
alpha = 1.0 * np.sin(2 * np.pi * 10 * t)
beta = 0.5 * np.sin(2 * np.pi * 20 * t)

# Components which should be attenuated
slow_drift = 2.0 * np.sin(2 * np.pi * 0.3 * t)
high_freq_noise = 0.7 * np.sin(2 * np.pi * 60 * t)

raw = alpha + beta + slow_drift + high_freq_noise
filtered = raw.copy()

# This modifies `filtered` in place
DataFilter.perform_bandpass(
    filtered,
    FS,
    1,  # centre frequency
    40.0,  # bandwidth: approximately 1–40 Hz
    5,
    FilterTypes.BESSEL_ZERO_PHASE,
    0,
)

# Longer nfft resolves lower frequencies:
# 250 / 2048 = about 0.12 Hz per frequency bin
nfft = 2048
overlap = 1024

psd_raw, freq_raw = DataFilter.get_psd_welch(
    raw,
    nfft,
    overlap,
    FS,
    WindowOperations.BLACKMAN_HARRIS,
)

psd_filtered, freq_filtered = DataFilter.get_psd_welch(
    filtered,
    nfft,
    overlap,
    FS,
    WindowOperations.BLACKMAN_HARRIS,
)

plt.figure(figsize=(13, 8))

# Time-domain comparison
plt.subplot(2, 1, 1)
plot_s = 4
plt.plot(t[:FS*plot_s], raw[:FS*plot_s], label="Raw", linewidth=1)
plt.plot(t[:FS*plot_s], filtered[:FS*plot_s], label="Filtered", linewidth=1)
plt.title("First 4 seconds of signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Frequency-domain comparison
plt.subplot(2, 1, 2)
plt.semilogy(freq_raw, psd_raw, label="Raw")
plt.semilogy(freq_filtered, psd_filtered, label="Filtered")
plt.xticks(np.arange(0,101,10))
plt.xlim(0,100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("Power spectral density before and after filtering")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()