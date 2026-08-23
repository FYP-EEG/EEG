"""
Author: Brian
Created on: 16/8/2026
Purpose: Synthetic SSVEP stream so the speller can be demoed / tested without hardware.

This is a stand-in for the real acquisition->engine bridge (a genuinely missing piece,
see docs/instructions_verification.md item B5). It emits (channels, n_times) windows
shaped exactly like what BCIEngine.process_frame() expects, so the same code path is
exercised in sim mode as with a live BrainFlow board.
"""

import numpy as np


class SyntheticSSVEPStream:
    """Emits sliding windows containing a chosen SSVEP frequency plus noise."""

    def __init__(self, n_channels=8, sample_freq=250, window=750,
                 target_freqs=(15.0, 20.0), occipital=(6, 7),
                 snr=1.6, seed=0, idle=False):
        self.n_channels = n_channels
        self.sample_freq = sample_freq
        self.window = window
        self.target_freqs = list(target_freqs)
        self.occipital = list(occipital)
        self.snr = snr
        self.rng = np.random.default_rng(seed)
        self.active = 0          # index into target_freqs; set by the demo/UI
        self.idle = idle         # when True, emit alpha-only windows (no target)
        self.t0 = 0.0

    def set_idle(self, flag=True):
        self.idle = bool(flag)

    def set_active(self, class_idx):
        """Pretend the user is now gazing at target `class_idx`."""
        self.active = int(class_idx) % len(self.target_freqs)

    def next_window(self):
        n = self.window
        t = self.t0 + np.arange(n) / self.sample_freq
        self.t0 += n / self.sample_freq

        data = self.rng.standard_normal((self.n_channels, n)) * 2.0
        # background alpha everywhere -- this is what causes the real 10 Hz bias
        data += 0.8 * np.sin(2 * np.pi * 10.0 * t)[None, :]
        for ch in self.occipital:
            if ch < self.n_channels:
                data[ch] += 4.0 * np.sin(2 * np.pi * self.rng.uniform(9.5, 10.5) * t
                                         + self.rng.uniform(0, 2 * np.pi))

        if self.idle:
            return data

        freq = self.target_freqs[self.active]
        for ch in self.occipital:
            if ch < self.n_channels:
                for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25)):
                    data[ch] += self.snr * amp * np.sin(2 * np.pi * h * freq * t)
        return data

    def inject_blink(self, data, amplitude=180.0):
        """Add a frontal high-voltage spike (Fp1/Fp2) to test artifact rejection."""
        mid = data.shape[1] // 2
        data[0, mid:mid + 20] += amplitude
        data[1, mid:mid + 20] += amplitude
        return data
