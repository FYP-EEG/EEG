"""
Author: Brian
Created on: 16/8/2026
Purpose: Step 6 -- synthetic Motor Imagery data with REALISTIC difficulty.

A first version of this generator produced 97% cross-subject accuracy for plain
CSP+LDA -- wildly inconsistent with the 61.4% the same pipeline scored on real
PhysioNet data. The cause: it added a large, deterministic, phase-locked sinusoid,
making the classes trivially separable.

Real motor imagery is an event-related DESYNCHRONISATION: a modest, noisy REDUCTION
in mu/beta BAND POWER over the contralateral sensorimotor cortex. It is:
  * a power change, not an added phase-locked oscillation (phases are random)
  * modest in size (~20-40% power change), with large trial-to-trial variance
  * subject-specific in both peak frequency and spatial pattern
  * accompanied by strong non-discriminative background rhythms

This version models all of that. It is calibrated so plain CSP+LDA lands near the
low-60s cross-subject, matching the real result -- which is what makes it usable
for RANKING pipelines.

Caveat: synthetic data can validate that a pipeline is implemented correctly and
behaves sensibly, but it cannot definitively rank methods for real EEG. Treat the
ordering here as a hypothesis to confirm on your own recordings.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def _bandlimited_noise(n_ch, n_times, fs, band, rng):
    """Random-phase oscillatory activity in a band (no fixed phase across trials)."""
    x = rng.standard_normal((n_ch, n_times))
    sos = butter(4, list(band), btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=-1)


def make_mi_dataset(n_subjects=8, trials_per_subject=60, n_channels=8,
                    n_times=750, fs=250, seed=0, erd_strength=0.35,
                    trial_jitter=0.30):
    """
    :param erd_strength: fractional band-power drop contralaterally (0.35 = -35%)
    :param trial_jitter: sd of per-trial multiplicative variation in that drop
    """
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []

    for s in range(n_subjects):
        # --- subject-specific traits
        mu = rng.uniform(9.0, 13.0)
        beta = rng.uniform(18.0, 26.0)
        mu_band = (mu - 2.0, mu + 2.0)
        beta_band = (beta - 3.0, beta + 3.0)
        # subject-specific volume conduction / electrode placement
        mix = np.eye(n_channels) + 0.45 * rng.standard_normal((n_channels, n_channels))
        # which channels carry the sensorimotor sources for THIS subject
        c_left, c_right = 2, 4
        subj_erd = erd_strength * rng.uniform(0.6, 1.4)     # responder variability

        for t in range(trials_per_subject):
            cls = t % 2                                     # 0 = left hand, 1 = right

            # background: broadband + strong non-discriminative alpha everywhere
            sig = rng.standard_normal((n_channels, n_times)) * 1.0
            sig += 1.4 * _bandlimited_noise(n_channels, n_times, fs, (8.0, 13.0), rng)

            # sensorimotor rhythms: random phase each trial (a POWER effect)
            mu_src = _bandlimited_noise(2, n_times, fs, mu_band, rng)
            beta_src = _bandlimited_noise(2, n_times, fs, beta_band, rng)

            # ERD is CONTRALATERAL: left-hand imagery desynchronises the right hemisphere
            drop = np.clip(subj_erd * rng.normal(1.0, trial_jitter), 0.0, 0.9)
            gain_l = (1.0 - drop) if cls == 1 else 1.0
            gain_r = (1.0 - drop) if cls == 0 else 1.0

            sig[c_left] += 1.5 * gain_l * mu_src[0] + 0.7 * gain_l * beta_src[0]
            sig[c_right] += 1.5 * gain_r * mu_src[1] + 0.7 * gain_r * beta_src[1]

            X.append(mix @ sig)
            y.append(cls)
            g.append(s)

    return np.array(X), np.array(y), np.array(g)
