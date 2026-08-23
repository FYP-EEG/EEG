"""
Author: Brian
Created on: 16/8/2026
Purpose: Single source of truth for electrode -> array-index mapping.

Fixes blocker B1: the Colab evaluation indexes a 64-channel Tsinghua Benchmark cap
(ssvep_channels=[53..63]) while the live rig is an 8-channel Cyton. Those indices
raise IndexError on live data. Keeping the maps in one place means the classifier
is configured by MONTAGE NAME, not by hard-coded integers.
"""

# --- Tsinghua Benchmark 64-channel cap (offline evaluation only) --------------
BENCHMARK_64 = {
    "name": "benchmark64",
    "n_channels": 64,
    # PO5 PO3 POz PO4 PO6 O1 Oz O2 CB1 -- full parieto-occipital cluster
    "ssvep": [53, 54, 55, 56, 57, 60, 61, 62, 63],
    "artifact": [0, 1],          # frontal, for blink detection
    "motor": [23, 24, 25, 26, 27, 28, 29, 30],
}

# --- OpenBCI Cyton, 8 channels ------------------------------------------------
# SSVEP-oriented placement. Wire the headset to match this or edit here.
CYTON_8_SSVEP = {
    "name": "cyton8_ssvep",
    "n_channels": 8,
    "labels": ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"],
    "ssvep": [6, 7],             # O1, O2
    "artifact": [0, 1],          # Fp1, Fp2
    "motor": [2, 3],             # C3, C4
}

# Montage used by the MI Colab notebook (PhysioNet 8-ch pick)
CYTON_8_MOTOR = {
    "name": "cyton8_motor",
    "n_channels": 8,
    "labels": ["FC3", "FC4", "C3", "CZ", "C4", "CP3", "CP4", "PZ"],
    "ssvep": [7],                # PZ -- poor for SSVEP, MI montage is not for SSVEP
    "artifact": [0, 1],
    "motor": [0, 1, 2, 3, 4, 5, 6, 7],
}

MONTAGES = {m["name"]: m for m in (BENCHMARK_64, CYTON_8_SSVEP, CYTON_8_MOTOR)}


def get(name):
    if name not in MONTAGES:
        raise KeyError(f"Unknown montage {name!r}. Known: {sorted(MONTAGES)}")
    return MONTAGES[name]


def validate(montage_name, data):
    """Raise a clear error instead of a bare IndexError deep inside CCA."""
    m = get(montage_name)
    n = data.shape[0]
    need = max(max(m["ssvep"]), max(m["artifact"])) + 1
    if n < need:
        raise ValueError(
            f"Montage {montage_name!r} needs >= {need} channels but data has {n}. "
            f"Are you feeding 8-channel Cyton data to a 64-channel config?"
        )
    return True
