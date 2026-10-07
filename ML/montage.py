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

# THE MOTOR-IMAGERY MONTAGE  (chosen 6/10/2026)
#
# This is now the montage for live MI recording, not just offline PhysioNet
# work. It matches the Colab starter model exactly - the notebook picks
# ['FC3','FC4','C3','CZ','C4','CP3','CP4','PZ'] - so a model trained there can
# be loaded here with no channel remapping.
#
# ONE KNOWN COST. `artifact` points at FC3/FC4, which are frontocentral MOTOR
# sites, not forehead. Two consequences, both to check at Layer 1 before
# trusting a session:
#   1. Blinks are SMALLER here than at Fp1/Fp2. A blink measured 221.6 uV at
#      the forehead; at FC3/FC4 it is attenuated, so the 100 uV threshold in
#      ArtifactDetector may miss some. If Layer 1 shows deliberate blinks going
#      undetected, lower threshold_uv rather than assuming the channel is clean.
#   2. These channels also carry the motor signal, so a lowered threshold can
#      reject genuine imagery as artifact. The two failure modes pull in
#      opposite directions; pick the threshold from recorded blinks, not a
#      default.
CYTON_8_MOTOR = {
    "name": "cyton8_motor",
    "n_channels": 8,
    "labels": ["FC3", "FC4", "C3", "CZ", "C4", "CP3", "CP4", "PZ"],
    "ssvep": [7],                # PZ -- poor for SSVEP, this is not an SSVEP montage
    "artifact": [0, 1],          # FC3, FC4 -- see the warning above
    "motor": [0, 1, 2, 3, 4, 5, 6, 7],
    "motor_lateral": [2, 4],     # C3, C4 -- the pair that carries ERD, and the
                                 # contrast the Layer 2 go/no-go gate measures
}

# --- Alternative MI montage, kept for older recordings ------------------------
# Added 21/9/2026, superseded by CYTON_8_MOTOR on 6/10/2026. Still registered so
# that recordings already saved with montage="cyton8_mi" keep loading; it is no
# longer the default anywhere. Its advantage over CYTON_8_MOTOR is real frontal
# blink channels; its cost is two fewer motor channels and a mismatch with the
# Colab starter model.
#
# Two deliberate differences from CYTON_8_MOTOR:
#   1. Fp1/Fp2 are kept on channels 0-1 purely for BLINK DETECTION. Giving up two
#      motor channels is worth it: without a frontal reference you cannot tell a
#      blink from motor activity, and eye artifacts are large enough to dominate
#      any classifier.
#   2. C3/C4 sit on channels 2/4 with CZ between them. C3/C4 over the left/right
#      motor cortex are where the mu/beta desynchronisation appears, and CZ gives
#      the spatial filters a midline reference to subtract common activity.
CYTON_8_MI = {
    "name": "cyton8_mi",
    "n_channels": 8,
    "labels": ["Fp1", "Fp2", "C3", "CZ", "C4", "CP3", "CP4", "PZ"],
    "ssvep": [6, 7],             # not meaningful here; kept so validate() works
    "artifact": [0, 1],          # Fp1, Fp2 -- real forehead sites for blinks
    "motor": [2, 3, 4, 5, 6, 7], # C3 CZ C4 CP3 CP4 PZ
    "motor_lateral": [2, 4],     # C3, C4 -- the pair that actually carries ERD
}

# Explicit rather than a comprehension (Brian, 6/10/2026) - a typo in a "name"
# field used to silently register a montage under the wrong key, which is how
# CYTON_8_SSVEP spent a while registered as "cyton8_motor".
MONTAGES = {
    "benchmark64":  BENCHMARK_64,
    "cyton8_ssvep": CYTON_8_SSVEP,
    "cyton8_motor": CYTON_8_MOTOR,
    "cyton8_mi":    CYTON_8_MI,
}

# Guard the thing the explicit dict is meant to prevent: a key that disagrees
# with the montage's own "name" would make get() and validate() diverge.
for _k, _m in MONTAGES.items():
    assert _k == _m["name"], f"montage registered as {_k!r} but names itself {_m['name']!r}"


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
