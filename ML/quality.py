"""
Author: Anson
Created on: 6/10/2026
Purpose: decide which recorded trials are fit to train on.

TWO DIFFERENT THINGS GO WRONG, AND THEY NEED DIFFERENT CHECKS
-------------------------------------------------------------
1. THE HARDWARE LIES. A dry electrode lifts, the cable is tugged, mains noise
   swamps a channel. Layer 1 catches this at the level of a whole recording -
   flat channels, 50 Hz, no alpha blocking. But a recording can pass Layer 1
   and still contain individual ruined trials.

2. THE USER IS HUMAN. They blink during the cue, swallow, shift in the chair,
   lose attention, or imagine the wrong hand. The EEG is perfectly recorded;
   the LABEL is wrong. No amount of signal processing fixes a trial that is
   labelled "left" while the person was thinking about lunch.

Case 2 is the dangerous one, because it is invisible. The data looks clean, the
model trains without complaint, and the error surfaces as accuracy that will
not rise no matter what you try. Mislabelled trials do not just add noise, they
actively teach the classifier the wrong boundary.

WHAT THIS MODULE DOES NOT DO
----------------------------
It cannot tell whether someone genuinely imagined a movement. Nothing can -
there is no ground truth beyond what they report. What it can do is remove
trials that are PHYSICALLY implausible as clean imagery, and trials the user
themselves marked as wrong by clicking the wrong button. Everything else is a
limit of the paradigm, not a bug.

WHAT THE MEASUREMENTS SAID  (and they contradicted the intuition)
------------------------------------------------------------------
Simulated at realistic difficulty - clean data scoring ~75%, inside this
project's measured 61.4-83.3% band - comparing raw training against screened:

    contamination            raw   screened   kept    gain
    clean                  75.7%      75.5%   100%    -0.2
    20% blinks             75.7%      72.6%    80%    -3.1
    20% electrode pops     74.8%      75.6%    82%    +0.8
    blinks + pops          74.5%      75.0%    70%    +0.5

Two findings worth keeping:

  * BLINKS BARELY MATTER TO CSP. 20% blinked trials scored the same as clean
    data, 75.7% both. Rejecting them cost 20% of the dataset and LOST 3.1
    points. CSP is a spatial filter; a blink is a spatially distinct component
    it learns to suppress. So the aggressive amplitude threshold used for LIVE
    lockout is wrong for OFFLINE training, and the default here is permissive
    on purpose.
  * SUDDEN STEPS DO MATTER. Electrode pops shift the covariance CSP is built
    from, and removing them pays for itself.

AND THE ONE THAT MATTERS MOST
-----------------------------
A user who imagines the WRONG hand produces clean EEG with a wrong label, and
NO signal-based rule can find it:

    mislabel rate      raw   signal screening   cue-log compliance
              0%    75.1%              75.0%                75.0%
             10%    68.6%              68.6%                74.8%
             20%    63.9%              64.5%                75.2%
             40%    52.7%              52.6%                73.9%

Statistical screening recovers nothing. The cue log recovers everything -
52.7% back to 73.9% at a 40% mistake rate. The conclusion is not a better
algorithm: it is that WHAT THE USER ACTUALLY DID MUST BE RECORDED AT THE TIME.
EEG/data_record.py already writes it to cue_log_*.csv. Use it.

REJECTING TOO MUCH IS ALSO A FAILURE
------------------------------------
Every rejection rule is a knob that can delete your dataset. `screen_trials`
reports the rejection RATE and refuses to be silent about it: losing more than
about a third of trials means the recording, the montage or the threshold is
wrong, not that the user is bad at the task.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

#: reasons a trial can be dropped, in the order they are tested
REASONS = ("amplitude", "flat", "jump", "outlier", "noncompliant")


def _robust_z(v):
    """z-score using median and MAD, so the outliers do not inflate the scale
    they are being measured against."""
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad <= 0:
        return np.zeros_like(v)
    return 0.6745 * (v - med) / mad


def _adaptive_metric_mask(metric, z=12.0, peer_factor=8.0,
                          history_factor=4.0, floor=1e-9):
    """Find gross metric outliers without assuming one microvolt scale."""
    values = np.asarray(metric, dtype=float)
    baseline = np.median(values, axis=0)
    spread = 1.4826 * np.median(np.abs(values - baseline), axis=0)
    history_limit = np.maximum.reduce([
        baseline + float(z) * spread,
        float(history_factor) * np.maximum(baseline, floor),
        np.full_like(baseline, floor),
    ])
    historical = values > history_limit[np.newaxis, :]

    peer = np.median(values, axis=1)
    peer_limit = float(peer_factor) * np.maximum(peer, floor)
    peer_outlier = values > peer_limit[:, np.newaxis]

    trial_scale = np.median(values, axis=1)
    trial_median = float(np.median(trial_scale))
    trial_spread = float(1.4826 * np.median(np.abs(trial_scale - trial_median)))
    trial_limit = max(trial_median + float(z) * trial_spread,
                      float(history_factor) * max(trial_median, floor), floor)
    boardwide = trial_scale > trial_limit
    mask = historical | peer_outlier | boardwide[:, np.newaxis]
    return mask, {
        "history_limit": history_limit,
        "peer_limit": peer_limit,
        "trial_limit": trial_limit,
    }


def _longest_true_run(mask):
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _clipping_fraction(X):
    fractions = np.zeros(X.shape[:2], dtype=float)
    spans = np.ptp(X, axis=2)
    for trial in range(X.shape[0]):
        for channel in range(X.shape[1]):
            row = X[trial, channel]
            tolerance = max(float(spans[trial, channel]) * 1e-7, 1e-9)
            low_mask = np.abs(row - np.min(row)) <= tolerance
            high_mask = np.abs(row - np.max(row)) <= tolerance
            longest = max(_longest_true_run(low_mask),
                          _longest_true_run(high_mask))
            # A periodic sine can revisit one exact maximum on every cycle.
            # Saturation creates a contiguous plateau, so require >=3 adjacent
            # extreme samples before treating repeated extrema as clipping.
            if longest >= 3:
                fractions[trial, channel] = max(
                    float(np.mean(low_mask)), float(np.mean(high_mask)))
    return fractions, spans


def screen_trials(X, labels=None, fs=250, amp_uv=None, jump_uv=None,
                  flat_uv=0.05, z_max=4.0, band=(8.0, 30.0), verbose=True,
                  adaptive_z=12.0, peer_factor=8.0, history_factor=4.0,
                  clip_fraction=0.02):
    """Flag trials that are unfit to train on.

    :param X: (n_trials, n_channels, n_times), microvolts
    :param amp_uv: optional explicit peak cap for a controlled experiment.
        The default is None: gross amplitude faults are found relative to the
        recording's robust per-channel, per-trial and peer-channel scales.
        Stable large-but-valid active-electrode scaling is therefore retained.
    :param jump_uv: optional explicit sample-jump cap. By default jumps are
        judged against robust measured step sizes rather than a fixed µV value.
    :param flat_uv: a channel with less variation than this is disconnected.
    :param z_max: robust z on log band power, computed WITHIN each class so a
        genuine class difference is not read as an outlier.
    :return: dict(keep=bool mask, reasons={reason: mask}, report=str)
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    reasons = {r: np.zeros(n, dtype=bool) for r in REASONS}

    if X.ndim != 3 or X.shape[2] < 2:
        raise ValueError(f"expected (trials,channels,times) with >=2 samples, got {X.shape}")
    if not np.isfinite(X).all():
        # Keep the public reason vocabulary stable: non-finite values are gross
        # amplitude failures and must never enter filtering/model fitting.
        nonfinite = ~np.isfinite(X).all(axis=(1, 2))
    else:
        nonfinite = np.zeros(n, dtype=bool)

    centred = X - np.median(X, axis=2, keepdims=True)
    peak_by_channel = np.max(np.abs(centred), axis=2)
    std_by_channel = np.std(X, axis=2)
    jump_by_channel = np.max(np.abs(np.diff(X, axis=2)), axis=2)
    clip_by_channel, span_by_channel = _clipping_fraction(X)

    amplitude_channels, amplitude_limits = _adaptive_metric_mask(
        peak_by_channel, z=adaptive_z, peer_factor=peer_factor,
        history_factor=history_factor, floor=max(float(flat_uv), 1e-9))
    jump_channels, jump_limits = _adaptive_metric_mask(
        jump_by_channel, z=adaptive_z, peer_factor=peer_factor,
        history_factor=history_factor, floor=max(float(flat_uv), 1e-9))
    clipping_channels = ((clip_by_channel >= float(clip_fraction)) &
                         (span_by_channel > float(flat_uv) * 20.0))
    amplitude_channels |= clipping_channels
    if amp_uv is not None:
        amplitude_channels |= peak_by_channel > float(amp_uv)
    if jump_uv is not None:
        jump_channels |= jump_by_channel > float(jump_uv)
    flat_channels = std_by_channel < float(flat_uv)

    reasons["amplitude"] = amplitude_channels.any(axis=1) | nonfinite
    reasons["flat"] = flat_channels.any(axis=1)
    reasons["jump"] = jump_channels.any(axis=1)

    # band power outliers, judged against this subject's own distribution
    sos = butter(4, list(band), btype="band", fs=fs, output="sos")
    power = np.log(np.var(sosfiltfilt(sos, X, axis=-1), axis=2) + 1e-12)
    lab = (np.asarray(labels) if labels is not None
           else np.zeros(n, dtype=int))
    for cls in np.unique(lab):
        m = lab == cls
        if m.sum() < 4:                       # too few to judge an outlier
            continue
        z = np.abs(np.apply_along_axis(_robust_z, 0, power[m]))
        reasons["outlier"][m] = z.max(axis=1) > z_max

    keep = ~np.any([reasons[r] for r in REASONS], axis=0)
    return {
        "keep": keep,
        "reasons": reasons,
        "channel_reasons": {
            "amplitude": amplitude_channels,
            "flat": flat_channels,
            "jump": jump_channels,
            "clipping": clipping_channels,
        },
        "metrics": {
            "peak_uv": peak_by_channel,
            "std_uv": std_by_channel,
            "jump_uv": jump_by_channel,
            "clip_fraction": clip_by_channel,
        },
        "adaptive_limits": {
            "amplitude_history_uv": amplitude_limits["history_limit"],
            "amplitude_trial_uv": amplitude_limits["trial_limit"],
            "jump_history_uv": jump_limits["history_limit"],
            "jump_trial_uv": jump_limits["trial_limit"],
        },
        "report": format_report(keep, reasons, verbose=verbose),
    }


def apply_compliance(result, complied):
    """Fold in cue-vs-click agreement from a cue log.

    :param complied: bool array, True where the user selected the cued button
    """
    complied = np.asarray(complied, dtype=bool)
    if len(complied) != len(result["keep"]):
        raise ValueError(f"cue log has {len(complied)} trials, data has "
                         f"{len(result['keep'])}")
    result["reasons"]["noncompliant"] = ~complied
    result["keep"] = result["keep"] & complied
    result["report"] = format_report(result["keep"], result["reasons"])
    return result


def load_cue_log(path):
    """Read a cue_log_*.csv written by EEG/data_record.py.

    :return: (intended, actual, complied) as lists
    """
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return [], [], []

    def pick(row, *names):
        for n in names:
            if n in row and row[n] != "":
                return row[n]
        return None

    intended = [pick(r, "intended", "cue", "target") for r in rows]
    actual = [pick(r, "actual", "clicked", "selected") for r in rows]
    complied = [i is not None and i == a for i, a in zip(intended, actual)]
    return intended, actual, complied


def format_report(keep, reasons, verbose=True):
    n = len(keep)
    kept = int(keep.sum())
    lines = [f"  trials: {n}  kept: {kept}  dropped: {n - kept} "
             f"({(n - kept) / max(1, n):.0%})"]
    for r in REASONS:
        c = int(reasons[r].sum())
        if c:
            lines.append(f"    {r:<14} {c:>4}")
    rate = (n - kept) / max(1, n)
    if rate > 0.33:
        lines.append("  WARNING: more than a third of trials dropped. That is "
                     "usually the recording, the montage or the threshold - "
                     "not the user. Check Layer 1 before lowering the bar.")
    if kept and reasons["noncompliant"].sum() / max(1, n) > 0.25:
        lines.append("  NOTE: high non-compliance. Either the cue is unclear "
                     "or the task is too fast; lengthen the cue rather than "
                     "discarding more data.")
    return "\n".join(lines)


def screen_and_report(X, labels=None, fs=250, cue_log=None, **kw):
    """Convenience: screen, optionally fold in a cue log, print, return mask."""
    res = screen_trials(X, labels=labels, fs=fs, **kw)
    if cue_log and Path(cue_log).exists():
        _, _, complied = load_cue_log(cue_log)
        if len(complied) == len(X):
            res = apply_compliance(res, complied)
    print(res["report"])
    return res["keep"]
