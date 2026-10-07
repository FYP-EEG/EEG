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


def screen_trials(X, labels=None, fs=250, amp_uv=400.0, jump_uv=75.0,
                  flat_uv=0.05, z_max=4.0, band=(8.0, 30.0), verbose=True):
    """Flag trials that are unfit to train on.

    :param X: (n_trials, n_channels, n_times), microvolts
    :param amp_uv: peak absolute amplitude above which a trial is artifact.
        400, far above the 100 uV live blink lockout, because the measurement
        above showed blink rejection LOSING 3.1 points: CSP already suppresses
        blinks, so dropping those trials is pure data loss. This catches only
        gross faults - a disconnected lead railing, a cable yanked.
    :param jump_uv: largest allowed sample-to-sample step. Catches electrode
        pops and cable knocks, which amplitude alone can miss when the
        excursion is brief.
    :param flat_uv: a channel with less variation than this is disconnected.
    :param z_max: robust z on log band power, computed WITHIN each class so a
        genuine class difference is not read as an outlier.
    :return: dict(keep=bool mask, reasons={reason: mask}, report=str)
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    reasons = {r: np.zeros(n, dtype=bool) for r in REASONS}

    centred = X - X.mean(axis=2, keepdims=True)
    reasons["amplitude"] = np.max(np.abs(centred), axis=(1, 2)) > amp_uv
    reasons["flat"] = np.any(X.std(axis=2) < flat_uv, axis=1)
    reasons["jump"] = np.max(np.abs(np.diff(X, axis=2)), axis=(1, 2)) > jump_uv

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
    return {"keep": keep, "reasons": reasons,
            "report": format_report(keep, reasons, verbose=verbose)}


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
