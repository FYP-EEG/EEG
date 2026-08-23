"""
Author: Brian
Created on: 16/8/2026
Purpose: Step 3 -- fit subject-specific parameters from a calibration recording and
         report the metrics that actually predict GUI usability.

Consumes the .npz produced by EEG/calibration_record.py and produces:
  * a confidence threshold fitted to THIS subject's idle data
  * false-positives per minute of idle  (the headline number for a GUI library)
  * precision / recall on attentive trials
  * ITR in bits/min
  * a saved profile .json the engine loads at startup

Run:
    python ML/calibration.py --recording dataset/calib_S01_*.npz
    python ML/calibration.py --recording ... --sweep      # threshold sweep table
"""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ML.hybrid_classifier import HybridSSVEPClassifier, NO_ACTION  # noqa: E402


# ------------------------------------------------------------------- utilities
def load(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"], d["y"], d["labels"].astype(str), int(d["fs"]),
            str(d["montage"]), list(d["target_freqs"]))


def itr_bits_per_min(n_classes, accuracy, selections_per_min):
    """Wolpaw ITR. Standard BCI metric -- report this, not bare accuracy."""
    if selections_per_min <= 0 or n_classes < 2:
        return 0.0
    p = min(max(accuracy, 1e-6), 1 - 1e-6)
    bits = (math.log2(n_classes) + p * math.log2(p)
            + (1 - p) * math.log2((1 - p) / (n_classes - 1)))
    return max(0.0, bits) * selections_per_min


def score_all(clf, X, verbose=True):
    """Return (best_scores, argmax_labels, artifact_mask)."""
    best, pred, art = [], [], []
    n = len(X)
    for i, w in enumerate(X):
        if clf.artifact_detection(w, is_raw=True):
            art.append(True)
            best.append(np.nan)
            pred.append(-99)
        else:
            art.append(False)
            s = clf.fbcca_scores(clf.apply_filter(w))
            best.append(float(np.max(s)))
            pred.append(int(np.argmax(s)))
        if verbose and (i + 1) % 50 == 0:
            print(f"    scored {i+1}/{n}")
    return np.array(best), np.array(pred), np.array(art)


# ----------------------------------------------------------------------- report
def analyse(recording, percentile=95.0, debounce=3, window=750, hop=250,
            sweep=False, out=None):
    X, y, labels, fs, montage, target_freqs = load(recording)
    print(f"\nLoaded {recording}")
    print(f"  {len(X)} windows, {X.shape[1]} channels, {X.shape[2]} samples, fs={fs}")
    print(f"  montage={montage}  targets={target_freqs}")
    uniq, counts = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, counts):
        print(f"    {u:<18} {c:>4} windows")

    clf = HybridSSVEPClassifier(target_freqs=target_freqs, sample_freq=fs,
                                montage=montage)

    print("\nScoring windows with FBCCA...")
    best, pred, art = score_all(clf, X)

    is_target = np.isin(y, [0, 1])
    is_idle = (y == -1)
    is_blinkblock = (y == 2)

    clean_idle = is_idle & ~art
    clean_tgt = is_target & ~art

    print("\n" + "=" * 70)
    print("CONFIDENCE SEPARATION (this is what makes NO_ACTION possible)")
    print("=" * 70)
    if clean_tgt.sum():
        print(f"  attentive  n={clean_tgt.sum():>4}  mean={np.nanmean(best[clean_tgt]):.4f}"
              f"  sd={np.nanstd(best[clean_tgt]):.4f}"
              f"  p05={np.nanpercentile(best[clean_tgt],5):.4f}")
    if clean_idle.sum():
        print(f"  idle       n={clean_idle.sum():>4}  mean={np.nanmean(best[clean_idle]):.4f}"
              f"  sd={np.nanstd(best[clean_idle]):.4f}"
              f"  p95={np.nanpercentile(best[clean_idle],95):.4f}")
    if clean_tgt.sum() and clean_idle.sum():
        d = ((np.nanmean(best[clean_tgt]) - np.nanmean(best[clean_idle]))
             / np.sqrt((np.nanvar(best[clean_tgt]) + np.nanvar(best[clean_idle])) / 2))
        print(f"  separation (Cohen's d) = {d:.2f}")

    print(f"\n  artifact windows rejected: {art.sum()}/{len(X)}"
          f"  (blink block caught: {(art & is_blinkblock).sum()}/{is_blinkblock.sum()})")

    # -------------------------------------------------------------- threshold
    if clean_idle.sum() == 0:
        print("\n!! No clean idle windows -- cannot calibrate a threshold.")
        print("   Re-record with idle / idle_distracted blocks (see Step 3 doc).")
        return None

    thresholds = ([np.nanpercentile(best[clean_idle], p)
                   for p in (80, 90, 95, 97.5, 99)] if sweep
                  else [np.nanpercentile(best[clean_idle], percentile)])
    plabels = [80, 90, 95, 97.5, 99] if sweep else [percentile]

    hop_s = hop / fs
    idle_minutes = clean_idle.sum() * hop_s / 60.0

    print("\n" + "=" * 70)
    print("GUI USABILITY METRICS  (accuracy alone does not predict these)")
    print("=" * 70)
    print(f"{'pct':>6} {'thresh':>8} {'FP/min idle':>12} {'precision':>10} "
          f"{'recall':>8} {'ITR b/min':>10}")
    print("-" * 70)

    rows = []
    for p, thr in zip(plabels, thresholds):
        acc_mask = best >= thr

        # consecutive-run debounce, mirroring BCIEngine
        fired = np.zeros(len(X), dtype=bool)
        run_lbl, run_len = -99, 0
        for i in range(len(X)):
            if art[i] or not acc_mask[i]:
                run_lbl, run_len = -99, 0
                continue
            if pred[i] == run_lbl:
                run_len += 1
            else:
                run_lbl, run_len = pred[i], 1
            if run_len >= debounce:
                fired[i] = True
                run_len = 0

        fp_idle = int((fired & is_idle).sum())
        tp = int((fired & is_target & (pred == y)).sum())
        wrong = int((fired & is_target & (pred != y)).sum())
        issued = tp + wrong + fp_idle
        precision = tp / issued if issued else 0.0
        recall = tp / max(1, int(is_target.sum()))
        fp_per_min = fp_idle / idle_minutes if idle_minutes > 0 else float("nan")
        sel_per_min = (tp + wrong) / max(1e-9, (is_target.sum() * hop_s / 60.0))
        acc_cond = tp / max(1, tp + wrong)
        itr = itr_bits_per_min(2, acc_cond, sel_per_min)

        rows.append(dict(percentile=p, threshold=float(thr), fp_per_min=fp_per_min,
                         precision=precision, recall=recall, itr=itr,
                         tp=tp, wrong=wrong, fp_idle=fp_idle))
        print(f"{p:>6} {thr:>8.4f} {fp_per_min:>12.2f} {precision:>10.1%} "
              f"{recall:>8.1%} {itr:>10.1f}")

    chosen = rows[len(rows) // 2] if sweep else rows[0]
    print("-" * 70)
    print(f"  idle recorded: {idle_minutes:.1f} min")
    print(f"  chosen threshold = {chosen['threshold']:.4f} "
          f"(p{chosen['percentile']})")

    # ------------------------------------------------------------------ profile
    profile = {
        "subject": Path(recording).stem,
        "montage": montage,
        "sample_freq": fs,
        "target_freqs": [float(f) for f in target_freqs],
        "confidence_threshold": float(chosen["threshold"]),
        "debounce_window": debounce,
        "window_samples": int(X.shape[2]),
        "hop_samples": hop,
        "artifact_threshold": float(clf.artifact_threshold),
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in chosen.items()},
        "n_windows": int(len(X)),
        "idle_minutes": float(idle_minutes),
    }
    out = out or (ROOT / "dataset" / f"profile_{Path(recording).stem}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(profile, indent=2))
    print(f"\nProfile saved -> {out}")
    print("Load it with:  BCIEngine.from_profile(path)")
    return profile


def main():
    ap = argparse.ArgumentParser(description="Fit subject profile from calibration data")
    ap.add_argument("--recording", required=True,
                    help="path to calib_*.npz (globs allowed -- newest is used)")
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--debounce", type=int, default=3)
    ap.add_argument("--hop", type=int, default=250)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.recording))
    if not matches:
        print(f"No recording matched {args.recording!r}")
        return 1
    analyse(matches[-1], percentile=args.percentile, debounce=args.debounce,
            hop=args.hop, sweep=args.sweep, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
