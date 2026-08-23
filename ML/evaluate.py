"""
Author: Brian
Created on: 16/8/2026
Purpose: Step 6 -- leakage-safe evaluation comparing FBCCA vs TRCA (SSVEP) and
         CSP+LDA vs FBCSP vs Riemannian (MI).

THE LEAKAGE TRAP THIS EXISTS TO AVOID
-------------------------------------
Step 3 records with a 750-sample window and a 250-sample hop, so CONSECUTIVE WINDOWS
SHARE 500 OF 750 SAMPLES (67% overlap). Verified directly:

    window[0][:, 250:] == window[1][:, :500]   ->  True

A random train/test split therefore puts near-duplicate windows on both sides and
reports a wildly inflated score. With 30 windows per class there are really only ~10
independent 5-second trials.

Everything here uses GROUPED cross-validation: overlapping windows from the same
recording block share a group id and always land in the same fold. Numbers produced
this way are lower than a naive split would give -- and they are the ones you can
defend in a report.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ML.hybrid_classifier import HybridSSVEPClassifier, NO_ACTION  # noqa: E402
from ML.trca import TRCAClassifier                                 # noqa: E402


# ----------------------------------------------------------------- grouping
def infer_groups(labels, window=750, hop=250):
    """Assign a group id per contiguous block of same-label windows.

    Windows within one recording block overlap, so they must never be split across
    folds. A block boundary is any change of label.
    """
    groups = np.zeros(len(labels), dtype=int)
    g = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            g += 1
        groups[i] = g
    return groups


def subdivide_groups(groups, y, n_sub=3):
    """Split each block into n_sub contiguous chunks so CV has enough folds.

    Chunks remain contiguous, so overlap still cannot cross a fold boundary except
    at the single seam between adjacent chunks. With hop=250 and window=750 that
    seam shares at most 2 windows; we drop those to keep the split clean.
    """
    out = np.zeros_like(groups)
    keep = np.ones(len(groups), dtype=bool)
    gid = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        parts = np.array_split(idx, n_sub)
        for pi, part in enumerate(parts):
            out[part] = gid
            if pi > 0 and len(part) >= 2:
                keep[part[:2]] = False        # discard the overlapping seam
            gid += 1
    return out, keep


def grouped_folds(groups, y, n_splits=5, seed=0):
    """Yield (train_idx, test_idx) with all of a group on one side, classes balanced."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    # label each group by its (single) class
    gcls = {g: y[groups == g][0] for g in uniq}
    by_class = {}
    for g in uniq:
        by_class.setdefault(gcls[g], []).append(g)
    for c in by_class:
        rng.shuffle(by_class[c])

    folds = [[] for _ in range(n_splits)]
    for c, gs in by_class.items():
        for i, g in enumerate(gs):
            folds[i % n_splits].append(g)

    for k in range(n_splits):
        test_g = set(folds[k])
        if not test_g:
            continue
        test = np.isin(groups, list(test_g))
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        yield np.where(train)[0], np.where(test)[0]


# ------------------------------------------------------------------ SSVEP
def evaluate_ssvep(npz_path, n_splits=5, percentile=95.0, verbose=True):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    labels = d["labels"].astype(str)
    fs = int(d["fs"])
    freqs = [float(f) for f in d["target_freqs"]]
    montage = str(d["montage"])

    is_t = np.isin(labels, ["target_0", "target_1"])
    y_all = np.where(labels == "target_1", 1, 0)
    idle_mask = np.isin(labels, ["idle", "idle_distracted"])

    Xt, yt = X[is_t], y_all[is_t]
    groups_all = infer_groups(labels)
    gt = groups_all[is_t]
    gt, keep = subdivide_groups(gt, yt, n_sub=3)
    Xt, yt, gt = Xt[keep], yt[keep], gt[keep]

    fb = HybridSSVEPClassifier(target_freqs=freqs, sample_freq=fs, montage=montage)
    idle_X = X[idle_mask]

    if verbose:
        print("\n" + "=" * 74)
        print("SSVEP: FBCCA (training-free) vs TRCA (subject-specific)")
        print("=" * 74)
        print(f"  {len(Xt)} target windows in {len(np.unique(gt))} independent groups")
        print(f"  {len(idle_X)} idle windows | grouped {n_splits}-fold CV (no leakage)")

    # --- FBCCA needs no training: evaluate on the same test folds for fairness
    fb_acc, trca_acc = [], []
    fb_conf, trca_conf = [], []
    n_folds = 0
    for train, test in grouped_folds(gt, yt, n_splits=n_splits):
        n_folds += 1
        # FBCCA
        preds, confs = [], []
        for i in test:
            s = fb.fbcca_scores(fb.apply_filter(Xt[i]))
            preds.append(int(np.argmax(s)))
            confs.append(float(np.max(s)))
        fb_acc.append(np.mean(np.array(preds) == yt[test]))
        fb_conf.append(np.mean(confs))

        # TRCA trained on this fold's training groups only
        try:
            tr = TRCAClassifier(target_freqs=freqs, sample_freq=fs,
                                channels=fb.ssvep_channels)
            Xtr = np.array([fb.apply_filter(x) for x in Xt[train]])
            tr.fit(Xtr, yt[train])
            preds, confs = [], []
            for i in test:
                s = tr.scores(fb.apply_filter(Xt[i]))
                preds.append(int(np.argmax(s)))
                confs.append(float(np.max(s)))
            trca_acc.append(np.mean(np.array(preds) == yt[test]))
            trca_conf.append(np.mean(confs))
        except Exception as exc:
            if verbose:
                print(f"    TRCA fold skipped: {exc}")

    res = {
        "n_folds": n_folds,
        "fbcca_acc": float(np.mean(fb_acc)) if fb_acc else float("nan"),
        "fbcca_sd": float(np.std(fb_acc)) if fb_acc else float("nan"),
        "trca_acc": float(np.mean(trca_acc)) if trca_acc else float("nan"),
        "trca_sd": float(np.std(trca_acc)) if trca_acc else float("nan"),
        "fbcca_conf": float(np.mean(fb_conf)) if fb_conf else float("nan"),
        "trca_conf": float(np.mean(trca_conf)) if trca_conf else float("nan"),
    }

    # --- idle separation for whichever method wins
    if verbose:
        print(f"\n  {'method':<12}{'accuracy':>18}{'mean conf (target)':>22}")
        print("  " + "-" * 52)
        print(f"  {'FBCCA':<12}{res['fbcca_acc']:>11.1%} +/-{res['fbcca_sd']:>5.1%}"
              f"{res['fbcca_conf']:>22.3f}")
        if trca_acc:
            print(f"  {'TRCA':<12}{res['trca_acc']:>11.1%} +/-{res['trca_sd']:>5.1%}"
                  f"{res['trca_conf']:>22.3f}")
        else:
            print("  TRCA        (insufficient data to train)")

    # idle rejection comparison on a full-data TRCA
    if len(idle_X) and trca_acc:
        tr_full = TRCAClassifier(target_freqs=freqs, sample_freq=fs,
                                 channels=fb.ssvep_channels)
        tr_full.fit(np.array([fb.apply_filter(x) for x in Xt]), yt)
        idle_f = [fb.apply_filter(w) for w in idle_X]
        tgt_f = [fb.apply_filter(w) for w in Xt]
        i_tr = np.array([np.max(tr_full.scores(w)) for w in idle_f])
        t_tr = np.array([np.max(tr_full.scores(w)) for w in tgt_f])
        i_fb = np.array([np.max(fb.fbcca_scores(w)) for w in idle_f])
        t_fb = np.array([np.max(fb.fbcca_scores(w)) for w in tgt_f])

        def d(a, b):
            return (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)

        res["fbcca_sep"] = float(d(i_fb, t_fb))
        res["trca_sep"] = float(d(i_tr, t_tr))
        if verbose:
            print(f"\n  idle-vs-target separation (Cohen's d, higher is better)")
            print(f"    FBCCA {res['fbcca_sep']:.2f}      TRCA {res['trca_sep']:.2f}")
            print("  NOTE: TRCA is fitted on ALL target data here, so this separation"
                  "\n        figure is optimistic; the accuracy above is the honest one.")
    return res


# --------------------------------------------------------------------- MI
def evaluate_mi(X, y, groups=None, sample_freq=250, n_splits=5, verbose=True):
    """Compare MI pipelines under grouped CV."""
    from ML.mi_advanced import PIPELINES

    X = np.asarray(X, float)
    y = np.asarray(y)
    if groups is None:
        groups = np.arange(len(y))

    if verbose:
        print("\n" + "=" * 74)
        print("Motor Imagery: CSP+LDA vs FBCSP vs Riemannian")
        print("=" * 74)
        print(f"  {len(X)} trials, {X.shape[1]} channels, {X.shape[2]} samples, "
              f"{len(np.unique(groups))} groups")

    results = {}
    for name, factory in PIPELINES.items():
        accs = []
        for train, test in grouped_folds(groups, y, n_splits=n_splits):
            try:
                pipe = factory(sample_freq=sample_freq)
                pipe.fit(X[train], y[train])
                accs.append(float(np.mean(pipe.predict(X[test]) == y[test])))
            except Exception as exc:
                if verbose:
                    print(f"    {name} fold failed: {exc}")
        if accs:
            results[name] = (float(np.mean(accs)), float(np.std(accs)))

    if verbose:
        print(f"\n  {'pipeline':<16}{'accuracy':>20}")
        print("  " + "-" * 36)
        for k, (m, sd) in sorted(results.items(), key=lambda kv: -kv[1][0]):
            print(f"  {k:<16}{m:>13.1%} +/-{sd:>5.1%}")
        print("\n  Chance = 50.0%. Cross-subject MI is hard; see accuracy_strategy.md")
    return results


def evaluate_mi_within_vs_cross(X, y, groups, sample_freq=250, verbose=True):
    """The comparison that actually answers 'should we upgrade the MI classifier?'

    Runs every pipeline twice: trained on OTHER subjects (cross) and trained on the
    SAME subject (within). The gap between the two columns tells you whether the
    bottleneck is the algorithm or the lack of per-subject calibration.
    """
    from ML.mi_advanced import PIPELINES
    X = np.asarray(X, float); y = np.asarray(y); groups = np.asarray(groups)
    rows = {}
    for name, factory in PIPELINES.items():
        cross = []
        for train, test in grouped_folds(groups, y, n_splits=4):
            try:
                p = factory(sample_freq=sample_freq)
                p.fit(X[train], y[train])
                cross.append(float(np.mean(p.predict(X[test]) == y[test])))
            except Exception:
                pass
        within = []
        for s in np.unique(groups):
            m = groups == s
            Xs, ys = X[m], y[m]
            n = len(ys) // 2
            if n < 8 or len(np.unique(ys[:n])) < 2:
                continue
            try:
                p = factory(sample_freq=sample_freq)
                p.fit(Xs[:n], ys[:n])
                within.append(float(np.mean(p.predict(Xs[n:]) == ys[n:])))
            except Exception:
                pass
        rows[name] = (np.mean(cross) if cross else float("nan"),
                      np.std(cross) if cross else float("nan"),
                      np.mean(within) if within else float("nan"),
                      np.std(within) if within else float("nan"))

    if verbose:
        print("\n" + "=" * 74)
        print("Motor Imagery: does the ALGORITHM or the CALIBRATION matter more?")
        print("=" * 74)
        print(f"  {'pipeline':<16}{'cross-subject':>20}{'within-subject':>22}{'gain':>10}")
        print("  " + "-" * 68)
        for k, (cm, cs, wm, ws) in sorted(rows.items(), key=lambda kv: -kv[1][2]):
            print(f"  {k:<16}{cm:>12.1%} +/-{cs:<5.1%}{wm:>14.1%} +/-{ws:<5.1%}"
                  f"{wm - cm:>+9.1%}")
        print("  " + "-" * 68)
        print("  Chance = 50%. The 'gain' column is what per-subject calibration buys.")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Step 6 model comparison")
    ap.add_argument("--recording", default=str(ROOT / "dataset" / "calib_*.npz"))
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--mi-synthetic", action="store_true",
                    help="also run the MI comparison on synthetic MI data")
    args = ap.parse_args()

    hits = sorted(glob.glob(args.recording))
    if not hits:
        print(f"No recording matched {args.recording}")
        return 1
    print(f"Recording: {hits[-1]}")
    evaluate_ssvep(hits[-1], n_splits=args.splits)

    if args.mi_synthetic:
        from ML.mi_synth import make_mi_dataset
        X, y, g = make_mi_dataset(n_subjects=6, trials_per_subject=40)
        evaluate_mi(X, y, groups=g, n_splits=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
