"""
Author: Anson
Created on: 26/9/2026
Purpose: train a motor-imagery model on ONE person's own data, on their own machine.

    from ML.train_local import train_user
    result = train_user("anson")

WHY LOCAL
---------
Measured on this project's own pipelines, one person / 40 trials:

    csp_lda       train   36 ms    predict  1.3 ms
    fbcsp         train  150 ms    predict  4.9 ms
    riemann_svm   train   31 ms    predict  3.2 ms
    riemann_lr    train   46 ms    predict  3.5 ms

There is nothing here a server does better. No GPU, no big model - a few matrix
operations. Sending EEG to a remote API would add network latency, an offline
failure mode, and a hosting bill, in exchange for arithmetic a laptop finishes
in under a second.

It also means raw EEG never leaves the machine, which removes a whole class of
privacy and ethics problems.

WHY PICK THE PIPELINE PER USER
------------------------------
Motor imagery is highly individual. One person's mu rhythm sits at 9 Hz, another
at 12.5 Hz; spatial patterns differ with skull and electrode placement. There is
no single best algorithm, so this trains ALL of them and keeps whichever scores
highest on that person's own data.

THE COLD-START PROBLEM
----------------------
Accuracy climbs with the amount of personal data (simulated, one subject):

     8 trials (~0.8 min)  ->  64.7%
    24 trials (~2.4 min)  ->  67.3%
    40 trials (~4.0 min)  ->  72.7%
   100 trials (~10  min)  ->  78.0%
   140 trials (~14  min)  ->  81.7%

So a first session is weak and improves as sessions accumulate. That is why
recordings are APPENDED rather than replaced, and why a shipped starter model is
worth having - see starter_model() below.
"""

import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# allow `python ML/train_local.py` as well as `from ML.train_local import ...`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.evaluate import grouped_folds          # noqa: E402
from ML.mi_advanced import PIPELINES           # noqa: E402
PROFILES = ROOT / "profiles"

#: below this many trials per class, a personal model is not worth trusting
MIN_TRIALS_PER_CLASS = 8

#: below this many SEPARATE recording blocks per class, the cross-validation
#: score is unreliable even if the trial count looks fine
MIN_BLOCKS_PER_CLASS = 3


# ------------------------------------------------------------------ loading
def load_user_data(user, root=None, include=("target_0", "target_1")):
    """Stack every recording this user has ever made.

    Sessions accumulate: each new recording is another file in profiles/<user>/,
    and all of them are used. `groups` keeps windows from the same recording
    block together so cross-validation cannot leak (consecutive windows overlap
    by 67%, so a random split would test on near-duplicates of the training set).
    """
    root = Path(root) if root else PROFILES
    udir = root / str(user).lower()
    files = sorted(glob.glob(str(udir / "calib_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No recordings in {udir}. Run the calibration first:\n"
            f"  python EEG/calibration_record.py --subject {user} "
            f"--backend brainflow --out profiles/{user}")

    Xs, ys, groups, idles, srcs = [], [], [], [], []
    gid = 0
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        X = d["X"].astype(np.float64)
        labels = d["labels"].astype(str)

        mask = np.isin(labels, list(include))
        if mask.any():
            Xs.append(X[mask])
            ys.append(np.array([list(include).index(l) for l in labels[mask]]))
            # new group id whenever the label changes = one recording block
            blocks = np.zeros(mask.sum(), dtype=int)
            sub = labels[mask]
            b = 0
            for i in range(1, len(sub)):
                if sub[i] != sub[i - 1]:
                    b += 1
                blocks[i] = b
            groups.append(blocks + gid)
            gid += b + 1
            srcs.extend([Path(fp).name] * int(mask.sum()))

        idle_mask = np.isin(labels, ["idle", "idle_distracted"])
        if idle_mask.any():
            idles.append(X[idle_mask])

    if not Xs:
        raise ValueError(f"Recordings found but no {include} windows in them.")

    return {
        "X": np.concatenate(Xs),
        "y": np.concatenate(ys),
        "groups": np.concatenate(groups),
        "idle": np.concatenate(idles) if idles else np.zeros((0, 8, 750)),
        "files": [Path(f).name for f in files],
        "fs": int(d["fs"]),
        "montage": str(d["montage"]),
    }


# ----------------------------------------------------------------- training
def train_user(user, root=None, sample_freq=None, n_splits=4, verbose=True):
    """Train every pipeline on this user's data and keep the best.

    :return: dict with the winner, all scores, and where the model was saved.
             None if there is not enough data yet.
    """
    root = Path(root) if root else PROFILES
    data = load_user_data(user, root=root)
    X, y, groups = data["X"], data["y"], data["groups"]
    fs = sample_freq or data["fs"]

    counts = np.bincount(y, minlength=2)
    if verbose:
        print(f"\nTraining a personal model for {user!r}")
        print(f"  {len(data['files'])} recording(s): {', '.join(data['files'])}")
        print(f"  {len(X)} trials  (class 0: {counts[0]}, class 1: {counts[1]})")
        print(f"  {len(np.unique(groups))} independent blocks, fs={fs}")

    """
    Guard against a misleading score.

    The calibration protocol records each class as ONE continuous block, so a
    short session gives only 1-2 blocks per class. Grouped CV then trains on a
    single block and tests on a single block, which produces wild numbers - 100%
    or below-chance - that say nothing about real performance.

    Measured on a 2-session test recording: fbcsp "scored" 100.0% and riemann_svm
    26.6% (below chance) from exactly this. Both were artefacts of having 4 blocks.
    """
    n_blocks = len(np.unique(groups))
    blocks_per_class = min(len(np.unique(groups[y == c])) for c in (0, 1))

    if counts.min() < MIN_TRIALS_PER_CLASS:
        if verbose:
            print(f"\n  NOT ENOUGH DATA - need at least {MIN_TRIALS_PER_CLASS} "
                  f"trials per class, have {counts.min()}.")
            print("  Record another session; they accumulate.")
        return None

    thin_cv = blocks_per_class < MIN_BLOCKS_PER_CLASS
    if thin_cv and verbose:
        print(f"\n  ! only {blocks_per_class} recording block(s) per class "
              f"({n_blocks} total).")
        print("    Cross-validation needs several separate blocks per class to")
        print("    mean anything. The accuracy below is NOT trustworthy - treat")
        print("    it as 'the code ran', not 'the model is this good'.")
        print("    Record more SEPARATE sessions rather than one long one.")

    # --- score every pipeline with leakage-safe grouped CV
    results = {}
    if verbose:
        print(f"\n  {'pipeline':<14}{'accuracy':>18}{'train ms':>11}")
        print("  " + "-" * 43)

    for name, factory in PIPELINES.items():
        accs, t0 = [], time.perf_counter()
        for tr, te in grouped_folds(groups, y, n_splits=n_splits):
            try:
                pipe = factory(sample_freq=fs)
                pipe.fit(X[tr], y[tr])
                accs.append(float(np.mean(pipe.predict(X[te]) == y[te])))
            except Exception:
                continue
        if accs:
            ms = (time.perf_counter() - t0) * 1000 / max(1, n_splits)
            results[name] = {"mean": float(np.mean(accs)),
                             "std": float(np.std(accs)),
                             "folds": len(accs)}
            if verbose:
                print(f"  {name:<14}{np.mean(accs):>11.1%} +/-{np.std(accs):<5.1%}"
                      f"{ms:>10.0f}")

    if not results:
        if verbose:
            print("  every pipeline failed - check the recording")
        return None

    best = max(results, key=lambda k: results[k]["mean"])
    score = results[best]["mean"]

    # --- refit the winner on everything, then save
    model = PIPELINES[best](sample_freq=fs)
    model.fit(X, y)

    udir = root / str(user).lower()
    udir.mkdir(parents=True, exist_ok=True)
    model_path = udir / "model.joblib"
    joblib.dump({
        "kind": "MILocalModel", "version": 1,
        "pipeline": best, "model": model,
        "sample_freq": fs, "montage": data["montage"],
        "n_trials": int(len(X)), "cv_accuracy": score,
        "cv_reliable": not thin_cv, "blocks_per_class": int(blocks_per_class),
        "all_scores": results,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "recordings": data["files"],
    }, model_path)

    (udir / "model_info.json").write_text(json.dumps({
        "pipeline": best, "cv_accuracy": score, "n_trials": int(len(X)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "recordings": data["files"], "all_scores": results,
    }, indent=2))

    if verbose:
        print(f"\n  winner: {best}  at {score:.1%}")
        if thin_cv:
            print("  (unreliable - too few recording blocks, see warning above)")
        else:
            _advise(score, len(X), verbose)
        print(f"  saved -> {model_path}")

    return {"user": str(user), "pipeline": best, "accuracy": score,
            "n_trials": int(len(X)), "path": str(model_path),
            "all_scores": results, "reliable": not thin_cv,
            "blocks_per_class": int(blocks_per_class)}


def _advise(score, n_trials, verbose=True):
    """Say plainly whether this model is good enough to use."""
    if not verbose:
        return
    if score < 0.60:
        print("  VERDICT: near chance. Check electrode contact at C3/C4 before")
        print("           recording more - more bad data will not help.")
    elif score < 0.70:
        print("  VERDICT: weak. Usable for testing, frustrating in real use.")
        print(f"           You have {n_trials} trials; ~100 is where it stabilises.")
    elif score < 0.80:
        print("  VERDICT: workable. Keep sessions short and buttons few.")
    else:
        print("  VERDICT: good for motor imagery.")


# ------------------------------------------------------------------ loading
def load_user_model(user, root=None):
    """Load this user's personal model, or None if they have not trained one."""
    root = Path(root) if root else PROFILES
    p = root / str(user).lower() / "model.joblib"
    if not p.exists():
        return None
    d = joblib.load(p)
    if d.get("kind") != "MILocalModel":
        raise ValueError(f"{p} is not a local MI model")
    return d


def starter_model(root=None):
    """The model shipped with the package, used before a user has trained one.

    Trained offline on public PhysioNet data and bundled as a file - NOT fetched
    from a server. It exists only to avoid a cold start: a first-time user gets
    roughly chance-plus-a-bit instead of nothing while their own data builds up.

    Returns None if the package was built without one.
    """
    p = Path(root) if root else (ROOT / "ML" / "starter_model.joblib")
    if not p.exists():
        return None
    return joblib.load(p)


def get_model(user, root=None):
    """Best available model for this user: personal first, starter as fallback."""
    m = load_user_model(user, root=root)
    if m is not None:
        return m, "personal"
    s = starter_model()
    if s is not None:
        return s, "starter"
    return None, "none"


# --------------------------------------------------------------------- CLI
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Train a personal motor-imagery model, locally")
    ap.add_argument("user", help="user name, e.g. anson")
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--info", action="store_true",
                    help="show the existing model instead of training")
    a = ap.parse_args()

    if a.info:
        m = load_user_model(a.user)
        if m is None:
            print(f"No personal model for {a.user!r} yet.")
            return 1
        print(f"\nModel for {a.user!r}")
        print(f"  pipeline    : {m['pipeline']}")
        print(f"  cv accuracy : {m['cv_accuracy']:.1%}")
        print(f"  trials      : {m['n_trials']}")
        print(f"  trained     : {m['trained_at']}")
        print(f"  recordings  : {', '.join(m['recordings'])}")
        return 0

    r = train_user(a.user, n_splits=a.splits)
    return 0 if r else 1


if __name__ == "__main__":
    raise SystemExit(main())
