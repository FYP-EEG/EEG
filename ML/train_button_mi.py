"""
Author: Anson
Created on: 13/10/2026
Purpose: train an experimental app-specific direct-button MI model locally.

This does NOT replace ML/train_local.py. The existing trainer remains the
supported two-class LEFT/RIGHT motor-imagery path and still writes model.joblib.
This file reads only:

    profiles/<user>/button_mi/<app>/calib_*.npz

and writes only:

    profiles/<user>/button_mi/<app>/button_model.joblib

Button classes are the imagined spatial reach-and-press actions recorded in the
real app UI. Only multi-class-capable Riemannian pipelines are considered; the
project's CSP and FBCSP implementations are binary and are intentionally not
modified.
"""

import argparse
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from ML.button_mi import (BUTTON_LABEL_PREFIX, button_profile_dir,
                          parse_button_label)
from ML.mi_advanced import make_riemannian, make_riemannian_lr

BUTTON_PIPELINES = {
    "riemann_svm": make_riemannian,
    "riemann_lr": make_riemannian_lr,
}
MIN_TRIALS_PER_CLASS = 8
MIN_SESSIONS = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.55


def load_button_data(user, app, root=None):
    """Load direct-button sessions without touching LEFT/RIGHT recordings."""
    directory = button_profile_dir(user, app, root=root)
    files = sorted(glob.glob(str(directory / "calib_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No button-MI recordings in {directory}. Record one with:\n"
            f"  python EEG/data_record.py --file {app} --button-mi "
            f"--subject {user} --trials 90 --cue 4 --board-id 0 "
            f"--serial-port COM4")

    sessions = []
    all_labels = set()
    fs_values, montages = set(), set()
    for path in files:
        with np.load(path, allow_pickle=True) as d:
            X = d["X"].astype(np.float64)
            labels = d["labels"].astype(str)
            mask = np.array([v.startswith(BUTTON_LABEL_PREFIX) for v in labels])
            if not mask.any():
                continue
            X, labels = X[mask], labels[mask]
            if len(X) != len(labels):
                raise ValueError(f"X/labels length mismatch in {path}")
            for label in labels:
                parse_button_label(label)  # validates the persisted contract
            sessions.append((Path(path).name, X, labels))
            all_labels.update(labels.tolist())
            fs_values.add(int(d["fs"]))
            montages.add(str(d["montage"]))

    if not sessions:
        raise ValueError(f"recordings found in {directory}, but none contain button-MI labels")
    if len(fs_values) != 1:
        raise ValueError(f"sessions use different sample rates: {sorted(fs_values)}")
    if len(montages) != 1:
        raise ValueError(f"sessions use different montages: {sorted(montages)}")

    labels_sorted = sorted(all_labels, key=lambda v: parse_button_label(v)[0])
    indices = [parse_button_label(v)[0] for v in labels_sorted]
    names = [parse_button_label(v)[1] for v in labels_sorted]
    if len(indices) != len(set(indices)):
        raise ValueError("two button labels use the same screen index")
    if len(names) != len(set(names)):
        raise ValueError("button names must be unique for direct dispatch")
    label_to_class = {label: i for i, label in enumerate(labels_sorted)}

    Xs, ys, groups, sources = [], [], [], []
    for session_id, (filename, X, labels) in enumerate(sessions):
        # Every app session should expose the same target buttons. Refuse a
        # silently changed layout instead of training mismatched class sets.
        missing = set(labels_sorted) - set(labels.tolist())
        if missing:
            missing_names = [parse_button_label(v)[1] for v in sorted(missing)]
            raise ValueError(f"{filename} is missing buttons: {missing_names}")
        Xs.append(X)
        ys.append(np.array([label_to_class[v] for v in labels], dtype=np.int64))
        groups.append(np.full(len(X), session_id, dtype=np.int64))
        sources.extend([filename] * len(X))

    return {
        "X": np.concatenate(Xs),
        "y": np.concatenate(ys),
        "groups": np.concatenate(groups),
        "class_map": {i: name for i, name in enumerate(names)},
        "stored_labels": labels_sorted,
        "files": [s[0] for s in sessions],
        "sources": sources,
        "fs": fs_values.pop(),
        "montage": montages.pop(),
        "directory": directory,
    }


def session_folds(groups, y, n_splits=3):
    """Session-held-out folds for shuffled, balanced button trials.

    ML.evaluate.grouped_folds assumes every group contains one class, which is
    correct for its old contiguous MI blocks but not for this recorder: here a
    group is a complete session containing every button. GroupKFold is the
    correct split and prevents any trial from the held-out day entering train.
    """
    groups = np.asarray(groups)
    y = np.asarray(y)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return
    splitter = GroupKFold(n_splits=min(int(n_splits), n_groups))
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[train])) == len(np.unique(y)) and \
                len(np.unique(y[test])) == len(np.unique(y)):
            yield train, test


def train_button_user(user, app, root=None, sample_freq=None, n_splits=3,
                      confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                      screen=True, verbose=True):
    """Train and save one user's direct-button model for one application."""
    data = load_button_data(user, app, root=root)

    if screen:
        from ML.quality import screen_trials
        screened = screen_trials(data["X"], data["y"], fs=data["fs"], verbose=False)
        keep = screened["keep"]
        if keep.sum() >= max(8, len(data["class_map"]) * 2) and not keep.all():
            if verbose:
                print("\n  Trial screening")
                print(screened["report"])
            data["X"] = data["X"][keep]
            data["y"] = data["y"][keep]
            data["groups"] = data["groups"][keep]

    X, y, groups = data["X"], data["y"], data["groups"]
    fs = int(sample_freq or data["fs"])
    n_classes = len(data["class_map"])
    counts = np.bincount(y, minlength=n_classes)
    sessions = len(np.unique(groups))
    chance = 1.0 / n_classes

    if n_classes < 2:
        raise ValueError("direct-button MI needs at least two buttons")
    if counts.min() < MIN_TRIALS_PER_CLASS:
        if verbose:
            print(f"\n  NOT ENOUGH DATA - need at least {MIN_TRIALS_PER_CLASS} "
                  f"trials per button, have {counts.min()}.")
        return None

    reliable = sessions >= MIN_SESSIONS
    if verbose:
        print(f"\nTraining direct-button MI for user={user!r}, app={app!r}")
        print(f"  buttons: {', '.join(data['class_map'].values())}")
        print(f"  {len(X)} trials: " + ", ".join(
            f"{data['class_map'][i]}={counts[i]}" for i in range(n_classes)))
        print(f"  {sessions} separate session(s), fs={fs}, chance={chance:.1%}")
        if not reliable:
            print(f"  ! need at least {MIN_SESSIONS} sessions for a trustworthy "
                  "session-held-out score")

    results = {}
    for name, factory in BUTTON_PIPELINES.items():
        scores, predictions, truths = [], [], []
        started = time.perf_counter()
        for train, test in session_folds(groups, y, n_splits=n_splits) or []:
            try:
                model = factory(sample_freq=fs)
                model.fit(X[train], y[train])
                pred = np.asarray(model.predict(X[test]), dtype=int)
                scores.append(float(balanced_accuracy_score(y[test], pred)))
                predictions.extend(pred.tolist())
                truths.extend(y[test].tolist())
            except Exception as exc:
                if verbose:
                    print(f"  {name} fold skipped: {exc}")
        if scores:
            results[name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "folds": len(scores),
                "confusion_matrix": confusion_matrix(
                    truths, predictions, labels=list(range(n_classes))).tolist(),
                "train_ms_per_fold": ((time.perf_counter() - started) * 1000
                                      / len(scores)),
            }

    # With one session there is no honest held-out-session score. Train a model
    # for plumbing tests, but mark the evaluation unavailable and unreliable.
    if results:
        best = max(results, key=lambda name: results[name]["mean"])
        score = results[best]["mean"]
    else:
        best = "riemann_lr"
        score = None
        reliable = False

    model = BUTTON_PIPELINES[best](sample_freq=fs)
    model.fit(X, y)

    directory = data["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "button_model.joblib"
    trained_at = datetime.now().isoformat(timespec="seconds")
    bundle = {
        "kind": "ButtonMILocalModel", "version": 1,
        "mode": "direct_button_mi", "pipeline": best, "model": model,
        "class_map": data["class_map"], "stored_labels": data["stored_labels"],
        "sample_freq": fs, "montage": data["montage"],
        "confidence_threshold": float(confidence_threshold),
        "n_trials": int(len(X)), "n_sessions": int(sessions),
        "cv_accuracy": score, "cv_reliable": bool(reliable and score is not None),
        "chance_accuracy": chance, "all_scores": results,
        "trained_at": trained_at, "recordings": data["files"],
    }
    joblib.dump(bundle, model_path)

    info = {k: v for k, v in bundle.items() if k != "model"}
    (directory / "button_model_info.json").write_text(json.dumps(info, indent=2))

    if verbose:
        print("\n  Session-held-out balanced accuracy")
        if results:
            for name, row in results.items():
                print(f"    {name:<14} {row['mean']:.1%} +/- {row['std']:.1%} "
                      f"({row['folds']} folds)")
            print(f"  winner: {best} at {score:.1%}; chance={chance:.1%}")
            print("  confusion matrix (rows=true, columns=predicted):")
            for row in results[best]["confusion_matrix"]:
                print("   ", row)
        else:
            print("    unavailable: record at least two complete sessions")
        if not bundle["cv_reliable"]:
            print("  VERDICT: experimental only; record at least three separate "
                  "sessions before judging separability.")
        elif score <= chance + 0.10:
            print("  VERDICT: near chance; these button-directed movements are "
                  "not separable enough yet.")
        else:
            print("  VERDICT: above chance in held-out sessions; confirm in live use.")
        print(f"  saved -> {model_path}")

    return {
        "user": str(user), "app": str(app), "pipeline": best,
        "accuracy": score, "chance": chance, "n_trials": int(len(X)),
        "n_sessions": int(sessions), "reliable": bundle["cv_reliable"],
        "path": str(model_path), "all_scores": results,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Train an experimental direct-button motor-imagery model")
    ap.add_argument("user", help="user name, e.g. anson")
    ap.add_argument("--app", required=True,
                    help="example/app key used while recording, e.g. rockpaperscissors")
    ap.add_argument("--root", default=None, help="profiles root (advanced/testing)")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    ap.add_argument("--no-screen", action="store_true")
    args = ap.parse_args()
    result = train_button_user(args.user, args.app, root=args.root,
                               n_splits=args.splits,
                               confidence_threshold=args.threshold,
                               screen=not args.no_screen)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
