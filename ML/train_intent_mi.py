"""
Author: Anson
Created on: 19/10/2026
Purpose: train reusable personal motor-intent task-set models locally.

Recordings and models are keyed by the sorted set of physical intentions, not
by application. A model trained for left_index/right_index/right_pinky can be
reused for Rock/Paper/Scissors, menu controls, or another app whose developer
maps those same intentions to different callbacks.
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

from ML.mi_advanced import make_riemannian, make_riemannian_lr
from ML.motor_intent import (IDLE, INTENT_LABEL_PREFIX, MotorIntentMap,
                             assess_intent_set, canonical_intents,
                             general_recording_dir, intent_profile_dir,
                             make_intent_label, parse_intent_label,
                             universal_profile_dir)

INTENT_PIPELINES = {
    "riemann_svm": make_riemannian,
    "riemann_lr": make_riemannian_lr,
}
MIN_TRIALS_PER_CLASS = 8
MIN_SESSIONS = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.55


def load_intent_data(user, intents, root=None, data_root=None):
    wanted = canonical_intents(intents)
    directory = intent_profile_dir(user, wanted, root=root)
    exact_files = sorted(glob.glob(str(directory / "calib_*.npz")))

    # Standalone EEG/data_record.py sessions are generalized training data and
    # never belong to a runtime profile. All physical labels are filtered to
    # this task set here. Personal in-app calibration can still contribute
    # task-specific sessions from the runtime profile directory above.
    general_dir = general_recording_dir(data_root)
    general_files = sorted(glob.glob(str(general_dir / "calib_*.npz")))

    # Backward compatibility for recordings made before the generalized pool.
    universal_dir = universal_profile_dir(user, root=root)
    universal_files = sorted(glob.glob(str(universal_dir / "calib_*.npz")))

    files = [(path, "profile") for path in exact_files]
    files += [(path, "general") for path in general_files]
    files += [(path, "legacy-universal") for path in universal_files]
    # Avoid double-reading if a caller deliberately points roots together.
    deduplicated = []
    seen_paths = set()
    for path, source in files:
        resolved = str(Path(path).resolve())
        if resolved not in seen_paths:
            deduplicated.append((path, source))
            seen_paths.add(resolved)
    files = deduplicated
    if not files:
        raise FileNotFoundError(
            f"No personal calibration recordings in {directory}, no generalized "
            f"training recordings in {general_dir}, and no legacy recordings "
            f"in {universal_dir}.")

    wanted_labels = {make_intent_label(v) for v in wanted}
    sessions, all_labels = [], set()
    fs_values, montages = set(), set()
    for path, source in files:
        with np.load(path, allow_pickle=True) as data:
            X = data["X"].astype(np.float64)
            labels = data["labels"].astype(str)
            mask = np.array([v.startswith(INTENT_LABEL_PREFIX) for v in labels])
            if not mask.any():
                continue
            X, labels = X[mask], labels[mask]
            decoded = [parse_intent_label(v) for v in labels]
            if source in ("general", "legacy-universal"):
                keep = np.array([v in set(wanted) or v == IDLE for v in decoded])
                X, labels = X[keep], labels[keep]
                decoded = [v for v, use in zip(decoded, keep) if use]
                if not len(X):
                    continue
                # General data may come from another application's smaller task
                # set. Only complete sessions are useful for session-held-out
                # evaluation of this requested model; incomplete ones are
                # ignored rather than treated as corrupt.
                if not wanted_labels.issubset(set(labels.tolist())):
                    continue
            else:
                unexpected = set(decoded) - set(wanted) - {IDLE}
                if unexpected:
                    raise ValueError(
                        f"{Path(path).name} contains a different task set: "
                        f"{sorted(unexpected)}")
            display_name = f"{source}/{Path(path).name}"
            sessions.append((display_name, X, labels))
            all_labels.update(labels.tolist())
            fs_values.add(int(data["fs"]))
            montages.add(str(data["montage"]))

    if not sessions:
        raise ValueError(
            "available recordings contain no complete session for requested "
            f"intentions {wanted!r}; checked {directory} and {general_dir}")
    if len(fs_values) != 1:
        raise ValueError(f"sessions use different sample rates: {sorted(fs_values)}")
    if len(montages) != 1:
        raise ValueError(f"sessions use different montages: {sorted(montages)}")

    missing_global = wanted_labels - all_labels
    if missing_global:
        raise ValueError(
            "recordings are missing required intentions: " +
            ", ".join(parse_intent_label(v) for v in sorted(missing_global)))

    class_names = list(wanted)
    idle_label = make_intent_label(IDLE)
    has_idle = idle_label in all_labels
    if has_idle:
        class_names.append(IDLE)
    if len(class_names) < 2:
        raise ValueError(
            "one-intent training needs idle trials; record again with "
            "--idle-trials 10 or more")

    labels_ordered = [make_intent_label(v) for v in class_names]
    to_class = {label: i for i, label in enumerate(labels_ordered)}
    Xs, ys, groups = [], [], []
    used_files = []
    for session_id, (filename, X, labels) in enumerate(sessions):
        present = set(labels.tolist())
        missing = wanted_labels - present
        if missing:
            raise ValueError(
                f"{filename} is missing intentions: " +
                ", ".join(parse_intent_label(v) for v in sorted(missing)))
        if has_idle and idle_label not in present:
            raise ValueError(f"{filename} is missing idle trials used by other sessions")
        keep = np.isin(labels, labels_ordered)
        X, labels = X[keep], labels[keep]
        Xs.append(X)
        ys.append(np.array([to_class[v] for v in labels], dtype=np.int64))
        groups.append(np.full(len(X), session_id, dtype=np.int64))
        used_files.append(filename)

    return {
        "X": np.concatenate(Xs), "y": np.concatenate(ys),
        "groups": np.concatenate(groups),
        "class_map": {i: name for i, name in enumerate(class_names)},
        "intents": wanted, "has_idle": has_idle, "files": used_files,
        "fs": fs_values.pop(), "montage": montages.pop(),
        "directory": directory,
    }


def session_folds(groups, y, n_splits=3):
    groups, y = np.asarray(groups), np.asarray(y)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return
    splitter = GroupKFold(n_splits=min(int(n_splits), n_groups))
    all_classes = len(np.unique(y))
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[train])) == all_classes and \
                len(np.unique(y[test])) == all_classes:
            yield train, test


def train_intent_user(user, intents, root=None, data_root=None,
                      sample_freq=None, n_splits=3,
                      confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                      screen=True, verbose=True):
    wanted = canonical_intents(intents)
    assessment = assess_intent_set(wanted)
    data = load_intent_data(user, wanted, root=root, data_root=data_root)

    if screen:
        from ML.quality import screen_trials
        screened = screen_trials(data["X"], data["y"], fs=data["fs"],
                                  verbose=False)
        keep = screened["keep"]
        min_left = max(8, len(data["class_map"]) * 2)
        if keep.sum() >= min_left and not keep.all():
            if verbose:
                print("\n  Trial screening")
                print(screened["report"])
            for key in ("X", "y", "groups"):
                data[key] = data[key][keep]

    X, y, groups = data["X"], data["y"], data["groups"]
    fs = int(sample_freq or data["fs"])
    n_classes = len(data["class_map"])
    counts = np.bincount(y, minlength=n_classes)
    sessions = len(np.unique(groups))
    chance = 1.0 / n_classes

    if counts.min() < MIN_TRIALS_PER_CLASS:
        if verbose:
            print(f"\n  NOT ENOUGH DATA - need {MIN_TRIALS_PER_CLASS} trials "
                  f"per recorded class, have {counts.min()}.")
        return None

    reliable = sessions >= MIN_SESSIONS
    if verbose:
        print(f"\nTraining reusable motor-intent model for {user!r}")
        print(f"  task set: {', '.join(wanted)}")
        print(f"  difficulty: {assessment['difficulty']}")
        for warning in assessment["warnings"]:
            print(f"  WARNING: {warning}")
        print(f"  {len(X)} trials: " + ", ".join(
            f"{data['class_map'][i]}={counts[i]}" for i in range(n_classes)))
        print(f"  {sessions} session(s), fs={fs}, chance={chance:.1%}")
        if not reliable:
            print(f"  ! need at least {MIN_SESSIONS} separate sessions for a "
                  "trustworthy session-held-out score")

    results = {}
    for name, factory in INTENT_PIPELINES.items():
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
                "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                "folds": len(scores),
                "confusion_matrix": confusion_matrix(
                    truths, predictions, labels=list(range(n_classes))).tolist(),
                "train_ms_per_fold": ((time.perf_counter() - started) * 1000
                                      / len(scores)),
            }

    if results:
        best = max(results, key=lambda name: results[name]["mean"])
        score = results[best]["mean"]
    else:
        best, score, reliable = "riemann_lr", None, False

    model = INTENT_PIPELINES[best](sample_freq=fs)
    model.fit(X, y)

    directory = data["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "intent_model.joblib"
    trained_at = datetime.now().isoformat(timespec="seconds")
    bundle = {
        "kind": "MotorIntentModel", "version": 1,
        "pipeline": best, "model": model, "intents": list(wanted),
        "class_map": data["class_map"], "has_idle": data["has_idle"],
        "sample_freq": fs, "montage": data["montage"],
        "confidence_threshold": float(confidence_threshold),
        "n_trials": int(len(X)), "n_sessions": int(sessions),
        "cv_accuracy": score,
        "cv_reliable": bool(reliable and score is not None),
        "chance_accuracy": chance, "all_scores": results,
        "assessment": assessment, "trained_at": trained_at,
        "recordings": data["files"],
    }
    joblib.dump(bundle, model_path)
    info = {k: v for k, v in bundle.items() if k != "model"}
    (directory / "intent_model_info.json").write_text(json.dumps(info, indent=2))

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
            print("  VERDICT: experimental only; collect at least three sessions.")
        elif score <= chance + 0.10:
            print("  VERDICT: near chance; do not enable this task set live.")
        else:
            print("  VERDICT: above chance across sessions; confirm live and "
                  "measure idle false activations.")
        print(f"  saved -> {model_path}")

    return {
        "user": str(user), "intents": wanted, "task_set": assessment["task_set"],
        "pipeline": best, "accuracy": score, "chance": chance,
        "n_trials": int(len(X)), "n_sessions": int(sessions),
        "reliable": bundle["cv_reliable"], "path": str(model_path),
        "all_scores": results,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Train a reusable motor-intent MI model locally")
    ap.add_argument("user", help="user name, e.g. anson")
    ap.add_argument("--intents", required=True,
                    help="comma-separated task set, e.g. "
                         "left_index,right_index,right_pinky")
    ap.add_argument("--root", default=None,
                    help="runtime profile/model root (default: profiles/)")
    ap.add_argument("--data-root", default=None,
                    help="generalized recording root (default: dataset/motor_intent)")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--threshold", type=float,
                    default=DEFAULT_CONFIDENCE_THRESHOLD)
    ap.add_argument("--no-screen", action="store_true")
    args = ap.parse_args()
    intents = [part.strip() for part in args.intents.split(",") if part.strip()]
    result = train_intent_user(
        args.user, intents, root=args.root, data_root=args.data_root,
        n_splits=args.splits, confidence_threshold=args.threshold,
        screen=not args.no_screen)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
