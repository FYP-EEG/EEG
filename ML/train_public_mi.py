"""Train category-specific public MI benchmarks/pretraining models.

Edited by Anson
Date: 19/11/2026
Purpose: compare classical and optional Transformer models on identical
         subject-held-out splits without presenting public accuracy as Cyton
         or personal motor-imagery validation. Hand training uses the audited
         cross-source task shared by all providers: left hand versus right hand.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from ML.deep_mi import SUPPORTED_DEEP_MODELS, BraindecodeMIClassifier
from ML.mi_advanced import make_riemannian, make_riemannian_lr
from ML.motor_intent import normalize_category
from ML.public_mi import combine_public_archives, public_model_dir, subject_folds
from ML.public_mi_experiments import OneVsRestFilterBankCSP, dataset_folds

PUBLIC_FBCSP_BANDS = (
    (8.0, 12.0), (12.0, 16.0), (16.0, 20.0),
    (20.0, 24.0), (24.0, 30.0),
)


def make_public_fbcsp(sample_freq=250):
    """Eight-channel, 8--30 Hz filter-bank CSP used by public comparisons."""
    return Pipeline([
        ("filter_bank_csp", OneVsRestFilterBankCSP(
            bands=PUBLIC_FBCSP_BANDS, sample_freq=sample_freq,
            n_components=4, reg=0.05)),
        ("select", SelectKBest(f_classif, k=20)),
        ("lda", LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto")),
    ])


CLASSICAL_PIPELINES = {
    "riemann_svm": make_riemannian,
    "riemann_lr": make_riemannian_lr,
    "fbcsp_lda": make_public_fbcsp,
}
PUBLIC_PIPELINES = tuple(CLASSICAL_PIPELINES) + SUPPORTED_DEEP_MODELS


def _factory(name, fs, epochs=40, batch_size=32, random_state=42,
             pretrained_path=None, verbose=False):
    if name in CLASSICAL_PIPELINES:
        if pretrained_path:
            raise ValueError("--pretrained applies only to EEGConformer/TCFormer")
        return CLASSICAL_PIPELINES[name](sample_freq=int(fs))
    return BraindecodeMIClassifier(
        model_name=name, sample_freq=int(fs), max_epochs=int(epochs),
        batch_size=int(batch_size), random_state=int(random_state),
        pretrained_path=pretrained_path, verbose=verbose)


def _counts(values):
    names, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def _select_trials(data, indices):
    trial_keys = {
        "X", "labels", "subjects", "sessions", "source_labels",
        "source_dataset",
    }
    filtered = dict(data)
    for key in trial_keys:
        filtered[key] = np.asarray(data[key])[indices]
    return filtered


def apply_public_training_policy(data, category):
    """Apply only task/source definitions proven safe for public training.

    Hand uses the left/right task shared by all four providers. Feet retains
    only providers containing both idle and both-feet, then deterministically
    balances each source/label cell. This prevents BNCI's feet-only trials from
    making provider identity a shortcut for the feet label.
    """
    category = normalize_category(category)
    labels = np.asarray(data["labels"]).astype(str)
    sources = np.asarray(data["source_dataset"]).astype(str)
    before = _counts(labels)
    before_sources = _counts(sources)

    if category == "hand":
        target = ("left_hand", "right_hand")
        keep = np.isin(labels, target)
        if not keep.any() or set(labels[keep]) != set(target):
            raise ValueError(
                "public hand training requires both left_hand and right_hand")
        indices = np.flatnonzero(keep)
        filtered = _select_trials(data, indices)
        return filtered, {
            "id": "cross-source-left-versus-right-hand-v1",
            "description": (
                "left_hand versus right_hand only; exclude idle and both_hands "
                "to match the personal hand specialist and all four public sources"),
            "target_labels": list(target),
            "before_labels": before,
            "after_labels": _counts(filtered["labels"]),
            "before_sources": before_sources,
            "after_sources": _counts(filtered["source_dataset"]),
            "removed_trials": int(len(labels) - len(indices)),
        }

    if category == "feet":
        target = ("idle", "both_feet")
        label_keep = np.isin(labels, target)
        candidates = sorted(set(sources[label_keep]))
        complete = [
            source for source in candidates
            if set(labels[(sources == source) & label_keep]) == set(target)
        ]
        excluded = sorted(set(candidates) - set(complete))
        if not complete:
            raise ValueError(
                "public feet training needs a source containing both idle and both_feet")
        eligible = label_keep & np.isin(sources, complete)
        per_cell = min(
            int(np.sum(eligible & (sources == source) & (labels == label)))
            for source in complete for label in target)
        rng = np.random.default_rng(42)
        selected = []
        for source in complete:
            for label in target:
                cell = np.flatnonzero(
                    eligible & (sources == source) & (labels == label))
                selected.extend(rng.choice(
                    cell, size=per_cell, replace=False).tolist())
        indices = np.sort(np.asarray(selected, dtype=int))
        filtered = _select_trials(data, indices)
        return filtered, {
            "id": "source-complete-idle-versus-both-feet-v1",
            "description": (
                "idle versus both_feet; retain only class-complete providers; "
                "equal trials per source/label cell"),
            "target_labels": list(target),
            "before_labels": before,
            "after_labels": _counts(filtered["labels"]),
            "before_sources": before_sources,
            "after_sources": _counts(filtered["source_dataset"]),
            "excluded_incomplete_sources": excluded,
            "trials_per_source_label_cell": per_cell,
            "policy_random_state": 42,
            "removed_trials": int(len(labels) - len(indices)),
        }

    return data, {
        "id": "audited-canonical-labels",
        "description": "retain the category's audited canonical labels",
        "before_labels": before,
        "after_labels": before,
        "before_sources": before_sources,
        "after_sources": before_sources,
    }


def _evaluate_folds(data, y, folds, pipeline, *, split_name, epochs,
                    batch_size, random_state, pretrained_path, verbose):
    scores, truths, predictions, details = [], [], [], []
    for fold_index, fold in enumerate(folds, start=1):
        train, test = fold[:2]
        held_out = fold[2] if len(fold) > 2 else None
        seed_offset = fold_index + (1000 if split_name == "dataset" else 0)
        model = _factory(
            pipeline, data["fs"], epochs=epochs, batch_size=batch_size,
            random_state=random_state + seed_offset,
            pretrained_path=pretrained_path, verbose=False)
        model.fit(data["X"][train], y[train])
        predicted = np.asarray(model.predict(data["X"][test]), dtype=int)
        score = float(balanced_accuracy_score(y[test], predicted))
        scores.append(score)
        truths.extend(y[test].tolist())
        predictions.extend(predicted.tolist())
        details.append({
            "fold": fold_index,
            "held_out": held_out,
            "train_trials": len(train),
            "test_trials": len(test),
            "balanced_accuracy": score,
        })
        if verbose:
            suffix = (f"held-out dataset={held_out}" if held_out else
                      f"held-out subjects={len(np.unique(data['subjects'][test]))}")
            print(f"  {split_name} fold {fold_index}/{len(folds)}: "
                  f"{score:.1%} ({suffix})")
    return {
        "scores": scores,
        "truths": truths,
        "predictions": predictions,
        "details": details,
    }


def train_public_category(archives, *, category=None, pipeline="riemann_lr",
                          output_root=None, n_splits=5, epochs=40,
                          batch_size=32, random_state=42,
                          pretrained_path=None, verbose=True):
    """Train one category model with subject and, when possible, dataset CV."""
    if pipeline not in PUBLIC_PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected {PUBLIC_PIPELINES}")
    data = combine_public_archives(archives, category=category)
    category = normalize_category(data["category"])
    data, training_policy = apply_public_training_policy(data, category)
    labels = sorted(set(data["labels"].tolist()))
    if len(labels) < 2:
        raise ValueError(
            f"{category} archive has only {labels}; "
            "classification/pretraining needs at least two classes")
    to_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([to_index[v] for v in data["labels"]], dtype=np.int64)
    subject_split = list(subject_folds(
        data["subjects"], data["labels"], n_splits=n_splits))
    if not subject_split:
        raise ValueError("no class-complete subject-held-out folds; import more subjects")
    dataset_split = list(dataset_folds(
        data["source_dataset"], data["labels"]))

    started = time.perf_counter()
    subject_result = _evaluate_folds(
        data, y, subject_split, pipeline, split_name="subject",
        epochs=epochs, batch_size=batch_size, random_state=random_state,
        pretrained_path=pretrained_path, verbose=verbose)
    dataset_result = (_evaluate_folds(
        data, y, dataset_split, pipeline, split_name="dataset",
        epochs=epochs, batch_size=batch_size, random_state=random_state,
        pretrained_path=pretrained_path, verbose=verbose)
        if dataset_split else None)

    model = _factory(
        pipeline, data["fs"], epochs=epochs, batch_size=batch_size,
        random_state=random_state, pretrained_path=pretrained_path,
        verbose=verbose and pipeline in SUPPORTED_DEEP_MODELS)
    model.fit(data["X"], y)
    directory = public_model_dir(category, root=output_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "category_model.joblib"
    subject_scores = subject_result["scores"]
    dataset_scores = dataset_result["scores"] if dataset_result else []
    bundle = {
        "kind": "PublicMICategoryModel", "version": 2,
        "runtime_ready": False,
        "purpose": "public benchmark/pretraining only; requires personal calibration/fine-tuning for live use",
        "category": category, "pipeline": pipeline, "model": model,
        "class_map": {index: label for label, index in to_index.items()},
        "labels": labels, "sample_freq": data["fs"],
        "channel_labels": data["channel_labels"].tolist(),
        "units": data["units"], "n_times": int(data["X"].shape[-1]),
        "n_trials": len(y), "n_subjects": len(np.unique(data["subjects"])),
        "source_datasets": sorted(set(data["source_dataset"].tolist())),
        "source_trial_counts": _counts(data["source_dataset"]),
        "training_policy": training_policy,
        "evaluation_split": "grouped by subject; no subject in both train and test",
        "cv_balanced_accuracy": float(np.mean(subject_scores)),
        "cv_std": float(np.std(subject_scores)),
        "cv_folds": len(subject_scores),
        "cv_fold_details": subject_result["details"],
        "chance_accuracy": 1.0 / len(labels),
        "confusion_matrix": confusion_matrix(
            subject_result["truths"], subject_result["predictions"],
            labels=list(range(len(labels)))).tolist(),
        "dataset_evaluation_split": (
            "leave one class-complete source dataset out" if dataset_result else None),
        "dataset_cv_balanced_accuracy": (
            float(np.mean(dataset_scores)) if dataset_scores else None),
        "dataset_cv_std": float(np.std(dataset_scores)) if dataset_scores else None,
        "dataset_cv_folds": len(dataset_scores),
        "dataset_cv_fold_details": (
            dataset_result["details"] if dataset_result else []),
        "dataset_confusion_matrix": (
            confusion_matrix(
                dataset_result["truths"], dataset_result["predictions"],
                labels=list(range(len(labels)))).tolist()
            if dataset_result else []),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_seconds": float(time.perf_counter() - started),
        "archives": [str(Path(value)) for value in archives],
        "pretrained_from": str(pretrained_path) if pretrained_path else None,
    }
    if isinstance(model, BraindecodeMIClassifier):
        bundle["transfer_report"] = model.transfer_summary()
    joblib.dump(bundle, path)
    info = {key: value for key, value in bundle.items() if key != "model"}
    (directory / "category_model_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    if verbose:
        print(f"\nSubject-held-out balanced accuracy: "
              f"{np.mean(subject_scores):.1%} +/- {np.std(subject_scores):.1%}; "
              f"chance={1 / len(labels):.1%}")
        if dataset_scores:
            print(f"Dataset-held-out balanced accuracy: "
                  f"{np.mean(dataset_scores):.1%} +/- {np.std(dataset_scores):.1%}")
        else:
            print("Dataset-held-out evaluation unavailable: need at least two "
                  "class-complete source datasets.")
        print("This is public-dataset benchmark performance, not personal Cyton validation.")
        print(f"saved -> {path}")
    return {
        "path": str(path), "category": category, "pipeline": pipeline,
        "accuracy": float(np.mean(subject_scores)),
        "std": float(np.std(subject_scores)),
        "dataset_accuracy": (
            float(np.mean(dataset_scores)) if dataset_scores else None),
        "dataset_std": float(np.std(dataset_scores)) if dataset_scores else None,
        "dataset_folds": len(dataset_scores),
        "chance": 1.0 / len(labels), "folds": len(subject_scores),
        "n_subjects": bundle["n_subjects"], "runtime_ready": False,
        "labels": labels, "training_policy": training_policy,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a public category MI benchmark/pretraining model")
    parser.add_argument("archives", nargs="+", help="standardized public epochs.npz files")
    parser.add_argument("--category", default=None)
    parser.add_argument("--pipeline", choices=PUBLIC_PIPELINES, default="riemann_lr")
    parser.add_argument("--output-root", default=None,
                        help="default models/public; supplied root contains category dirs")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", default=None,
                        help="optional compatible PublicMICategoryModel for deep weight initialization")
    args = parser.parse_args(argv)
    train_public_category(
        args.archives, category=args.category, pipeline=args.pipeline,
        output_root=args.output_root, n_splits=args.splits,
        epochs=args.epochs, batch_size=args.batch_size,
        random_state=args.seed, pretrained_path=args.pretrained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
