"""Train category-specific public MI benchmarks/pretraining models.

Edited by Anson
Date: 10/11/2026
Purpose: compare classical and optional Transformer models on identical
         subject-held-out splits without presenting public accuracy as Cyton
         or personal motor-imagery validation.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from ML.deep_mi import SUPPORTED_DEEP_MODELS, BraindecodeMIClassifier
from ML.mi_advanced import make_riemannian, make_riemannian_lr
from ML.motor_intent import normalize_category
from ML.public_mi import combine_public_archives, public_model_dir, subject_folds

CLASSICAL_PIPELINES = {
    "riemann_svm": make_riemannian,
    "riemann_lr": make_riemannian_lr,
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


def train_public_category(archives, *, category=None, pipeline="riemann_lr",
                          output_root=None, n_splits=5, epochs=40,
                          batch_size=32, random_state=42,
                          pretrained_path=None, verbose=True):
    """Train one category model; evaluation always holds out whole subjects."""
    if pipeline not in PUBLIC_PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}; expected {PUBLIC_PIPELINES}")
    data = combine_public_archives(archives, category=category)
    category = normalize_category(data["category"])
    labels = sorted(set(data["labels"].tolist()))
    if len(labels) < 2:
        raise ValueError(
            f"{category} archive has only {labels}; classification/pretraining needs at least two classes")
    to_index = {label: index for index, label in enumerate(labels)}
    y = np.asarray([to_index[v] for v in data["labels"]], dtype=np.int64)
    folds = list(subject_folds(data["subjects"], data["labels"], n_splits=n_splits))
    if not folds:
        raise ValueError("no class-complete subject-held-out folds; import more subjects")

    scores, truths, predictions = [], [], []
    started = time.perf_counter()
    for fold_index, (train, test) in enumerate(folds, start=1):
        model = _factory(
            pipeline, data["fs"], epochs=epochs, batch_size=batch_size,
            random_state=random_state + fold_index,
            pretrained_path=pretrained_path, verbose=False)
        model.fit(data["X"][train], y[train])
        predicted = np.asarray(model.predict(data["X"][test]), dtype=int)
        scores.append(float(balanced_accuracy_score(y[test], predicted)))
        truths.extend(y[test].tolist()); predictions.extend(predicted.tolist())
        if verbose:
            print(f"  fold {fold_index}/{len(folds)}: {scores[-1]:.1%} "
                  f"(held-out subjects={len(np.unique(data['subjects'][test]))})")

    model = _factory(
        pipeline, data["fs"], epochs=epochs, batch_size=batch_size,
        random_state=random_state, pretrained_path=pretrained_path,
        verbose=verbose and pipeline in SUPPORTED_DEEP_MODELS)
    model.fit(data["X"], y)
    directory = public_model_dir(category, root=output_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "category_model.joblib"
    bundle = {
        "kind": "PublicMICategoryModel", "version": 1,
        "runtime_ready": False,
        "purpose": "public benchmark/pretraining only; requires personal calibration/fine-tuning for live use",
        "category": category, "pipeline": pipeline, "model": model,
        "class_map": {index: label for label, index in to_index.items()},
        "labels": labels, "sample_freq": data["fs"],
        "channel_labels": data["channel_labels"].tolist(),
        "units": data["units"], "n_times": int(data["X"].shape[-1]),
        "n_trials": len(y), "n_subjects": len(np.unique(data["subjects"])),
        "source_datasets": sorted(set(data["source_dataset"].tolist())),
        "evaluation_split": "grouped by subject; no subject in both train and test",
        "cv_balanced_accuracy": float(np.mean(scores)),
        "cv_std": float(np.std(scores)), "cv_folds": len(scores),
        "chance_accuracy": 1.0 / len(labels),
        "confusion_matrix": confusion_matrix(
            truths, predictions, labels=list(range(len(labels)))).tolist(),
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
        print(f"\nSubject-held-out balanced accuracy: {np.mean(scores):.1%} "
              f"+/- {np.std(scores):.1%}; chance={1 / len(labels):.1%}")
        print("This is public-dataset benchmark performance, not personal Cyton validation.")
        print(f"saved -> {path}")
    return {"path": str(path), "category": category, "pipeline": pipeline,
            "accuracy": float(np.mean(scores)), "std": float(np.std(scores)),
            "chance": 1.0 / len(labels), "folds": len(scores),
            "n_subjects": bundle["n_subjects"], "runtime_ready": False}


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
