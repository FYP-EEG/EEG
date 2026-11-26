"""Leakage-aware public motor-imagery experiment comparisons.

This module deliberately does not publish or replace ``models/public``.  It
compares preprocessing and classical estimators against standardized public
archives and writes evidence to a report through ``tools/compare_public_mi.py``.

The experiment policies are intentionally stricter than the baseline trainer:

* hand is reduced to the common left-hand versus right-hand task;
* feet keeps only sources that contain every target class, then balances every
  source/label cell (this currently excludes one-class BNCI feet trials);
* subject-held-out and leave-one-dataset-out results are reported separately;
* optional per-subject Riemannian alignment uses no labels, but is identified as
  transductive because all epochs from each held-out subject define its reference;
* a frequency recipe is skipped if the archive was already filtered more
  narrowly.  In particular, 8--30 Hz archives cannot test Ofner's 0.3--3 Hz
  movement-related cortical-potential range.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ML.mi_advanced import CSP, TangentSpaceMapper, riemannian_mean
from ML.motor_intent import MI_CATEGORIES, normalize_category
from ML.public_mi import (
    PUBLIC_MI_DATASETS,
    combine_public_archives,
    load_standardized_archive,
    public_archive_path,
    subject_folds,
)

EXPERIMENT_VERSION = 1
HAND_LABELS = ("left_hand", "right_hand")
FEET_LABELS = ("idle", "both_feet")

# Each recipe has one explicit spectral hypothesis.  The fixed-band Riemannian
# baseline, aligned variant, and filter-bank CSP are kept distinct so reports do
# not attribute an alignment gain to the frequency bank (or vice versa).
CATEGORY_EXPERIMENT_RECIPES = {
    "finger": (
        {"id": "sensorimotor_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((8.0, 30.0),), "alignment": False},
        {"id": "sensorimotor_aligned_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((8.0, 30.0),), "alignment": True},
        {"id": "mu_beta_fbcsp_lda", "estimator": "fbcsp_lda",
         "bands_hz": ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0),
                      (20.0, 24.0), (24.0, 30.0)), "alignment": False},
    ),
    "hand": (
        {"id": "left_right_sensorimotor_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((8.0, 30.0),), "alignment": False},
        {"id": "left_right_aligned_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((8.0, 30.0),), "alignment": True},
        {"id": "left_right_mu_beta_fbcsp_lda", "estimator": "fbcsp_lda",
         "bands_hz": ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0),
                      (20.0, 24.0), (24.0, 30.0)), "alignment": False},
    ),
    "upper_limb": (
        {"id": "mrcc_0_3_3_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((0.3, 3.0),), "alignment": False},
        {"id": "low_0_5_8_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((0.5, 8.0),), "alignment": False},
        {"id": "low_0_5_8_aligned_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((0.5, 8.0),), "alignment": True},
        {"id": "low_frequency_fbcsp_lda", "estimator": "fbcsp_lda",
         "bands_hz": ((0.5, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)),
         "alignment": False},
        {"id": "sensorimotor_8_30_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((8.0, 30.0),), "alignment": False},
    ),
    "feet": (
        {"id": "source_complete_sensorimotor_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((8.0, 30.0),), "alignment": False},
        {"id": "source_complete_aligned_riemann_lr", "estimator": "riemann_lr",
         "bands_hz": ((8.0, 30.0),), "alignment": True},
        {"id": "source_complete_mu_beta_fbcsp_lda", "estimator": "fbcsp_lda",
         "bands_hz": ((8.0, 12.0), (12.0, 18.0),
                      (18.0, 24.0), (24.0, 30.0)), "alignment": False},
    ),
}


def _copy_rows(data, indices):
    """Select trials while retaining scalar archive metadata."""
    indices = np.asarray(indices, dtype=int)
    n_trials = len(data["X"])
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == n_trials:
            result[key] = value[indices]
        else:
            result[key] = value
    return result


def _counts(values):
    names, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts)}


def harmonize_experiment_data(data, category, *, random_state=42):
    """Apply the declared label/source policy for one category.

    Feet is not allowed to pool a source that contributes only ``both_feet``
    with another source that alone contributes ``idle``.  Such pooling makes
    label and provider inseparable.  We retain only class-complete sources and
    draw the same number of trials from every source/label cell.
    """
    category = normalize_category(category)
    labels = np.asarray(data["labels"]).astype(str)
    sources = np.asarray(data["source_dataset"]).astype(str)
    before = {
        "trials": len(labels),
        "labels": _counts(labels),
        "sources": _counts(sources),
    }
    policy = "preserve audited canonical labels"
    excluded_sources = []

    if category == "hand":
        target_labels = HAND_LABELS
        keep = np.isin(labels, target_labels)
        data = _copy_rows(data, np.flatnonzero(keep))
        policy = "left_hand versus right_hand only; idle and both_hands excluded"
    elif category == "feet":
        target_labels = FEET_LABELS
        label_keep = np.isin(labels, target_labels)
        candidate_sources = sorted(set(sources[label_keep]))
        complete_sources = [
            source for source in candidate_sources
            if set(labels[(sources == source) & label_keep]) == set(target_labels)
        ]
        excluded_sources = sorted(set(candidate_sources) - set(complete_sources))
        if not complete_sources:
            raise ValueError(
                "feet comparison has no source containing both idle and both_feet; "
                "cross-source class completion would confound provider with label")
        eligible = label_keep & np.isin(sources, complete_sources)
        cell_counts = [
            int(np.sum(eligible & (sources == source) & (labels == label)))
            for source in complete_sources for label in target_labels
        ]
        per_cell = min(cell_counts)
        if per_cell < 1:
            raise ValueError("feet source/label balancing found an empty cell")
        rng = np.random.default_rng(int(random_state))
        selected = []
        for source in complete_sources:
            for label in target_labels:
                candidates = np.flatnonzero(
                    eligible & (sources == source) & (labels == label))
                selected.extend(rng.choice(
                    candidates, size=per_cell, replace=False).tolist())
        data = _copy_rows(data, np.sort(np.asarray(selected, dtype=int)))
        policy = (
            "idle versus both_feet; retain class-complete sources only; "
            "equal trials per source/label cell")
    else:
        target_labels = tuple(sorted(set(labels)))

    final_labels = np.asarray(data["labels"]).astype(str)
    final_sources = np.asarray(data["source_dataset"]).astype(str)
    if len(set(final_labels)) < 2:
        raise ValueError(f"{category} experiment needs at least two labels")
    audit = {
        "policy": policy,
        "target_labels": list(target_labels),
        "before": before,
        "after": {
            "trials": len(final_labels),
            "labels": _counts(final_labels),
            "sources": _counts(final_sources),
        },
        "excluded_incomplete_sources": excluded_sources,
    }
    return data, audit


def archive_frequency_range(paths):
    """Return the common recoverable passband declared by archive provenance."""
    ranges = []
    details = []
    for path in paths:
        row = load_standardized_archive(path)
        declared = row["preprocessing"].get("bandpass_hz")
        if declared is None:
            current = (0.0, float(row["fs"]) / 2.0)
        else:
            if len(declared) != 2:
                raise ValueError(f"invalid bandpass provenance in {path}: {declared!r}")
            current = (float(declared[0]), float(declared[1]))
        ranges.append(current)
        details.append({
            "archive": str(Path(path)),
            "source_dataset": row["source_dataset"],
            "declared_bandpass_hz": list(current),
        })
    if not ranges:
        raise ValueError("at least one archive is required")
    common = (max(value[0] for value in ranges),
              min(value[1] for value in ranges))
    if common[0] >= common[1]:
        raise ValueError(f"archives have no common recoverable passband: {ranges}")
    return common, details


def recipe_frequency_compatible(recipe, available_range, *, tolerance=1e-9):
    low, high = map(float, available_range)
    return all(
        float(band[0]) >= low - tolerance and
        float(band[1]) <= high + tolerance and
        float(band[0]) < float(band[1])
        for band in recipe["bands_hz"])


class OneVsRestFilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter-bank CSP supporting binary and multiclass public categories."""

    def __init__(self, bands=((8.0, 12.0), (12.0, 16.0)), sample_freq=250,
                 n_components=4, reg=0.05, order=4):
        self.bands = bands
        self.sample_freq = sample_freq
        self.n_components = n_components
        self.reg = reg
        self.order = order

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("filter-bank CSP needs at least two classes")
        self.sos_ = [
            butter(int(self.order), tuple(map(float, band)), btype="band",
                   fs=float(self.sample_freq), output="sos")
            for band in self.bands
        ]
        # Binary CSP already represents both ends of the generalized eigenvalue
        # spectrum.  Multiclass data gets one CSP decomposition per one-vs-rest
        # contrast for each band.
        targets = self.classes_[:1] if len(self.classes_) == 2 else self.classes_
        self.csps_ = []
        for sos in self.sos_:
            filtered = sosfiltfilt(sos, X, axis=-1)
            bank = []
            for target in targets:
                binary = (y == target).astype(int)
                bank.append(CSP(
                    n_components=int(self.n_components), reg=float(self.reg)
                ).fit(filtered, binary))
            self.csps_.append(bank)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        features = []
        for sos, bank in zip(self.sos_, self.csps_):
            filtered = sosfiltfilt(sos, X, axis=-1)
            features.extend(csp.transform(filtered) for csp in bank)
        return np.hstack(features)


def _make_estimator(recipe, fs, n_classes):
    bands = tuple(tuple(map(float, band)) for band in recipe["bands_hz"])
    if recipe["estimator"] == "riemann_lr":
        if len(bands) != 1:
            raise ValueError("riemann_lr recipe must declare exactly one band")
        return Pipeline([
            ("tangent_space", TangentSpaceMapper(
                sample_freq=float(fs), band=bands[0], reg=0.05)),
            ("scale", StandardScaler()),
            ("logistic_regression", LogisticRegression(
                max_iter=2000, class_weight="balanced")),
        ])
    if recipe["estimator"] == "fbcsp_lda":
        binary_factor = 1 if int(n_classes) == 2 else int(n_classes)
        feature_count = len(bands) * binary_factor * 4
        return Pipeline([
            ("filter_bank_csp", OneVsRestFilterBankCSP(
                bands=bands, sample_freq=float(fs), n_components=4, reg=0.05)),
            ("select", SelectKBest(f_classif, k=min(24, feature_count))),
            ("lda", LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto")),
        ])
    raise ValueError(f"unknown experiment estimator {recipe['estimator']!r}")


def _regularized_covariance(trial, reg=0.05):
    trial = np.asarray(trial, dtype=float)
    trial = trial - trial.mean(axis=1, keepdims=True)
    covariance = trial @ trial.T / max(1, trial.shape[1] - 1)
    trace = float(np.trace(covariance))
    if trace > 0:
        covariance /= trace
    channels = covariance.shape[0]
    return ((1.0 - reg) * covariance +
            reg * np.eye(channels, dtype=float) / channels)


def _inverse_sqrt_spd(matrix):
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    values = np.maximum(values, 1e-12)
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def riemannian_align_by_group(X, groups, *, reg=0.05):
    """Whiten each subject by its unlabeled geometric-mean covariance.

    This is transductive for held-out subjects: every epoch from that subject is
    used to estimate the reference covariance, but labels are never used.
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups).astype(str)
    if len(X) != len(groups):
        raise ValueError("alignment groups must match trial count")
    result = np.empty_like(X, dtype=float)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        covariances = np.asarray([
            _regularized_covariance(X[index], reg=reg) for index in indices
        ])
        reference = riemannian_mean(covariances)
        whitening = _inverse_sqrt_spd(reference)
        centered = X[indices] - X[indices].mean(axis=2, keepdims=True)
        result[indices] = np.einsum("ij,njt->nit", whitening, centered)
    if not np.isfinite(result).all():
        raise ValueError("Riemannian alignment produced non-finite samples")
    return result


def dataset_folds(source_datasets, labels):
    """Yield class-complete leave-one-dataset-out folds and held-out IDs."""
    sources = np.asarray(source_datasets).astype(str)
    labels = np.asarray(labels).astype(str)
    classes = set(labels)
    for source in np.unique(sources):
        test = np.flatnonzero(sources == source)
        train = np.flatnonzero(sources != source)
        if (len(train) and len(test) and set(labels[train]) == classes and
                set(labels[test]) == classes):
            yield train, test, str(source)


def _evaluate_folds(X, labels, folds, recipe, fs, *, split_name):
    labels = np.asarray(labels).astype(str)
    classes = sorted(set(labels))
    to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([to_index[label] for label in labels], dtype=int)
    scores, truths, predictions, details = [], [], [], []
    for fold_number, fold in enumerate(folds, start=1):
        train, test = fold[:2]
        held_out = fold[2] if len(fold) > 2 else None
        model = _make_estimator(recipe, fs, len(classes))
        model.fit(X[train], y[train])
        predicted = np.asarray(model.predict(X[test]), dtype=int)
        score = float(balanced_accuracy_score(y[test], predicted))
        scores.append(score)
        truths.extend(y[test].tolist())
        predictions.extend(predicted.tolist())
        details.append({
            "fold": int(fold_number),
            "held_out": held_out,
            "train_trials": len(train),
            "test_trials": len(test),
            "balanced_accuracy": score,
        })
    if not scores:
        reason = ("need at least two class-complete datasets"
                  if split_name == "dataset-held-out" else
                  "no class-complete subject-held-out folds")
        return {"available": False, "reason": reason, "folds": 0}
    return {
        "available": True,
        "mean_balanced_accuracy": float(np.mean(scores)),
        "std_balanced_accuracy": float(np.std(scores)),
        "folds": len(scores),
        "fold_details": details,
        "confusion_matrix": confusion_matrix(
            truths, predictions, labels=list(range(len(classes)))).tolist(),
    }


def discover_category_archives(processed_root, category):
    category = normalize_category(category)
    paths = []
    for dataset_id, entry in PUBLIC_MI_DATASETS.items():
        if entry.get("importer") and category in entry["categories"]:
            path = public_archive_path(dataset_id, category, root=processed_root)
            if path.exists():
                paths.append(path)
    return paths


def run_category_experiments(archives, category, *, n_splits=5,
                             random_state=42, verbose=True):
    """Evaluate all declared recipes for one category without writing a model."""
    category = normalize_category(category)
    archives = [Path(path) for path in archives]
    data = combine_public_archives(archives, category=category)
    data, harmonization = harmonize_experiment_data(
        data, category, random_state=random_state)
    available_range, frequency_sources = archive_frequency_range(archives)
    labels = np.asarray(data["labels"]).astype(str)
    subjects = np.asarray(data["subjects"]).astype(str)
    sources = np.asarray(data["source_dataset"]).astype(str)
    subject_split = list(subject_folds(
        subjects, labels, n_splits=int(n_splits)))
    dataset_split = list(dataset_folds(sources, labels))
    aligned = None
    recipe_results = []

    for recipe_template in CATEGORY_EXPERIMENT_RECIPES[category]:
        recipe = {
            **recipe_template,
            "bands_hz": [list(map(float, band))
                         for band in recipe_template["bands_hz"]],
        }
        if not recipe_frequency_compatible(recipe, available_range):
            recipe_results.append({
                **recipe, "status": "skipped",
                "reason": (
                    f"requires {recipe['bands_hz']} Hz but archive provenance "
                    f"only preserves common range {list(available_range)} Hz"),
            })
            if verbose:
                print(f"  SKIP {recipe['id']}: {recipe_results[-1]['reason']}")
            continue
        started = time.perf_counter()
        try:
            if recipe["alignment"]:
                if aligned is None:
                    aligned = riemannian_align_by_group(data["X"], subjects)
                X = aligned
            else:
                X = data["X"]
            subject_result = _evaluate_folds(
                X, labels, subject_split, recipe, data["fs"],
                split_name="subject-held-out")
            dataset_result = _evaluate_folds(
                X, labels, dataset_split, recipe, data["fs"],
                split_name="dataset-held-out")
            result = {
                **recipe, "status": "success",
                "subject_held_out": subject_result,
                "dataset_held_out": dataset_result,
                "seconds": float(time.perf_counter() - started),
            }
            recipe_results.append(result)
            if verbose:
                subject_text = (
                    f"{subject_result['mean_balanced_accuracy']:.1%}"
                    if subject_result["available"] else "unavailable")
                dataset_text = (
                    f"{dataset_result['mean_balanced_accuracy']:.1%}"
                    if dataset_result["available"] else "unavailable")
                print(f"  {recipe['id']}: subjects={subject_text}; "
                      f"datasets={dataset_text}")
        except Exception as exc:  # noqa: BLE001 - preserve other experiment recipes
            recipe_results.append({
                **recipe, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": float(time.perf_counter() - started),
            })
            if verbose:
                print(f"  FAIL {recipe['id']}: {type(exc).__name__}: {exc}")

    successful = [row for row in recipe_results if row["status"] == "success"]
    subject_ranked = [
        row for row in successful if row["subject_held_out"]["available"]
    ]
    dataset_ranked = [
        row for row in successful if row["dataset_held_out"]["available"]
    ]
    return {
        "category": category,
        "archives": [str(path) for path in archives],
        "archive_common_bandpass_hz": list(available_range),
        "archive_frequency_provenance": frequency_sources,
        "harmonization": harmonization,
        "n_subjects": len(set(subjects)),
        "n_datasets": len(set(sources)),
        "labels": sorted(set(labels)),
        "chance_accuracy": 1.0 / len(set(labels)),
        "alignment_protocol": (
            "per-subject unlabeled geometric-mean covariance whitening; "
            "transductive on each held-out subject's complete epoch batch"),
        "recipes": recipe_results,
        "best_subject_recipe": (
            max(subject_ranked, key=lambda row:
                row["subject_held_out"]["mean_balanced_accuracy"])["id"]
            if subject_ranked else None),
        "best_dataset_recipe": (
            max(dataset_ranked, key=lambda row:
                row["dataset_held_out"]["mean_balanced_accuracy"])["id"]
            if dataset_ranked else None),
    }


def run_public_mi_experiments(processed_root, report_root, *, categories=MI_CATEGORIES,
                              n_splits=5, random_state=42, verbose=True):
    """Run report-only comparisons.  Existing public models are never touched."""
    started = datetime.now().isoformat(timespec="seconds")
    report = {
        "kind": "PublicMIExperimentalComparison",
        "version": EXPERIMENT_VERSION,
        "started_at": started,
        "status": "running",
        "current_models_modified": False,
        "processed_root": str(Path(processed_root)),
        "evaluation": {
            "primary_metric": "balanced accuracy",
            "subject_split": "whole subjects held out",
            "dataset_split": "leave one class-complete source dataset out",
            "random_epoch_split": False,
        },
        "categories": {},
    }
    for category in categories:
        category = normalize_category(category)
        archives = discover_category_archives(processed_root, category)
        if not archives:
            report["categories"][category] = {
                "category": category, "status": "missing",
                "reason": "no standardized archives found",
            }
            continue
        if verbose:
            print(f"\n[{category}] archives={len(archives)}")
        report["categories"][category] = {
            "status": "success",
            **run_category_experiments(
                archives, category, n_splits=n_splits,
                random_state=random_state, verbose=verbose),
        }
    report["status"] = "success"
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")

    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_root / f"public_mi_experiment_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (report_root / "latest_experiment.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = report_root / f"public_mi_experiment_{stamp}.csv"
    write_experiment_csv(report, csv_path)
    return report, json_path, csv_path


def write_experiment_csv(report, path):
    fields = (
        "category", "recipe", "status", "estimator", "bands_hz", "alignment",
        "subject_mean_balanced_accuracy", "subject_std", "subject_folds",
        "dataset_mean_balanced_accuracy", "dataset_std", "dataset_folds", "note",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category, result in report.get("categories", {}).items():
            for recipe in result.get("recipes", []):
                subject = recipe.get("subject_held_out", {})
                dataset = recipe.get("dataset_held_out", {})
                note = (recipe.get("reason") or recipe.get("error") or
                        dataset.get("reason") or "")
                writer.writerow({
                    "category": category,
                    "recipe": recipe.get("id"),
                    "status": recipe.get("status"),
                    "estimator": recipe.get("estimator"),
                    "bands_hz": json.dumps(recipe.get("bands_hz")),
                    "alignment": recipe.get("alignment"),
                    "subject_mean_balanced_accuracy":
                        subject.get("mean_balanced_accuracy"),
                    "subject_std": subject.get("std_balanced_accuracy"),
                    "subject_folds": subject.get("folds"),
                    "dataset_mean_balanced_accuracy":
                        dataset.get("mean_balanced_accuracy"),
                    "dataset_std": dataset.get("std_balanced_accuracy"),
                    "dataset_folds": dataset.get("folds"),
                    "note": note,
                })
    return Path(path)
