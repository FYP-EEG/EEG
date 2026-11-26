"""Reproducible public motor-imagery evidence report and visualizations.

All numeric project results generated here come from complete-subject or
complete-source held-out folds.  Class-separation figures use one declared
representative held-out fold: CSP/tangent/PCA objects are fitted on that fold's
training subjects, then only transform held-out subjects for the colored points.
The plots are descriptive and are never presented as an extra accuracy metric.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from ML.mi_advanced import TangentSpaceMapper, make_riemannian, make_riemannian_lr
from ML.motor_intent import MI_CATEGORIES, normalize_category
from ML.public_mi import (
    PUBLIC_MI_DATASETS,
    combine_public_archives,
    load_standardized_archive,
    subject_folds,
)
from ML.public_mi_experiments import OneVsRestFilterBankCSP, dataset_folds
from ML.train_public_mi import (
    PUBLIC_FBCSP_BANDS,
    apply_public_training_policy,
    make_public_fbcsp,
)

EVIDENCE_VERSION = 1
DEFAULT_PIPELINES = {
    "finger": "riemann_lr",
    "hand": "fbcsp_lda",
    "upper_limb": "fbcsp_lda",
    "feet": "riemann_svm",
}

# These values preserve the completed project's prior audited run even when the
# large archives/model bundles are not copied into a lightweight source tree.
# They are always labelled historical and never substituted for recomputation.
HISTORICAL_PROJECT_RESULTS = {
    "openneuro_ds008446": (
        "20.467% ± 2.579 percentage points SH-BA; 5-class finger; "
        "5 subjects/1,500 trials; Riemannian LR; chance 20%"),
    "physionet_eegmmidb": (
        "61.281% ± 3.959 pp SH-BA for idle vs both feet (20 subjects/898 trials, "
        "Riemannian SVM); leave-PhysioNet-out hand DH-BA 60.652% (900 trials)"),
    "bnci2014_001": (
        "67.014% leave-BNCI-out DH-BA for binary left vs right hand (1,440 trials); "
        "BNCI feet excluded from idle-vs-feet because it has no harmonized idle"),
    "ofner2017": (
        "18.667% ± 1.092 pp SH-BA; 7 classes including idle; "
        "5 subjects/2,100 trials; FBCSP-LDA; chance 14.286%"),
    "cho2017": "65.400% leave-Cho-out DH-BA for binary hand (1,000 trials)",
    "lee2019_mi": "67.200% leave-OpenBMI-out DH-BA for binary hand (1,000 trials)",
    "hand_overall": (
        "66.213% ± 5.846 pp SH-BA and 65.066% ± 2.643 pp DH-BA; "
        "35 subjects/4,340 trials; FBCSP-LDA"),
}

LITERATURE_COMPARISONS = {
    "openneuro_ds008446": {
        "task": "Five unsided finger imagery classes; 20 participants; 62 scalp EEG; 512 Hz.",
        "result": "No authoritative final numeric benchmark is exposed by the release or associated FeatureStability README; do not invent one.",
        "metric": "Not available",
        "protocol": "Associated code describes condition-specific nested three-fold RBF-SVM analysis.",
        "references": [
            "https://openneuro.org/datasets/ds008446/versions/1.0.1",
            "https://github.com/lauramdamm/FeatureStability",
        ],
    },
    "physionet_eegmmidb": {
        "task": "Imagined left/right fist and both fists/both feet; 109 volunteers; 64 EEG; 160 Hz.",
        "result": "Cho et al. associated reanalysis: 60.42% ± 11.68 percentage points.",
        "metric": "Ordinary accuracy",
        "protocol": "Within-subject CSP+FLDA over 109 subjects; not the project's cross-subject/source BA.",
        "references": [
            "https://physionet.org/content/eegmmidb/1.0.0/",
            "https://academic.oup.com/gigascience/article/6/7/gix034/3796323",
        ],
    },
    "bnci2014_001": {
        "task": "Four-class left hand/right hand/feet/tongue imagery; 9 subjects; 22 EEG; 250 Hz.",
        "result": "BCI Competition IV 2a winner: mean Cohen's κ approximately 0.57.",
        "metric": "Cohen's kappa, not accuracy",
        "protocol": "Subject-specific train/evaluation sessions; four-class FBCSP multiclass system.",
        "references": [
            "https://www.bbci.de/competition/iv/results/",
            "https://doi.org/10.3389/fnins.2012.00039",
        ],
    },
    "ofner2017": {
        "task": "Six right upper-limb movement imagery classes plus rest; 15 subjects; 61 EEG; 512 Hz.",
        "result": "27% six-movement accuracy; 73% imagined movement-versus-rest accuracy.",
        "metric": "Ordinary accuracy",
        "protocol": "Within-subject 0.3–3 Hz time-domain/MRCP processing; the default 1–40 Hz archive still omits 0.3–1 Hz.",
        "references": [
            "https://doi.org/10.1371/journal.pone.0182578",
            "https://pubmed.ncbi.nlm.nih.gov/28797109/",
        ],
    },
    "cho2017": {
        "task": "Binary left/right kinesthetic hand imagery; 52 participants; 64 EEG; 512 Hz.",
        "result": "67.46% ± 13.17 percentage points over 50 usable subjects.",
        "metric": "Ordinary accuracy",
        "protocol": "Within-subject CSP+FLDA with repeated random 70/30 subset splits.",
        "references": [
            "https://doi.org/10.1093/gigascience/gix034",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC5493744/",
        ],
    },
    "lee2019_mi": {
        "task": "Binary left/right hand grasp imagery; 54 subjects; 62 EEG; 1,000 Hz.",
        "result": "Mean MI accuracy 71.1% (SD about 15 percentage points); session means around 70.1% and 72.2%.",
        "metric": "Ordinary accuracy",
        "protocol": "Subject-specific CSP-family offline/online and session analyses.",
        "references": [
            "https://doi.org/10.1093/gigascience/giz002",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/",
        ],
    },
}


def _git_commit(root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def archive_fingerprint(path, *, content_hash=False):
    path = Path(path)
    row = load_standardized_archive(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path) if content_hash else None,
        "source_dataset": row["source_dataset"],
        "category": row["category"],
        "schema_version": int(row["schema_version"]),
        "shape": list(map(int, row["X"].shape)),
        "subjects": len(set(row["subjects"].tolist())),
        "sessions": len(set(row["sessions"].tolist())),
        "labels": {str(name): int(count) for name, count in zip(
            *np.unique(row["labels"], return_counts=True))},
        "fs": float(row["fs"]),
        "units": row["units"],
        "channel_labels": row["channel_labels"].tolist(),
        "preprocessing": row["preprocessing"],
        "citation": row["citation"],
        "license": row["license"],
    }


def _make_pipeline(name, fs):
    if name == "fbcsp_lda":
        return make_public_fbcsp(sample_freq=fs)
    if name == "riemann_lr":
        return make_riemannian_lr(sample_freq=fs)
    if name == "riemann_svm":
        return make_riemannian(sample_freq=fs)
    raise ValueError(
        f"evidence report supports classical grouped pipelines only, got {name!r}")


def _evaluate(data, pipeline_name, n_splits):
    labels = np.asarray(data["labels"]).astype(str)
    classes = sorted(set(labels))
    to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([to_index[label] for label in labels], dtype=int)
    subject_split = list(subject_folds(
        data["subjects"], labels, n_splits=n_splits))
    if not subject_split:
        raise ValueError("no class-complete complete-subject folds")
    sources = np.asarray(data["source_dataset"]).astype(str)
    dataset_split = list(dataset_folds(sources, labels))

    def run(folds, split_name):
        scores, truths, predictions, details = [], [], [], []
        for number, fold in enumerate(folds, 1):
            train, test = fold[:2]
            held_out = (fold[2] if len(fold) > 2 else
                        sorted(set(np.asarray(data["subjects"])[test].tolist())))
            model = _make_pipeline(pipeline_name, data["fs"])
            model.fit(data["X"][train], y[train])
            predicted = np.asarray(model.predict(data["X"][test]), dtype=int)
            score = float(balanced_accuracy_score(y[test], predicted))
            scores.append(score)
            truths.extend(y[test].tolist())
            predictions.extend(predicted.tolist())
            details.append({
                "fold": number, "held_out": held_out,
                "train_trials": len(train), "test_trials": len(test),
                "balanced_accuracy": score,
            })
        if not scores:
            return {"available": False, "reason": f"no {split_name} folds"}
        return {
            "available": True,
            "mean_balanced_accuracy": float(np.mean(scores)),
            "std_balanced_accuracy": float(np.std(scores)),
            "scores": scores, "fold_details": details,
            "truths": truths, "predictions": predictions,
            "confusion_matrix": confusion_matrix(
                truths, predictions, labels=list(range(len(classes)))).tolist(),
        }

    return {
        "classes": classes, "y": y,
        "subject_folds": subject_split,
        "subject_held_out": run(subject_split, "subject-held-out"),
        "dataset_held_out": run(dataset_split, "dataset-held-out"),
    }


def _pca_coordinates(train_features, test_features, seed):
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)
    components = min(2, train_scaled.shape[0], train_scaled.shape[1])
    pca = PCA(n_components=components, random_state=seed).fit(train_scaled)
    coordinates = pca.transform(test_scaled)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(coordinates))])
    return coordinates, pca.explained_variance_ratio_.tolist()


def _plot_scatter(path, coordinates, labels, classes, title, subtitle):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for index, label in enumerate(classes):
        mask = labels == label
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=38,
                   alpha=.78, color=cmap(index % 10), label=label,
                   edgecolors="white", linewidths=.35)
        if mask.any():
            center = coordinates[mask].mean(axis=0)
            ax.scatter(center[0], center[1], s=145, marker="X",
                       color=cmap(index % 10), edgecolors="black", linewidths=.8)
    ax.axhline(0, color="#aaaaaa", linewidth=.6, alpha=.5)
    ax.axvline(0, color="#aaaaaa", linewidth=.6, alpha=.5)
    ax.set_title(title)
    ax.text(.5, 1.01, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9, color="#444444")
    ax.set_xlabel("Training-fitted PCA component 1")
    ax.set_ylabel("Training-fitted PCA component 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_confusion(path, matrix, classes, title):
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=int)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float),
                           where=row_sum != 0)
    fig, ax = plt.subplots(figsize=(max(6, .85 * len(classes) + 3),
                                    max(5, .75 * len(classes) + 2)))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(classes)):
        for column in range(len(classes)):
            color = "white" if normalized[row, column] > .55 else "black"
            ax.text(column, row,
                    f"{matrix[row, column]}\n{normalized[row, column]:.0%}",
                    ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_folds(path, details, chance, title):
    import matplotlib.pyplot as plt

    scores = [row["balanced_accuracy"] for row in details]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(range(1, len(scores) + 1), scores, color="#2a9d8f")
    ax.axhline(chance, color="#d1495b", linestyle="--", label=f"chance {chance:.1%}")
    ax.axhline(np.mean(scores), color="#264653", linestyle=":",
               label=f"mean {np.mean(scores):.1%}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Complete-subject fold")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_covariances(path, covariances, labels, classes, channel_labels, title):
    import matplotlib.pyplot as plt

    means = []
    for label in classes:
        values = covariances[labels == label]
        means.append(values.mean(axis=0) if len(values) else
                     np.zeros(covariances.shape[1:]))
    vmax = max(float(np.max(np.abs(value))) for value in means) or 1.0
    columns = min(4, len(classes))
    rows = int(np.ceil(len(classes) / columns))
    fig, axes = plt.subplots(rows, columns,
                             figsize=(4.0 * columns, 3.7 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(classes):
            ax.axis("off")
            continue
        image = ax.imshow(means[index], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(classes[index], fontsize=9)
        ax.set_xticks(range(len(channel_labels)), channel_labels,
                      rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(channel_labels)), channel_labels, fontsize=7)
    fig.suptitle(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.72,
                 label="Trace-normalized regularized covariance")
    fig.subplots_adjust(left=.06, right=.94, bottom=.12, top=.88, wspace=.28, hspace=.38)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _representative_figures(data, evaluation, output_dir, category, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = evaluation["subject_folds"][0]
    labels = np.asarray(data["labels"]).astype(str)
    classes = evaluation["classes"]
    test_labels = labels[test]
    y = evaluation["y"]
    fs = float(data["fs"])
    figures = []

    csp = OneVsRestFilterBankCSP(
        bands=PUBLIC_FBCSP_BANDS, sample_freq=fs, n_components=4, reg=.05)
    csp.fit(data["X"][train], y[train])
    train_csp = csp.transform(data["X"][train])
    test_csp = csp.transform(data["X"][test])
    csp_coordinates, csp_variance = _pca_coordinates(
        train_csp, test_csp, seed)
    csp_path = output_dir / f"{category}_csp_class_separation.png"
    _plot_scatter(
        csp_path, csp_coordinates, test_labels, classes,
        f"{category}: held-out CSP/FBCSP class relationships",
        "Colored points are held-out subjects; CSP, scaling and PCA fit training subjects only")
    figures.append({
        "kind": "csp_class_separation", "path": str(csp_path),
        "classes": classes, "class_colors": "matplotlib tab10",
        "pca_explained_variance_ratio": csp_variance,
    })

    tangent = TangentSpaceMapper(sample_freq=fs, band=(8.0, 30.0), reg=.05)
    tangent.fit(data["X"][train], y[train])
    train_tangent = tangent.transform(data["X"][train])
    test_tangent = tangent.transform(data["X"][test])
    tangent_coordinates, tangent_variance = _pca_coordinates(
        train_tangent, test_tangent, seed)
    tangent_path = output_dir / f"{category}_tangent_pca.png"
    _plot_scatter(
        tangent_path, tangent_coordinates, test_labels, classes,
        f"{category}: held-out 36-feature tangent-space projection",
        "Fréchet mean, feature scaling and PCA fit training subjects only")
    figures.append({
        "kind": "tangent_pca", "path": str(tangent_path),
        "tangent_features": int(test_tangent.shape[1]),
        "pca_explained_variance_ratio": tangent_variance,
    })

    test_covariances = tangent._covs(data["X"][test])
    covariance_path = output_dir / f"{category}_class_covariances.png"
    _plot_covariances(
        covariance_path, test_covariances, test_labels, classes,
        data["channel_labels"].tolist(),
        f"{category}: held-out class covariance summaries")
    figures.append({"kind": "class_covariance_summary",
                    "path": str(covariance_path)})

    held_subjects = sorted(set(np.asarray(data["subjects"])[test].tolist()))
    return figures, {
        "fold": 1,
        "training_subjects": sorted(set(np.asarray(data["subjects"])[train].tolist())),
        "held_out_subjects": held_subjects,
        "train_trials": len(train), "test_trials": len(test),
        "fit_boundary": (
            "CSP, tangent mean, StandardScaler and PCA use training indices only; "
            "colored points/covariance summaries use held-out indices"),
    }


def _model_info(model_root, category):
    path = Path(model_root) / category / "category_model_info.json"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "path": str(path),
                "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "path": str(path.resolve()), "metadata": payload}


def _markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ")
                  for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def _format_reported_counts(value):
    if not value:
        return "Not reported by the authoritative release"
    parts = [f"{key}={count}" for key, count in value.items()
             if key != "reported_n"]
    reported = value.get("reported_n")
    return ", ".join(parts) + (f" (n={reported})" if reported is not None else "")


def _format_age(value):
    if not value:
        return "Not reported by the authoritative release"
    parts = []
    if value.get("mean_years") is not None:
        mean = f"mean {value['mean_years']:g}"
        if value.get("std_years") is not None:
            mean += f" ± {value['std_years']:g}"
        parts.append(mean + " years")
    if value.get("range_years") is not None:
        parts.append(
            f"range {value['range_years'][0]:g}–{value['range_years'][1]:g} years")
    if value.get("reported_n") is not None:
        parts.append(f"n={value['reported_n']}")
    return "; ".join(parts) or "Not reported by the authoritative release"


def _source_rows():
    rows = []
    for dataset_id, literature in LITERATURE_COMPARISONS.items():
        registry = PUBLIC_MI_DATASETS[dataset_id]
        cohort = registry["cohort"]
        rows.append({
            "Dataset": dataset_id,
            "Authoritative source": registry["download"],
            "Participants": cohort["participants"],
            "Population": cohort["population"],
            "Sex/gender as reported": _format_reported_counts(
                cohort["sex_or_gender_as_reported"]),
            "Age": _format_age(cohort["age"]),
            "Handedness": _format_reported_counts(cohort["handedness"]),
            "Cohort completeness": cohort["completeness_note"],
            "Source trial design": cohort["source_trial_design"],
            "Cohort references": " ; ".join(cohort["references"]),
            "Source protocol": literature["task"],
            "Published result": literature["result"],
            "Published metric": literature["metric"],
            "Published validation": literature["protocol"],
            "References": " ; ".join(literature["references"]),
            "Historical project result": HISTORICAL_PROJECT_RESULTS[dataset_id],
        })
    return rows


def _archive_inventory_rows(report):
    """One row per expected dataset/category, including explicit missing rows."""
    fingerprints = {}
    for category_row in report["categories"].values():
        for archive in category_row.get("archives", []):
            fingerprints[(archive["source_dataset"], archive["category"])] = archive
    rows = []
    for dataset_id, registry in PUBLIC_MI_DATASETS.items():
        if not registry.get("importer"):
            continue
        for category in registry["categories"]:
            archive = fingerprints.get((dataset_id, category))
            if archive is None:
                rows.append({
                    "Dataset": dataset_id, "Category": category,
                    "Status": "missing", "Source participants": registry["subjects"],
                    "Imported participants": "", "Sessions": "", "Trials": "",
                    "Channels": "", "Samples per trial": "",
                    "Total channel samples": "", "FS Hz": "",
                    "Nominal epoch seconds": "", "Archive size bytes": "",
                    "Archive path": "", "Preprocessing": "",
                })
                continue
            trials, channels, samples = archive["shape"]
            rows.append({
                "Dataset": dataset_id, "Category": category,
                "Status": "available", "Source participants": registry["subjects"],
                "Imported participants": archive["subjects"],
                "Sessions": archive["sessions"], "Trials": trials,
                "Channels": channels, "Samples per trial": samples,
                "Total channel samples": trials * channels * samples,
                "FS Hz": archive["fs"],
                "Nominal epoch seconds": samples / archive["fs"],
                "Archive size bytes": archive["size_bytes"],
                "Archive path": archive["path"],
                "Preprocessing": json.dumps(
                    archive["preprocessing"], sort_keys=True),
            })
    return rows


def _write_dataset_csv(path, rows):
    columns = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _relpath(path, base):
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(Path(path))


def generate_public_mi_evidence(processed_root, report_root, *,
                                model_root, categories=MI_CATEGORIES,
                                n_splits=5, seed=42, pipelines=None,
                                content_hash=False, inventory_only=False,
                                command=None, project_root=None):
    """Generate report, JSON, comparison CSV and truthful archive-backed plots."""
    processed_root = Path(processed_root)
    report_root = Path(report_root)
    model_root = Path(model_root)
    project_root = Path(project_root or Path(__file__).resolve().parents[1])
    report_root.mkdir(parents=True, exist_ok=True)
    figure_root = report_root / "figures"
    pipelines = dict(pipelines or {})
    report = {
        "kind": "PublicMIEvidenceReport",
        "version": EVIDENCE_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(project_root),
        "command": command,
        "processed_root": str(processed_root.resolve()),
        "model_root": str(model_root.resolve()),
        "evaluation": {
            "primary_metric": "balanced accuracy",
            "subject_split": "complete subjects held out",
            "dataset_split": "leave one class-complete source dataset out",
            "random_epoch_split": False,
            "n_splits": int(n_splits), "random_seed": int(seed),
        },
        "historical_results_source": (
            "reports/public_mi/IMPLEMENTATION_AND_VALIDATION_AUDIT.md; "
            "historical only when archive/model artifacts are absent"),
        "historical_project_results": HISTORICAL_PROJECT_RESULTS,
        "literature": LITERATURE_COMPARISONS,
        "cohorts": {
            dataset_id: PUBLIC_MI_DATASETS[dataset_id]["cohort"]
            for dataset_id in LITERATURE_COMPARISONS
        },
        "categories": {},
    }

    for category_value in categories:
        category = normalize_category(category_value)
        archive_paths = []
        for dataset_id, entry in PUBLIC_MI_DATASETS.items():
            if entry.get("importer") and category in entry["categories"]:
                candidate = processed_root / dataset_id / category / "epochs.npz"
                if candidate.exists():
                    archive_paths.append(candidate)
        category_report = {
            "category": category,
            "pipeline": pipelines.get(category, DEFAULT_PIPELINES[category]),
            "archives": [],
            "model_info": _model_info(model_root, category),
        }
        if not archive_paths:
            category_report.update({
                "status": "missing",
                "reason": "no standardized archive found; no plots or recomputed metrics generated",
            })
            report["categories"][category] = category_report
            continue
        category_report["archives"] = [archive_fingerprint(
            path, content_hash=content_hash) for path in archive_paths]
        if inventory_only:
            category_report.update({
                "status": "inventory_only",
                "reason": "archive metadata recorded; evaluation disabled by --inventory-only",
            })
            report["categories"][category] = category_report
            continue

        try:
            data = combine_public_archives(archive_paths, category=category)
            data, policy = apply_public_training_policy(data, category)
            evaluation = _evaluate(
                data, category_report["pipeline"], n_splits=n_splits)
            subject = evaluation["subject_held_out"]
            dataset = evaluation["dataset_held_out"]
            figures, figure_protocol = _representative_figures(
                data, evaluation, figure_root, category, seed)
            confusion_path = figure_root / f"{category}_subject_confusion.png"
            _plot_confusion(
                confusion_path, subject["confusion_matrix"], evaluation["classes"],
                f"{category}: grouped out-of-fold confusion matrix")
            figures.append({"kind": "subject_oof_confusion",
                            "path": str(confusion_path)})
            fold_path = figure_root / f"{category}_subject_fold_scores.png"
            chance = 1.0 / len(evaluation["classes"])
            _plot_folds(
                fold_path, subject["fold_details"], chance,
                f"{category}: complete-subject balanced accuracy")
            figures.append({"kind": "subject_fold_scores", "path": str(fold_path)})
            category_report.update({
                "status": "success",
                "training_policy": policy,
                "labels": evaluation["classes"],
                "chance_balanced_accuracy": chance,
                "n_trials": len(data["X"]),
                "n_subjects": len(set(data["subjects"].tolist())),
                "n_datasets": len(set(data["source_dataset"].tolist())),
                "subject_held_out": {key: value for key, value in subject.items()
                                     if key not in {"truths", "predictions"}},
                "dataset_held_out": {key: value for key, value in dataset.items()
                                     if key not in {"truths", "predictions"}},
                "figures": figures,
                "figure_protocol": figure_protocol,
            })
        except Exception as exc:  # noqa: BLE001 - preserve other categories
            category_report.update({
                "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        report["categories"][category] = category_report

    source_rows = _source_rows()
    csv_path = report_root / "dataset_source_literature_comparison.csv"
    _write_dataset_csv(csv_path, source_rows)
    inventory_rows = _archive_inventory_rows(report)
    inventory_csv_path = report_root / "dataset_archive_inventory.csv"
    _write_dataset_csv(inventory_csv_path, inventory_rows)
    report["dataset_archive_inventory"] = inventory_rows
    json_path = report_root / "evidence_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    successful = [row for row in report["categories"].values()
                  if row.get("status") == "success"]
    lines = [
        "# Public motor-imagery evidence report",
        "",
        f"Generated: `{report['created']}`  ",
        f"Git commit: `{report['git_commit'] or 'unavailable'}`  ",
        f"Processed root: `{report['processed_root']}`  ",
        f"Model root: `{report['model_root']}`",
        "",
        ("> **Metric boundary.** Project scores are balanced accuracy from grouped "
        "complete-subject or complete-source folds. Published accuracy and Cohen's "
        "kappa below retain their original names. They are not converted or treated "
        "as directly comparable."),
        "",
        "## Dataset sources and participant cohorts",
        "",
        ("Sex/gender labels below preserve each source's terminology and include "
        "an explicit reported-n when available. Missing demographics stay missing; "
        "they are never inferred from names, signals, or subject IDs."),
        "",
        _markdown_table(source_rows, [
            "Dataset", "Authoritative source", "Participants", "Population",
            "Sex/gender as reported", "Age", "Handedness",
            "Source trial design", "Cohort completeness", "Cohort references"]),
        "",
        "## Published protocols, results, and historical project run",
        "",
        _markdown_table(source_rows, [
            "Dataset", "Source protocol", "Published result", "Published metric",
            "Published validation", "References", "Historical project result"]),
        "",
        ("Historical project numbers above are an explicit audit snapshot from "
        "`IMPLEMENTATION_AND_VALIDATION_AUDIT.md`. They are not silently treated "
        "as recomputed values when the corresponding archives/model bundles are absent."),
        "",
        "### Historical category summary",
        "",
        f"- Hand overall: {HISTORICAL_PROJECT_RESULTS['hand_overall']}",
        "",
        "## Standardized archive inventory",
        "",
        ("`Trials` counts imported epochs after task selection. `Samples per trial` "
        "is the temporal length of one channel; `Total channel samples` is "
        "trials × channels × temporal samples. These archive-backed counts are "
        "kept separate from source-design counts."),
        "",
        _markdown_table(inventory_rows, [
            "Dataset", "Category", "Status", "Source participants",
            "Imported participants", "Sessions", "Trials", "Channels",
            "Samples per trial", "Total channel samples", "FS Hz",
            "Nominal epoch seconds"]),
        "",
        "## Recomputed archive-backed evidence",
        "",
        ("Evaluation holds out whole subjects; no random epoch split is used. "
        "Dataset-held-out results are emitted only for class-complete sources."),
        "",
    ]
    if not successful:
        lines.extend([
            ("**No archive-backed category was evaluated in this workspace.** "
            "No CSP/tangent/confusion/covariance plot has been fabricated. Restore "
            "`dataset/public/processed/<dataset>/<category>/epochs.npz` and rerun "
            "the command recorded in `evidence_report.json`."),
            "",
        ])
    for category, row in report["categories"].items():
        lines.extend([f"### {category}", ""])
        if row.get("status") != "success":
            lines.extend([
                (f"Status: **{row.get('status')}** — "
                f"{row.get('reason') or row.get('error') or 'not evaluated'}"), ""])
            continue
        subject = row["subject_held_out"]
        dataset = row["dataset_held_out"]
        model_info = row.get("model_info", {})
        if model_info.get("available"):
            metadata = model_info["metadata"]
            selected_subject = metadata.get("cv_balanced_accuracy")
            selected_dataset = metadata.get("dataset_cv_balanced_accuracy")
            lines.extend([
                f"- Saved selected model: `{metadata.get('pipeline', 'unknown')}`",
                (f"- Selected-model subject-held-out BA: "
                 f"**{float(selected_subject):.3%} ± {float(metadata.get('cv_std', 0)):.3%}**"
                 if selected_subject is not None else
                 "- Selected-model subject-held-out BA: unavailable"),
                (f"- Selected-model dataset-held-out BA: "
                 f"**{float(selected_dataset):.3%} ± "
                 f"{float(metadata.get('dataset_cv_std', 0)):.3%}**"
                 if selected_dataset is not None else
                 "- Selected-model dataset-held-out BA: unavailable"),
                f"- Selected-model trials / subjects: {metadata.get('n_trials', 'n/a')} / {metadata.get('n_subjects', 'n/a')}",
            ])
            candidates = metadata.get("selection", {}).get("candidates", [])
            if candidates:
                candidate_rows = []
                for candidate in candidates:
                    candidate_rows.append({
                        "Pipeline": candidate.get("pipeline"),
                        "Status": candidate.get("status"),
                        "Subject-held-out BA": (
                            f"{float(candidate['accuracy']):.3%} ± "
                            f"{float(candidate.get('std', 0)):.3%}"
                            if candidate.get("accuracy") is not None else
                            candidate.get("error", "unavailable")),
                        "Dataset-held-out BA": (
                            f"{float(candidate['dataset_accuracy']):.3%} ± "
                            f"{float(candidate.get('dataset_std', 0)):.3%}"
                            if candidate.get("dataset_accuracy") is not None else
                            "unavailable"),
                    })
                lines.extend([
                    "", "Candidate model comparison:", "",
                    _markdown_table(candidate_rows, [
                        "Pipeline", "Status", "Subject-held-out BA",
                        "Dataset-held-out BA"]), "",
                ])
        lines.extend([
            f"- Evidence recomputation pipeline: `{row['pipeline']}`",
            f"- Trials / subjects / datasets: {row['n_trials']} / {row['n_subjects']} / {row['n_datasets']}",
            f"- Classes: {', '.join(row['labels'])}",
            f"- Chance: {row['chance_balanced_accuracy']:.3%}",
            f"- Subject-held-out BA: **{subject['mean_balanced_accuracy']:.3%} ± {subject['std_balanced_accuracy']:.3%}** across {len(subject['scores'])} folds",
        ])
        if dataset.get("available"):
            lines.append(
                f"- Dataset-held-out BA: **{dataset['mean_balanced_accuracy']:.3%} ± {dataset['std_balanced_accuracy']:.3%}** across {len(dataset['scores'])} folds")
        else:
            lines.append(f"- Dataset-held-out BA: unavailable ({dataset.get('reason')})")
        lines.extend(["", "Figures:", ""])
        captions = {
            "csp_class_separation": "CSP/FBCSP held-out class-colored relationship plot",
            "tangent_pca": "36-feature tangent-space held-out PCA",
            "class_covariance_summary": "Held-out class covariance summaries",
            "subject_oof_confusion": "Grouped out-of-fold confusion matrix",
            "subject_fold_scores": "Complete-subject fold balanced accuracy",
        }
        for figure in row["figures"]:
            relative = _relpath(figure["path"], report_root)
            lines.extend([f"**{captions[figure['kind']]}**", "",
                          f"![{captions[figure['kind']]}]({relative})", ""])
        lines.extend([
            ("The colored CSP and tangent points are held-out subjects from one "
            "declared representative fold. Feature extraction, scaling and PCA "
            "were fitted only on that fold's training subjects. The plot is "
            "descriptive; the reported BA comes from all grouped out-of-fold predictions."),
            "",
        ])

    lines.extend([
        "## Reproducibility and limitations",
        "",
        "- `evidence_report.json` records archive paths, shape, provenance, preprocessing, file size/mtime, evaluation policy, model metadata availability, and figure fit boundary.",
        "- Pass `--hash-inputs` to add full SHA-256 hashes. Hashing is optional because complete processed archives may be large.",
        "- Public cross-subject scores are not personal Cyton performance and do not validate spontaneous/asynchronous control.",
        "- Default 1–40 Hz archives preserve broader EEG than the feature filters but still cannot reconstruct Ofner's 0.3–1 Hz MRCP content.",
        "- Report-only complete-test-subject RA remains transductive and is not promoted here as an inductive production result.",
        "- All figures are generated from actual standardized arrays. If arrays are missing, the report says so and emits no synthetic figure.",
        "",
        f"Machine-readable dataset comparison: `{csv_path.name}`  ",
        f"Machine-readable archive inventory: `{inventory_csv_path.name}`  ",
        f"Machine-readable full report: `{json_path.name}`",
    ])
    markdown_path = report_root / "EVIDENCE_REPORT.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, markdown_path, json_path, csv_path
