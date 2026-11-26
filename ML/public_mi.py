"""Licensed public motor-imagery dataset registry and standardized archives.

Edited by Anson
Date: 10/11/2026
Purpose: keep source task semantics, licensing, provenance, and leakage-safe
         subject/session metadata attached to every imported public EEG epoch.

Public epochs are benchmark/pretraining data. They are not personal Cyton
calibration and are never silently concatenated with personal recordings.
"""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from ML.motor_intent import (
    CATEGORY_FEET,
    CATEGORY_FINGER,
    CATEGORY_HAND,
    CATEGORY_UPPER_LIMB,
    IDLE,
    MI_CATEGORIES,
    intent_category,
    normalize_category,
)

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ROOT = ROOT / "dataset" / "public"
PUBLIC_RAW_ROOT = PUBLIC_ROOT / "raw"
PUBLIC_PROCESSED_ROOT = PUBLIC_ROOT / "processed"
PUBLIC_MODEL_ROOT = ROOT / "models" / "public"
ARCHIVE_SCHEMA_VERSION = 1
CYTON_MOTOR_CHANNELS = ("F3", "C3", "CZ", "P3", "PZ", "F4", "C4", "P4")

# Verified against the published release sidecars, NEMAR sourcedata manifests,
# and MOABB 1.7.2 dataset loaders. These contracts make provider changes fail
# before an archive is silently written with a different schema.
PUBLIC_DATA_FORMATS = {
    "openneuro_ds008446": {
        "source_format": "BIDS EDF + TSV v1.0.1",
        "native_sfreq": 512.0,
        # Direct read_raw_edf sees 62 scalp + Ref1/Ref2 + Marker as EEG;
        # _channels.tsv identifies Marker as TRIG, and it is never picked.
        "eeg_channel_count": 65,
        "source_events": tuple(str(value) for value in range(1, 8)),
        "native_target_channels": CYTON_MOTOR_CHANNELS,
        "interpolated_target_channels": (),
    },
    "physionet_eegmmidb": {
        "source_format": "EDF+ annotations",
        "native_sfreq": 160.0,
        # The released imagery/execution EDFs for these three subjects are
        # natively 128 Hz. MOABB independently documents the same exception.
        "native_sfreq_exceptions": {
            "S088": 128.0, "S092": 128.0, "S100": 128.0},
        "eeg_channel_count": 64,
        "source_events": ("T0", "T1", "T2"),
        "native_target_channels": CYTON_MOTOR_CHANNELS,
        "interpolated_target_channels": (),
    },
    "bnci2014_001": {
        "source_format": "MAT via MOABB/NEMAR",
        "native_sfreq": 250.0,
        "eeg_channel_count": 22,
        "source_events": ("left_hand", "right_hand", "feet", "tongue"),
        "native_target_channels": ("C3", "CZ", "PZ", "C4"),
        "interpolated_target_channels": ("F3", "P3", "F4", "P4"),
    },
    "ofner2017": {
        "source_format": "GDF via MOABB/NEMAR",
        "native_sfreq": 512.0,
        "eeg_channel_count": 61,
        "source_events": (
            "right_elbow_flexion", "right_elbow_extension",
            "right_supination", "right_pronation", "right_hand_close",
            "right_hand_open", "rest",
        ),
        "native_target_channels": CYTON_MOTOR_CHANNELS,
        "interpolated_target_channels": (),
    },
    "cho2017": {
        "source_format": "MAT via MOABB/NEMAR",
        "native_sfreq": 512.0,
        "eeg_channel_count": 64,
        "source_events": ("left_hand", "right_hand"),
        "native_target_channels": CYTON_MOTOR_CHANNELS,
        "interpolated_target_channels": (),
    },
    "lee2019_mi": {
        "source_format": "MAT via MOABB/NEMAR",
        "native_sfreq": 1000.0,
        "eeg_channel_count": 62,
        "source_events": ("left_hand", "right_hand"),
        "native_target_channels": CYTON_MOTOR_CHANNELS,
        "interpolated_target_channels": (),
    },
}

# ds008446 publishes finger identity but does not publish hand laterality in its
# task labels. Explicit "unsided" labels prevent downstream code from inventing
# left/right finger classes.
UNSIDED_FINGER_LABELS = (
    "finger_thumb_unsided", "finger_index_unsided",
    "finger_middle_unsided", "finger_ring_unsided",
    "finger_pinky_unsided",
)

PUBLIC_MI_DATASETS = {
    "openneuro_ds008446": {
        "name": "Random and Sequential Order Finger Motor Imagery",
        "categories": (CATEGORY_FINGER,),
        "paradigm": "imagery",
        "modality": "EEG",
        "subjects": 20,
        "channels": 62,
        "sample_rate": 512,
        "format": "BIDS/EDF",
        "license": "CC0-1.0",
        "redistribute_derived": True,
        "access": "public",
        "download": "https://openneuro.org/datasets/ds008446/versions/1.0.1/download",
        "citation": "Damm, Jiang & Demosthenous (2026), doi:10.18112/openneuro.ds008446.v1.0.1",
        "cohort": {
            "population": "healthy volunteers",
            "participants": 20,
            "sex_or_gender_as_reported": {
                "female": 13, "male": 4, "other": 3, "reported_n": 20},
            "age": {"mean_years": 28.3, "range_years": [19, 57],
                    "reported_n": 20},
            "handedness": {"right": 18, "left": 2, "reported_n": 20},
            "source_trial_design": (
                "80 EEG recordings across random and sequential cue-order "
                "conditions; actual imported epoch counts are reported from archives"),
            "completeness_note": (
                "Aggregate demographics are from release participant metadata as "
                "summarized by OpenNeuro/EEGDash; retain the non-binary 'other' "
                "category rather than forcing a male/female split."),
            "references": (
                "https://openneuro.org/datasets/ds008446/versions/1.0.1",
                "https://eegdash.org/api/dataset/eegdash.dataset.DS008446.html",
            ),
        },
        "importer": "bids_finger",
        "source_label_map": {
            "3": "finger_thumb_unsided", "4": "finger_index_unsided",
            "5": "finger_middle_unsided", "6": "finger_ring_unsided",
            "7": "finger_pinky_unsided",
        },
        "notes": "Finger cues have no hand side in the released task labels; never map them to left_* or right_*.",
    },
    "physionet_eegmmidb": {
        "name": "EEG Motor Movement/Imagery Dataset",
        "categories": (CATEGORY_HAND, CATEGORY_FEET),
        "paradigm": "imagery and execution; importer accepts imagery runs only",
        "modality": "EEG",
        "subjects": 109,
        "channels": 64,
        "sample_rate": 160,
        "format": "EDF+",
        "license": "ODC-By-1.0",
        "redistribute_derived": True,
        "access": "public",
        "download": "https://physionet.org/content/eegmmidb/1.0.0/",
        "citation": "Schalk (2009), EEG Motor Movement/Imagery Dataset v1.0.0, doi:10.13026/C28G6P",
        "cohort": {
            "population": "volunteers; clinical status not specified in the release",
            "participants": 109,
            "sex_or_gender_as_reported": None,
            "age": None,
            "handedness": None,
            "source_trial_design": (
                "14 runs per participant; this workflow downloads only imagery "
                "runs 4, 6, 8, 10, 12 and 14 and counts retained annotated epochs"),
            "completeness_note": (
                "The authoritative PhysioNet release states 109 volunteers but "
                "does not publish an age, sex/gender or handedness breakdown. "
                "The report leaves these fields missing rather than inferring them."),
            "references": (
                "https://physionet.org/content/eegmmidb/1.0.0/",),
        },
        "importer": "physionet",
        "source_label_map": {
            "R04:T0": IDLE, "R04:T1": "left_hand", "R04:T2": "right_hand",
            "R08:T0": IDLE, "R08:T1": "left_hand", "R08:T2": "right_hand",
            "R12:T0": IDLE, "R12:T1": "left_hand", "R12:T2": "right_hand",
            "R06:T0": IDLE, "R06:T1": "both_hands", "R06:T2": "both_feet",
            "R10:T0": IDLE, "R10:T1": "both_hands", "R10:T2": "both_feet",
            "R14:T0": IDLE, "R14:T1": "both_hands", "R14:T2": "both_feet",
        },
        "notes": (
            "T1/T2 meanings depend on run number. Execution runs "
            "3/5/7/9/11/13 must not be imported as imagery. Released files "
            "for S088, S092 and S100 are 128 Hz rather than the usual 160 Hz; "
            "the importer validates those subject-specific exceptions and "
            "resamples every retained epoch to 250 Hz."),
    },
    "bnci2014_001": {
        "name": "BCI Competition IV 2a",
        "categories": (CATEGORY_HAND, CATEGORY_FEET),
        "paradigm": "imagery",
        "modality": "EEG",
        "subjects": 9,
        "channels": 22,
        "sample_rate": 250,
        "format": "MAT via MOABB/NEMAR",
        "license": "CC-BY-ND-4.0",
        "redistribute_derived": False,
        "access": "public",
        "download": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html",
        "citation": "Tangermann et al. (2012), doi:10.3389/fnins.2012.00055",
        "cohort": {
            "population": "9 participants; health/demographic breakdown not stated",
            "participants": 9,
            "sex_or_gender_as_reported": None,
            "age": None,
            "handedness": None,
            "source_trial_design": (
                "2 sessions × 288 trials per participant; 72 trials per class per "
                "session; 5,184 four-class source trials"),
            "completeness_note": (
                "The competition description gives the participant count and "
                "protocol but no reliable age or sex/gender table; fields remain missing."),
            "references": (
                "https://www.bbci.de/competition/iv/desc_2a.pdf",
                "https://doi.org/10.3389/fnins.2012.00055",
            ),
        },
        "importer": "moabb",
        "moabb_class": "BNCI2014_001",
        "source_label_map": {"left_hand": "left_hand", "right_hand": "right_hand", "feet": "both_feet"},
        "ignored_source_labels": ("tongue",),
        "notes": "The tongue class is intentionally excluded; CC-BY-ND restricts redistribution of derived archives.",
    },
    "ofner2017": {
        "name": "Upper-limb Motor Imagery Dataset from Ofner et al.",
        "categories": (CATEGORY_UPPER_LIMB,),
        "paradigm": "imagery and execution; importer selects imagery session only",
        "modality": "EEG",
        "subjects": 15,
        "channels": 61,
        "sample_rate": 512,
        "format": "GDF via MOABB",
        "license": "CC-BY-4.0",
        "redistribute_derived": True,
        "access": "public",
        "download": "https://doi.org/10.5281/zenodo.834976",
        "citation": "Ofner et al. (2017), doi:10.1371/journal.pone.0182578",
        "cohort": {
            "population": "healthy volunteers",
            "participants": 15,
            "sex_or_gender_as_reported": {
                "female": 9, "male": 6, "reported_n": 15},
            "age": {"mean_years": 27.0, "std_years": 5.0,
                    "range_years": [22, 40], "reported_n": 15},
            "handedness": {"right": 14, "left": 1, "reported_n": 15},
            "source_trial_design": (
                "MI session: 10 runs × 42 trials per participant; 60 trials for "
                "each of six movements and rest; 6,300 MI trials total"),
            "completeness_note": "Complete aggregate cohort description in the paper.",
            "references": (
                "https://doi.org/10.1371/journal.pone.0182578",),
        },
        "importer": "moabb",
        "moabb_class": "Ofner2017",
        "source_label_map": {
            "right_elbow_flexion": "right_elbow_flexion",
            "right_elbow_extension": "right_elbow_extension",
            "right_supination": "right_forearm_supination",
            "right_pronation": "right_forearm_pronation",
            "right_hand_close": "right_hand_close",
            "right_hand_open": "right_hand_open",
            "rest": IDLE,
        },
        "notes": "All movement classes are right-sided. Movement-execution data is not imported by default.",
    },
    "cho2017": {
        "name": "Cho et al. 2017 Motor Imagery",
        "categories": (CATEGORY_HAND,),
        "paradigm": "imagery",
        "modality": "EEG with EMG recorded for movement checking",
        "subjects": 52,
        "channels": 64,
        "sample_rate": 512,
        "format": "MAT via MOABB",
        "license": "CC-BY-4.0",
        "redistribute_derived": True,
        "access": "public",
        "download": "https://doi.org/10.5524/100295",
        "citation": "Cho et al. (2017), doi:10.1093/gigascience/gix034",
        "cohort": {
            "population": "healthy volunteers",
            "participants": 52,
            "sex_or_gender_as_reported": {
                "female": 19, "male": 33, "reported_n": 52},
            "age": {"mean_years": 24.8, "std_years": 3.86,
                    "reported_n": 52},
            "handedness": {"right": 50, "both": 2, "reported_n": 52},
            "source_trial_design": (
                "100 or 120 trials per class (200–240 per participant); 10,520 "
                "left/right imagery trials in the source collection"),
            "completeness_note": (
                "The paper reports all 52 participants; its published 67.46% result "
                "uses 50 usable subjects."),
            "references": (
                "https://doi.org/10.1093/gigascience/gix034",
                "https://doi.org/10.5524/100295",
            ),
        },
        "importer": "moabb",
        "moabb_class": "Cho2017",
        "source_label_map": {"left_hand": "left_hand", "right_hand": "right_hand"},
        "notes": "Labels are left/right hand kinesthetic finger-movement imagery, not individual-finger classes.",
    },
    "lee2019_mi": {
        "name": "OpenBMI / Lee et al. 2019 Motor Imagery",
        "categories": (CATEGORY_HAND,),
        "paradigm": "imagery",
        "modality": "EEG with EMG",
        "subjects": 54,
        "channels": 62,
        "sample_rate": 1000,
        "format": "MAT via MOABB/NEMAR",
        "license": "CC-BY-4.0",
        "redistribute_derived": True,
        "access": "public",
        "download": "https://doi.org/10.5524/100542",
        "citation": "Lee et al. (2019), doi:10.1093/gigascience/giz002",
        "cohort": {
            "population": "healthy volunteers",
            "participants": 54,
            "sex_or_gender_as_reported": {
                "female": 25, "male": 29, "reported_n": 54},
            "age": {"mean_years": 24.8, "std_years": 3.8,
                    "range_years": [24, 35], "reported_n": 54},
            "handedness": {
                "right": 50, "left": 2, "ambidextrous": 2, "reported_n": 54},
            "source_trial_design": (
                "2 sessions on different days with offline training and online test "
                "phases; 100 trials per phase per session; 400 MI trials per "
                "participant in the full source design"),
            "completeness_note": (
                "The workflow importer uses the labelled MI data exposed by MOABB; "
                "the archive inventory reports the exact retained trial count."),
            "references": (
                "https://doi.org/10.1093/gigascience/giz002",
                "https://doi.org/10.5524/100542",
            ),
        },
        "importer": "moabb",
        "moabb_class": "Lee2019_MI",
        "source_label_map": {"left_hand": "left_hand", "right_hand": "right_hand"},
        "notes": "Importer uses labelled training runs only. The GigaDB EEG is CC-BY-4.0; GPL-3.0 applies to the OpenBMI software and is inherited by NEMAR derivative metadata.",
    },
    "zenodo_18470278": {
        "name": "Motor imagery with an extra virtual arm",
        "categories": (CATEGORY_UPPER_LIMB,),
        "paradigm": "imagery",
        "modality": "EEG",
        "subjects": 28,
        "channels": 54,
        "sample_rate": None,
        "format": "MAT",
        "license": "not stated on record",
        "redistribute_derived": False,
        "access": "restricted",
        "download": "https://zenodo.org/records/18470278",
        "citation": "Leal Pinheiro et al. (2026), related publication doi:10.1109/TMRB.2025.3625073",
        "importer": None,
        "source_label_map": {"left_arm": "left_arm", "right_arm": "right_arm", "extra_virtual_arm": "extra_virtual_arm"},
        "notes": "Files are restricted and license is unstated. No automated importer is enabled until access and reuse terms are verified.",
    },
}


def get_public_dataset(dataset_id):
    """Return a copy of one registry entry."""
    key = str(dataset_id).strip().lower()
    if key not in PUBLIC_MI_DATASETS:
        raise KeyError(f"unknown public MI dataset {dataset_id!r}; known: {sorted(PUBLIC_MI_DATASETS)}")
    row = dict(PUBLIC_MI_DATASETS[key])
    row["id"] = key
    return row


def public_archive_path(dataset_id, category, root=None):
    """Canonical processed path: dataset/public/processed/id/category/epochs.npz."""
    get_public_dataset(dataset_id)
    category = normalize_category(category)
    base = Path(root) if root is not None else PUBLIC_PROCESSED_ROOT
    return base / str(dataset_id).lower() / category / "epochs.npz"


def public_model_dir(category, root=None):
    base = Path(root) if root is not None else PUBLIC_MODEL_ROOT
    return base / normalize_category(category)


def public_label_category(label):
    label = str(label)
    if label == IDLE:
        return None
    if label in UNSIDED_FINGER_LABELS:
        return CATEGORY_FINGER
    if label == "extra_virtual_arm":
        return CATEGORY_UPPER_LIMB
    return intent_category(label)


def _as_trial_vector(name, values, n_trials):
    arr = np.asarray(values).astype(str)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), n_trials)
    if arr.ndim != 1 or len(arr) != n_trials:
        raise ValueError(f"{name} must have one value per trial ({n_trials}), got {arr.shape}")
    return arr


def write_standardized_archive(path, X, labels, subjects, sessions, *,
                               source_dataset, source_labels, channel_labels,
                               fs, units, imagery_or_execution="imagery",
                               preprocessing=None, category=None):
    """Validate and write a provenance-rich public EEG epoch archive.

    The archive intentionally stores source labels alongside canonical labels.
    One archive has one category, montage/order, sample rate, unit, and paradigm.
    """
    entry = get_public_dataset(source_dataset)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or min(X.shape) == 0:
        raise ValueError(f"X must be non-empty (trials, channels, samples), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    n_trials = len(X)
    labels = _as_trial_vector("labels", labels, n_trials)
    source_labels = _as_trial_vector("source_labels", source_labels, n_trials)
    subjects = _as_trial_vector("subjects", subjects, n_trials)
    sessions = _as_trial_vector("sessions", sessions, n_trials)
    channels = np.asarray(channel_labels).astype(str)
    if channels.ndim != 1 or len(channels) != X.shape[1] or len(set(channels)) != len(channels):
        raise ValueError("channel_labels must be unique and match X.shape[1]")
    if float(fs) <= 0:
        raise ValueError("fs must be positive")
    if not str(units).strip():
        raise ValueError("units must be explicit (normally 'volts' for MNE imports)")
    paradigm = str(imagery_or_execution).strip().lower()
    if paradigm not in {"imagery", "execution"}:
        raise ValueError("imagery_or_execution must be 'imagery' or 'execution'")

    non_idle_categories = {public_label_category(v) for v in labels if v != IDLE}
    if None in non_idle_categories:
        raise ValueError("archive contains an unknown canonical task label")
    if category is None:
        if len(non_idle_categories) != 1:
            raise ValueError(f"archive must contain exactly one non-idle category, got {sorted(non_idle_categories)}")
        category = next(iter(non_idle_categories))
    category = normalize_category(category)
    if non_idle_categories and non_idle_categories != {category}:
        raise ValueError(f"labels belong to {sorted(non_idle_categories)}, not category {category!r}")
    if category not in entry["categories"]:
        raise ValueError(f"registry says {source_dataset!r} does not provide category {category!r}")

    declared_map = entry["source_label_map"]
    allowed = set(declared_map.values())
    unknown_labels = set(labels) - allowed
    if unknown_labels:
        raise ValueError(f"labels are not declared in registry mapping: {sorted(unknown_labels)}")
    mismatches = [(source, label) for source, label in zip(source_labels, labels)
                  if declared_map.get(source) != label]
    if mismatches:
        raise ValueError(
            "source/canonical label pairs do not match the licensed dataset "
            f"registry mapping; first mismatch={mismatches[0]!r}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing = dict(preprocessing or {})
    metadata = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "source_dataset": source_dataset,
        "category": category,
        "license": entry["license"],
        "citation": entry["citation"],
        "download": entry["download"],
        "imagery_or_execution": paradigm,
        "units": str(units),
        "fs": float(fs),
        "channel_labels": channels.tolist(),
        "preprocessing": preprocessing,
        "redistribute_derived": bool(entry["redistribute_derived"]),
    }
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary, X=X, labels=labels, subjects=subjects, sessions=sessions,
        source_dataset=np.asarray(source_dataset), source_labels=source_labels,
        channel_labels=channels, fs=np.asarray(float(fs)),
        units=np.asarray(str(units)),
        imagery_or_execution=np.asarray(paradigm),
        license=np.asarray(entry["license"]), citation=np.asarray(entry["citation"]),
        category=np.asarray(category),
        preprocessing_json=np.asarray(json.dumps(preprocessing, sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        schema_version=np.asarray(ARCHIVE_SCHEMA_VERSION, dtype=np.int64),
    )
    temporary.replace(path)
    return path


def load_standardized_archive(path, *, imagery_only=True):
    """Load and revalidate a standardized archive without pickle."""
    path = Path(path)
    required = {
        "X", "labels", "subjects", "sessions", "source_dataset",
        "source_labels", "channel_labels", "fs", "units",
        "imagery_or_execution", "license", "citation", "category",
        "preprocessing_json", "schema_version",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing standardized fields: {sorted(missing)}")
        out = {key: np.array(data[key]) for key in required}
    if int(out["schema_version"]) != ARCHIVE_SCHEMA_VERSION:
        raise ValueError(f"unsupported public archive schema {int(out['schema_version'])}")
    out["X"] = out["X"].astype(np.float64)
    for key in ("labels", "subjects", "sessions", "source_labels", "channel_labels"):
        out[key] = out[key].astype(str)
    for key in ("source_dataset", "units", "imagery_or_execution", "license", "citation", "category"):
        out[key] = str(out[key])
    out["fs"] = float(out["fs"])
    out["preprocessing"] = json.loads(str(out.pop("preprocessing_json")))
    if imagery_only and out["imagery_or_execution"] != "imagery":
        raise ValueError(f"{path} contains {out['imagery_or_execution']}, not motor imagery")
    if len(out["X"]) != len(out["labels"]):
        raise ValueError(f"{path} has inconsistent trial metadata lengths")
    if len(out["channel_labels"]) != out["X"].shape[1]:
        raise ValueError(f"{path} channel metadata does not match X")
    if {public_label_category(v) for v in out["labels"] if v != IDLE} != {out["category"]}:
        raise ValueError(f"{path} mixes or mislabels motor-intent categories")
    return out


def combine_public_archives(paths, *, category=None):
    """Combine standardized public archives only after strict modality checks."""
    rows = [load_standardized_archive(path) for path in paths]
    if not rows:
        raise ValueError("at least one standardized archive is required")
    wanted = normalize_category(category or rows[0]["category"])
    reference = rows[0]
    checks = ("category", "fs", "units", "imagery_or_execution")
    for row in rows:
        if row["category"] != wanted:
            raise ValueError(f"cannot mix {row['category']!r} into {wanted!r} category training")
        for key in checks:
            if row[key] != reference[key]:
                raise ValueError(f"public archives use different {key}: {reference[key]!r} vs {row[key]!r}")
        if row["X"].shape[1:] != reference["X"].shape[1:]:
            raise ValueError("public archives need identical channel/sample shape; explicitly pick/resample first")
        if row["channel_labels"].tolist() != reference["channel_labels"].tolist():
            raise ValueError("public archives use different channel labels/order")
    return {
        "X": np.concatenate([r["X"] for r in rows]),
        "labels": np.concatenate([r["labels"] for r in rows]),
        "subjects": np.concatenate([
            np.asarray([f"{r['source_dataset']}:{v}" for v in r["subjects"]]) for r in rows]),
        "sessions": np.concatenate([
            np.asarray([f"{r['source_dataset']}:{s}:{v}" for s, v in zip(r["subjects"], r["sessions"])])
            for r in rows]),
        "source_labels": np.concatenate([r["source_labels"] for r in rows]),
        "source_dataset": np.concatenate([
            np.repeat(r["source_dataset"], len(r["X"])) for r in rows]),
        "category": wanted, "fs": reference["fs"],
        "units": reference["units"],
        "channel_labels": reference["channel_labels"],
        "imagery_or_execution": "imagery",
    }


def grouped_folds(groups, labels, n_splits=5):
    """Yield class-complete folds with no subject/session in both sides."""
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels).astype(str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return
    splitter = GroupKFold(n_splits=min(int(n_splits), len(unique_groups)))
    classes = set(labels)
    for train, test in splitter.split(np.zeros(len(labels)), labels, groups):
        if set(labels[train]) == classes and set(labels[test]) == classes:
            yield train, test


def subject_folds(subjects, labels, n_splits=5):
    """Public-data evaluation split: entire subjects are held out."""
    yield from grouped_folds(subjects, labels, n_splits=n_splits) or []


def session_folds(sessions, labels, n_splits=5):
    """Personal/session-domain evaluation split: entire sessions are held out."""
    yield from grouped_folds(sessions, labels, n_splits=n_splits) or []


def _require_mne():
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("public EDF/BIDS import needs MNE: python -m pip install '.[public-mi]'") from exc
    mne.set_log_level("ERROR")
    return mne


def _subject_from_path(path):
    match = re.search(r"(?:^|[/\\])(sub-[A-Za-z0-9]+|S\d{3})(?:[/\\]|$)", str(path))
    if match:
        return match.group(1)
    stem = Path(path).stem
    physionet = re.fullmatch(r"(S\d{3})R\d{2}", stem, flags=re.IGNORECASE)
    return physionet.group(1).upper() if physionet else stem


def _session_from_name(path):
    name = Path(path).stem
    task = re.search(r"task-([A-Za-z0-9]+)", name)
    run = re.search(r"run-([A-Za-z0-9]+)", name)
    if task or run:
        return "_".join(v.group(1) for v in (task, run) if v)
    physio_run = re.search(r"R(\d{2})", name, re.IGNORECASE)
    return f"run-{physio_run.group(1)}" if physio_run else name


def _rename_upper(raw):
    raw.rename_channels(lambda value: value.strip().strip(".").upper())


def _resample_with_events(raw, events, fs_out):
    if abs(float(raw.info["sfreq"]) - float(fs_out)) < 1e-9:
        return raw, events
    result = raw.resample(float(fs_out), events=events, verbose=False)
    return result if isinstance(result, tuple) else (raw, events)


def public_channel_plan(dataset_id, target_channels=CYTON_MOTOR_CHANNELS):
    """Return the audited exact/interpolated target-channel plan."""
    try:
        contract = PUBLIC_DATA_FORMATS[dataset_id]
    except KeyError as exc:
        raise KeyError(f"no verified data-format contract for {dataset_id!r}") from exc
    targets = tuple(str(value).upper() for value in target_channels)
    if len(set(targets)) != len(targets):
        raise ValueError(f"target channels must be unique, got {targets}")
    allowed = set(contract["interpolated_target_channels"])
    return {
        "target_channels": targets,
        "native_channels": tuple(value for value in targets if value not in allowed),
        "interpolated_channels": tuple(value for value in targets if value in allowed),
    }


def public_native_sfreqs(dataset_id):
    """Return every verified native sampling rate for one source."""
    try:
        contract = PUBLIC_DATA_FORMATS[dataset_id]
    except KeyError as exc:
        raise KeyError(f"no verified data-format contract for {dataset_id!r}") from exc
    values = {float(contract["native_sfreq"])}
    values.update(float(value) for value in
                  contract.get("native_sfreq_exceptions", {}).values())
    return tuple(sorted(values))


def _epochs_data(epochs):
    """Return a writable copy across supported MNE get_data signatures."""
    try:
        return epochs.get_data(copy=True)
    except TypeError:  # MNE 1.5 has no copy keyword
        return np.array(epochs.get_data(), copy=True)


def _validate_raw_format(dataset_id, raw, path, target_channels):
    contract = PUBLIC_DATA_FORMATS[dataset_id]
    sfreq = float(raw.info["sfreq"])
    subject = _subject_from_path(path)
    expected_sfreq = float(contract.get(
        "native_sfreq_exceptions", {}).get(subject, contract["native_sfreq"]))
    if not np.isclose(sfreq, expected_sfreq, rtol=0, atol=1e-6):
        raise ValueError(
            f"{path} violates the verified {dataset_id} sampling contract: "
            f"expected {expected_sfreq} Hz for {subject}, got {sfreq} Hz")
    channel_types = raw.get_channel_types()
    eeg_count = sum(value == "eeg" for value in channel_types)
    if eeg_count != int(contract["eeg_channel_count"]):
        type_counts = {
            value: channel_types.count(value) for value in sorted(set(channel_types))}
        raise ValueError(
            f"{path} violates the verified {dataset_id} channel contract: "
            f"expected {contract['eeg_channel_count']} EEG channels, got "
            f"{eeg_count}; types={type_counts}")
    plan = public_channel_plan(dataset_id, target_channels)
    available = {str(value).upper() for value in raw.ch_names}
    missing = [value for value in plan["native_channels"] if value not in available]
    if missing:
        raise ValueError(
            f"{path} violates the verified {dataset_id} target-channel "
            f"contract; missing native channels {missing}")
    return plan


def _validate_moabb_dataset(dataset_id, dataset, entry, target_channels):
    contract = PUBLIC_DATA_FORMATS[dataset_id]
    selected_events = tuple(entry["source_label_map"])
    undeclared = set(selected_events) - set(contract["source_events"])
    if undeclared:
        raise ValueError(
            f"{dataset_id} registry events are outside its verified format: "
            f"{sorted(undeclared)}")
    event_id = getattr(dataset, "event_id", {})
    missing_events = [value for value in selected_events if value not in event_id]
    if missing_events:
        raise ValueError(
            f"MOABB {entry['moabb_class']} changed its event vocabulary; "
            f"missing {missing_events}, available={sorted(event_id)}")
    interval = tuple(getattr(dataset, "interval", ()))
    if len(interval) != 2 or float(interval[1]) <= float(interval[0]):
        raise ValueError(
            f"MOABB {entry['moabb_class']} has an invalid trial interval "
            f"{interval}")
    plan = public_channel_plan(dataset_id, target_channels)
    # Current MOABB exposes this catalog metadata without downloading data.
    # Older supported releases may not, so post-load validation remains
    # mandatory even when this early check cannot run.
    metadata = getattr(dataset, "metadata", None)
    acquisition = getattr(metadata, "acquisition", None)
    if acquisition is not None:
        native_fs = float(acquisition.sampling_rate)
        if not np.isclose(
                native_fs, contract["native_sfreq"], rtol=0, atol=1e-6):
            raise ValueError(
                f"MOABB {entry['moabb_class']} metadata changed sampling rate: "
                f"expected {contract['native_sfreq']}, got {native_fs}")
        channel_types = dict(acquisition.channel_types)
        if int(channel_types.get("eeg", -1)) != int(
                contract["eeg_channel_count"]):
            raise ValueError(
                f"MOABB {entry['moabb_class']} metadata changed EEG count: "
                f"expected {contract['eeg_channel_count']}, got "
                f"{channel_types.get('eeg')}")
        catalog_channels = {
            str(value).upper() for value in (acquisition.sensors or ())}
        missing_native = [
            value for value in plan["native_channels"]
            if value not in catalog_channels]
        if missing_native:
            raise ValueError(
                f"MOABB {entry['moabb_class']} metadata is missing verified "
                f"native targets {missing_native}")
    return plan


def _validate_moabb_epochs(dataset_id, epochs, labels, metadata, fs_out,
                           target_channels):
    contract = PUBLIC_DATA_FORMATS[dataset_id]
    channel_types = epochs.get_channel_types()
    eeg_count = sum(value == "eeg" for value in channel_types)
    if eeg_count != int(contract["eeg_channel_count"]):
        raise ValueError(
            f"{dataset_id} yielded {eeg_count} EEG channels; verified format "
            f"requires {contract['eeg_channel_count']}")
    actual_fs = float(epochs.info["sfreq"])
    if not np.isclose(actual_fs, float(fs_out), rtol=0, atol=1e-6):
        raise ValueError(
            f"{dataset_id} resampling contract failed: expected {fs_out} Hz, "
            f"got {actual_fs} Hz")
    labels = np.asarray(labels).astype(str)
    expected_events = set(PUBLIC_MI_DATASETS[dataset_id]["source_label_map"])
    if set(labels) != expected_events:
        raise ValueError(
            f"{dataset_id} event contract changed: expected "
            f"{sorted(expected_events)}, got {sorted(set(labels))}")
    if len(labels) != len(epochs):
        raise ValueError(
            f"{dataset_id} returned {len(epochs)} epochs but {len(labels)} labels")
    required_metadata = {"subject", "session", "run"}
    missing_metadata = required_metadata - set(getattr(metadata, "columns", ()))
    if missing_metadata or len(metadata) != len(epochs):
        raise ValueError(
            f"{dataset_id} metadata contract failed: missing columns "
            f"{sorted(missing_metadata)}, rows={len(metadata)}, epochs={len(epochs)}")
    plan = public_channel_plan(dataset_id, target_channels)
    available = {str(value).upper() for value in epochs.ch_names}
    missing_native = [
        value for value in plan["native_channels"] if value not in available]
    if missing_native:
        raise ValueError(
            f"{dataset_id} is missing verified native target channels "
            f"{missing_native}; available={sorted(available)}")
    unexpected_missing = [
        value for value in plan["target_channels"]
        if value not in available and value not in plan["interpolated_channels"]]
    if unexpected_missing:
        raise ValueError(
            f"{dataset_id} has unapproved missing targets {unexpected_missing}")
    return plan


def _standardize_moabb_channels(epochs, dataset_id, target_channels):
    """Pick exact targets and interpolate only the audited BNCI omissions."""
    mne = _require_mne()
    plan = public_channel_plan(dataset_id, target_channels)
    available = {str(value).upper(): value for value in epochs.ch_names}
    missing = [
        value for value in plan["target_channels"] if value not in available]
    unapproved = set(missing) - set(plan["interpolated_channels"])
    if unapproved:
        raise ValueError(
            f"{dataset_id} cannot normalize unapproved missing channels "
            f"{sorted(unapproved)}")
    if missing:
        montage_names = mne.channels.get_builtin_montages()
        montage_name = (
            "colin27_1005" if "colin27_1005" in montage_names
            else "standard_1005")
        montage = mne.channels.make_standard_montage(montage_name)
        montage_case = {value.upper(): value for value in montage.ch_names}
        missing_case = []
        for value in missing:
            if value not in montage_case:
                raise ValueError(
                    f"cannot interpolate {dataset_id} channel {value}; it is "
                    f"absent from MNE montage {montage_name}")
            missing_case.append(montage_case[value])
        zeros = np.zeros(
            (len(epochs), len(missing_case), len(epochs.times)), dtype=float)
        missing_epochs = mne.EpochsArray(
            zeros, mne.create_info(
                missing_case, float(epochs.info["sfreq"]), ch_types="eeg"),
            events=np.array(epochs.events, copy=True),
            event_id=dict(epochs.event_id), tmin=float(epochs.tmin),
            baseline=epochs.baseline,
            metadata=(None if epochs.metadata is None
                      else epochs.metadata.copy()),
            verbose=False)
        epochs.add_channels([missing_epochs], force_update_info=True)
        epochs.set_montage(montage, on_missing="raise")
        prior_bads = list(epochs.info["bads"])
        if prior_bads:
            raise ValueError(
                f"{dataset_id} has unexpected pre-existing bad channels "
                f"{prior_bads}; refusing to mix them with virtual targets")
        epochs.info["bads"] = missing_case
        epochs.interpolate_bads(reset_bads=True, verbose=False)

    available = {str(value).upper(): value for value in epochs.ch_names}
    picks = [available[value] for value in plan["target_channels"]]
    epochs.pick(picks)
    data = _epochs_data(epochs)
    if data.shape[1] != len(plan["target_channels"]) or not np.isfinite(data).all():
        raise ValueError(
            f"{dataset_id} channel normalization produced invalid data shape "
            f"{data.shape}")
    return epochs, tuple(missing)


def read_ds008446_events(eeg_path, sfreq, n_times, source_map):
    """Reconstruct finger-cue onsets from ds008446 v1.0.1's event TSV.

    This release stores one row per EEG sample and scales ``onset`` by an
    additional sampling-rate factor. Reading the EDF marker channel therefore
    does not reliably yield codes 3..7. Consecutive labels are collapsed into
    block transitions and samples are reconstructed as ``round(onset * fs**2)``.
    """
    eeg_path = Path(eeg_path)
    suffix = "_eeg.edf"
    if not eeg_path.name.endswith(suffix):
        raise ValueError(f"unexpected ds008446 EEG filename: {eeg_path.name}")
    events_path = eeg_path.with_name(
        eeg_path.name[:-len(suffix)] + "_events.tsv")
    if not events_path.is_file():
        raise FileNotFoundError(
            f"missing BIDS event sidecar for {eeg_path}: {events_path}")

    sfreq = float(sfreq)
    n_times = int(n_times)
    if not np.isfinite(sfreq) or sfreq <= 0 or n_times <= 0:
        raise ValueError(
            f"invalid EDF dimensions for {eeg_path}: sfreq={sfreq}, "
            f"n_times={n_times}")

    transitions = []
    previous = None
    with events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"onset", "trial_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{events_path} must contain onset and trial_type columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                onset = float(row["onset"])
                code = int(float(row["trial_type"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid ds008446 event row {row_number} in "
                    f"{events_path}: {row}") from exc
            if not np.isfinite(onset) or onset < 0:
                raise ValueError(
                    f"invalid event onset {onset} at row {row_number} in "
                    f"{events_path}")
            if code != previous:
                transitions.append((onset, code, row_number))
                previous = code
    if not transitions:
        raise ValueError(f"no event transitions found in {events_path}")

    declared = {int(code) for code in source_map}
    events = []
    for onset, code, row_number in transitions:
        if code not in declared:
            continue
        sample = round(onset * sfreq**2)
        if not 0 <= sample < n_times:
            raise ValueError(
                f"reconstructed marker {code} from row {row_number} of "
                f"{events_path} is outside {eeg_path.name}: onset={onset}, "
                f"sample={sample}, valid range=0..{n_times - 1}, "
                f"sfreq={sfreq}")
        if events and sample <= events[-1][0]:
            raise ValueError(
                f"non-increasing reconstructed finger events in {events_path}: "
                f"sample {sample} follows {events[-1][0]}")
        events.append((sample, 0, code))
    if not events:
        raise ValueError(
            f"no declared finger events {sorted(declared)} reconstructed from "
            f"{events_path} for {eeg_path.name}")
    return np.asarray(events, dtype=int)


def import_openneuro_finger(bids_root, output_root=None, *, fs_out=250.0,
                            window_s=3.0, tmin=0.5,
                            target_channels=CYTON_MOTOR_CHANNELS,
                            bandpass=(1.0, 40.0)):
    """Import local ds008446 BIDS EDF files without inventing finger side."""
    mne = _require_mne()
    root = Path(bids_root)
    files = sorted(root.glob("sub-*/eeg/*_eeg.edf"))
    if not files:
        raise FileNotFoundError(f"no ds008446 *_eeg.edf files under {root}")
    source_map = get_public_dataset("openneuro_ds008446")["source_label_map"]
    Xs, labels, source_labels, subjects, sessions = [], [], [], [], []
    channels = [str(v).upper() for v in target_channels]
    n_samples = round(window_s * fs_out)
    for path in files:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        _rename_upper(raw)
        _validate_raw_format(
            "openneuro_ds008446", raw, path, target_channels)
        events = read_ds008446_events(
            path, raw.info["sfreq"], raw.n_times, source_map)
        raw, events = _resample_with_events(raw, events, fs_out)
        missing = [v for v in channels if v not in raw.ch_names]
        if missing:
            raise ValueError(f"{path} is missing requested EEG channels {missing}")
        raw.pick(channels)
        if bandpass:
            raw.filter(*bandpass, fir_design="firwin", skip_by_annotation="edge", verbose=False)
        present_codes = set(events[:, -1].astype(int).tolist())
        wanted_events = {
            code: int(code) for code in source_map if int(code) in present_codes}
        epochs = mne.Epochs(raw, events, event_id=wanted_events,
                            tmin=float(tmin), tmax=float(tmin + window_s),
                            baseline=None, preload=True, verbose=False)
        data = _epochs_data(epochs)[..., :n_samples]
        if not len(data):
            raise ValueError(
                f"no finger imagery epochs extracted from {path}: "
                f"{len(events)} reconstructed cues span samples "
                f"{events[0, 0]}..{events[-1, 0]} of {raw.n_times}, "
                f"epoch interval={tmin}..{tmin + window_s}s")
        src = epochs.events[:, -1].astype(str)
        canonical = np.asarray([source_map[v] for v in src])
        Xs.append(data.astype(np.float32)); labels.append(canonical); source_labels.append(src)
        subjects.append(np.repeat(_subject_from_path(path), len(data)))
        sessions.append(np.repeat(_session_from_name(path), len(data)))
    if not Xs:
        raise ValueError("no finger imagery epochs were extracted")
    out = public_archive_path("openneuro_ds008446", CATEGORY_FINGER, root=output_root)
    return write_standardized_archive(
        out, np.concatenate(Xs), np.concatenate(labels), np.concatenate(subjects),
        np.concatenate(sessions), source_dataset="openneuro_ds008446",
        source_labels=np.concatenate(source_labels), channel_labels=channels,
        fs=fs_out, units="volts", category=CATEGORY_FINGER,
        preprocessing={"bandpass_hz": list(bandpass) if bandpass else None,
                       "tmin_s": float(tmin), "window_s": float(window_s),
                       "native_format": PUBLIC_DATA_FORMATS["openneuro_ds008446"]["source_format"],
                       "native_sfreq": PUBLIC_DATA_FORMATS["openneuro_ds008446"]["native_sfreq"],
                       "resampled_hz": float(fs_out), "channel_subset": channels,
                       "interpolated_channels": [],
                       "event_reconstruction": "ds008446_v1.0.1_onset_times_sfreq_squared",
                       "laterality": "not published; canonical labels remain unsided"})


_PHYSIONET_IMAGERY_MAPS = {
    4: {"T1": "left_hand", "T2": "right_hand"},
    8: {"T1": "left_hand", "T2": "right_hand"},
    12: {"T1": "left_hand", "T2": "right_hand"},
    6: {"T1": "both_hands", "T2": "both_feet"},
    10: {"T1": "both_hands", "T2": "both_feet"},
    14: {"T1": "both_hands", "T2": "both_feet"},
}


def physionet_run_number(path):
    match = re.search(r"R(\d{2})(?:\.edf)?$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot infer EEGMMIDB run number from {Path(path).name!r}")
    return int(match.group(1))


def physionet_imagery_label_map(path, include_rest=False):
    """Context-qualified EEGMMIDB labels; rejects execution/baseline runs."""
    run = physionet_run_number(path)
    if run not in _PHYSIONET_IMAGERY_MAPS:
        raise ValueError(f"EEGMMIDB run {run:02d} is not a motor-imagery run")
    result = dict(_PHYSIONET_IMAGERY_MAPS[run])
    if include_rest:
        result["T0"] = IDLE
    return result


def import_physionet_imagery(paths, output_root=None, *, fs_out=250.0,
                              window_s=3.0, tmin=0.5, include_rest=True,
                              target_channels=CYTON_MOTOR_CHANNELS,
                              bandpass=(1.0, 40.0)):
    """Import only EEGMMIDB imagery runs and split hand/feet archives."""
    mne = _require_mne()
    if isinstance(paths, (str, Path)):
        paths = sorted(glob.glob(str(paths), recursive=True))
    paths = [Path(v) for v in paths]
    if not paths:
        raise FileNotFoundError("no PhysioNet EDF files given")
    channels = [str(v).upper() for v in target_channels]
    n_samples = round(window_s * fs_out)
    native_sfreqs_observed = set()
    buckets = {category: {key: [] for key in ("X", "labels", "source_labels", "subjects", "sessions")}
               for category in (CATEGORY_HAND, CATEGORY_FEET)}
    for path in paths:
        try:
            run_number = physionet_run_number(path)
            label_map = physionet_imagery_label_map(path, include_rest=include_rest)
        except ValueError:
            continue
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        _rename_upper(raw)
        _validate_raw_format(
            "physionet_eegmmidb", raw, path, target_channels)
        native_sfreqs_observed.add(float(raw.info["sfreq"]))
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        raw, events = _resample_with_events(raw, events, fs_out)
        missing = [v for v in channels if v not in raw.ch_names]
        if missing:
            raise ValueError(f"{path} is missing requested EEG channels {missing}")
        raw.pick(channels)
        if bandpass:
            raw.filter(*bandpass, fir_design="firwin", skip_by_annotation="edge", verbose=False)
        missing_annotations = set(label_map) - set(event_id)
        if missing_annotations:
            raise ValueError(
                f"{path} violates the verified PhysioNet event contract; "
                f"missing {sorted(missing_annotations)}, available={sorted(event_id)}")
        valid_event_id = {name: event_id[name] for name in label_map}
        epochs = mne.Epochs(raw, events, event_id=valid_event_id,
                            tmin=float(tmin), tmax=float(tmin + window_s),
                            baseline=None, preload=True, verbose=False)
        reverse = {value: name for name, value in valid_event_id.items()}
        data = _epochs_data(epochs)[..., :n_samples]
        src = np.asarray([reverse[int(v)] for v in epochs.events[:, -1]])
        canonical = np.asarray([label_map[v] for v in src])
        qualified_src = np.asarray([f"R{run_number:02d}:{v}" for v in src])
        subject, session = _subject_from_path(path), _session_from_name(path)
        categories = {public_label_category(v) for v in canonical if v != IDLE}
        for category in categories:
            keep = np.asarray([(v == IDLE or public_label_category(v) == category) for v in canonical])
            bucket = buckets[category]
            bucket["X"].append(data[keep].astype(np.float32))
            bucket["labels"].append(canonical[keep])
            bucket["source_labels"].append(qualified_src[keep])
            bucket["subjects"].append(np.repeat(subject, keep.sum()))
            bucket["sessions"].append(np.repeat(session, keep.sum()))
    outputs = {}
    for category, bucket in buckets.items():
        if not bucket["X"]:
            continue
        out = public_archive_path("physionet_eegmmidb", category, root=output_root)
        outputs[category] = write_standardized_archive(
            out, np.concatenate(bucket["X"]), np.concatenate(bucket["labels"]),
            np.concatenate(bucket["subjects"]), np.concatenate(bucket["sessions"]),
            source_dataset="physionet_eegmmidb",
            source_labels=np.concatenate(bucket["source_labels"]),
            channel_labels=channels, fs=fs_out, units="volts", category=category,
            preprocessing={"bandpass_hz": list(bandpass) if bandpass else None,
                           "tmin_s": float(tmin), "window_s": float(window_s),
                           "native_format": PUBLIC_DATA_FORMATS["physionet_eegmmidb"]["source_format"],
                           "native_sfreq": PUBLIC_DATA_FORMATS["physionet_eegmmidb"]["native_sfreq"],
                           "native_sfreqs_observed": sorted(native_sfreqs_observed),
                           "resampled_hz": float(fs_out), "channel_subset": channels,
                           "interpolated_channels": [],
                           "imagery_runs_only": sorted(_PHYSIONET_IMAGERY_MAPS)})
    if not outputs:
        raise ValueError("no usable EEGMMIDB imagery epochs extracted; provide runs 04/06/08/10/12/14")
    return outputs


def moabb_motor_imagery_options(events, *, tmin=0.5, window_s=3.0,
                                fs_out=250.0, bandpass=(1.0, 40.0)):
    """Build version-safe MOABB MotorImagery options for selected events."""
    events = list(events)
    if not events:
        raise ValueError("at least one MOABB motor-imagery event is required")
    fmin, fmax = bandpass if bandpass else (None, None)
    return {
        "events": events,
        # MOABB's used_events() compares its matched count to n_classes.
        # Some releases do not infer this value when events is explicit and
        # otherwise fail with: int < NoneType.
        "n_classes": len(events),
        "tmin": float(tmin),
        "tmax": float(tmin + window_s),
        "fmin": fmin,
        "fmax": fmax,
        "resample": float(fs_out),
    }


def import_moabb_dataset(dataset_id, output_root=None, *, subjects=None,
                         fs_out=250.0, window_s=3.0, tmin=0.5,
                         target_channels=CYTON_MOTOR_CHANNELS,
                         bandpass=(1.0, 40.0)):
    """Download/import a supported MOABB dataset and write category archives."""
    entry = get_public_dataset(dataset_id)
    if entry.get("importer") != "moabb":
        raise ValueError(f"{dataset_id!r} is not a MOABB-backed registry entry")
    try:
        import moabb.datasets as moabb_datasets
        from moabb.paradigms import MotorImagery
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MOABB import needs: python -m pip install '.[public-mi]'") from exc
    cls = getattr(moabb_datasets, entry["moabb_class"])
    if dataset_id == "ofner2017":
        dataset = cls(imagined=True, executed=False)
    elif dataset_id == "lee2019_mi":
        dataset = cls(train_run=True, test_run=False, resting_state=False)
    else:
        dataset = cls()
    subjects = list(subjects) if subjects is not None else list(dataset.subject_list)
    _validate_moabb_dataset(dataset_id, dataset, entry, target_channels)
    paradigm_options = moabb_motor_imagery_options(
        entry["source_label_map"], tmin=tmin, window_s=window_s,
        fs_out=fs_out, bandpass=bandpass)
    paradigm = MotorImagery(**paradigm_options)
    epochs, source_labels, metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects, return_epochs=True)
    plan = _validate_moabb_epochs(
        dataset_id, epochs, source_labels, metadata, fs_out, target_channels)
    epochs, interpolated = _standardize_moabb_channels(
        epochs, dataset_id, target_channels)
    channels = list(plan["target_channels"])
    n_samples = round(window_s * fs_out)
    data = _epochs_data(epochs)
    if data.shape[-1] < n_samples:
        raise ValueError(
            f"{dataset_id} epochs have only {data.shape[-1]} samples; "
            f"{n_samples} are required")
    X = data[..., :n_samples].astype(np.float32)
    if X.shape[1:] != (len(channels), n_samples) or not np.isfinite(X).all():
        raise ValueError(
            f"{dataset_id} standardized epoch contract failed: shape={X.shape}, "
            f"finite={bool(np.isfinite(X).all())}")
    source_labels = np.asarray(source_labels).astype(str)
    mapped = np.asarray([entry["source_label_map"][v] for v in source_labels])
    meta_subjects = np.asarray(metadata["subject"]).astype(str)
    meta_sessions = np.asarray([
        f"{session}:{run}" for session, run in zip(metadata["session"], metadata["run"])
    ]).astype(str)
    outputs = {}
    for category in MI_CATEGORIES:
        keep = np.asarray([(v == IDLE or public_label_category(v) == category) for v in mapped])
        if not keep.any() or not any(v != IDLE for v in mapped[keep]):
            continue
        out = public_archive_path(dataset_id, category, root=output_root)
        outputs[category] = write_standardized_archive(
            out, X[keep], mapped[keep], meta_subjects[keep], meta_sessions[keep],
            source_dataset=dataset_id, source_labels=source_labels[keep],
            channel_labels=channels, fs=fs_out, units="volts", category=category,
            preprocessing={"loader": "MOABB MotorImagery", "moabb_class": entry["moabb_class"],
                           "native_format": PUBLIC_DATA_FORMATS[dataset_id]["source_format"],
                           "native_sfreq": PUBLIC_DATA_FORMATS[dataset_id]["native_sfreq"],
                           "native_eeg_channels": PUBLIC_DATA_FORMATS[dataset_id]["eeg_channel_count"],
                           "bandpass_hz": list(bandpass) if bandpass else None,
                           "tmin_s": float(tmin), "window_s": float(window_s),
                           "resampled_hz": float(fs_out), "channel_subset": channels,
                           "interpolated_channels": list(interpolated),
                           "interpolation_method": ("MNE spherical spline" if interpolated else None),
                           "execution_included": False})
    if not outputs:
        raise ValueError(f"{dataset_id} yielded no registered category labels")
    return outputs


def registry_summary():
    return [{"id": key, "name": value["name"], "categories": list(value["categories"]),
             "access": value["access"], "license": value["license"],
             "importer": value["importer"]}
            for key, value in PUBLIC_MI_DATASETS.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="List or import licensed public motor-imagery EEG data")
    parser.add_argument("--list", action="store_true", help="print the dataset registry")
    parser.add_argument("--dataset", choices=sorted(PUBLIC_MI_DATASETS))
    parser.add_argument("--input", help="local BIDS root or EDF glob for direct importers")
    parser.add_argument("--output-root", default=None,
                        help="processed root (default: dataset/public/processed)")
    parser.add_argument("--subjects", help="MOABB subject numbers, comma-separated")
    args = parser.parse_args(argv)
    if args.list or not args.dataset:
        print(json.dumps(registry_summary(), indent=2))
        return 0
    entry = get_public_dataset(args.dataset)
    if entry["access"] != "public" or entry["importer"] is None:
        raise SystemExit(f"{args.dataset} cannot be auto-imported: {entry['notes']}")
    if entry["importer"] == "bids_finger":
        if not args.input:
            raise SystemExit("--input must point to the downloaded ds008446 BIDS root")
        result = {CATEGORY_FINGER: import_openneuro_finger(args.input, args.output_root)}
    elif entry["importer"] == "physionet":
        if not args.input:
            raise SystemExit("--input must be an EEGMMIDB EDF glob, e.g. dataset/public/raw/eegmmidb/S*/*.edf")
        result = import_physionet_imagery(args.input, args.output_root)
    else:
        subjects = ([int(v) for v in args.subjects.split(",") if v.strip()]
                    if args.subjects else None)
        result = import_moabb_dataset(args.dataset, args.output_root, subjects=subjects)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
