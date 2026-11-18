"""Licensed public motor-imagery dataset registry and standardized archives.

Edited by Anson
Date: 10/11/2026
Purpose: keep source task semantics, licensing, provenance, and leakage-safe
         subject/session metadata attached to every imported public EEG epoch.

Public epochs are benchmark/pretraining data. They are not personal Cyton
calibration and are never silently concatenated with personal recordings.
"""

import argparse
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
CYTON_MOTOR_CHANNELS = ("FC3", "FC4", "C3", "CZ", "C4", "CP3", "CP4", "PZ")

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
        "importer": "physionet",
        "source_label_map": {
            "R04:T0": IDLE, "R04:T1": "left_hand", "R04:T2": "right_hand",
            "R08:T0": IDLE, "R08:T1": "left_hand", "R08:T2": "right_hand",
            "R12:T0": IDLE, "R12:T1": "left_hand", "R12:T2": "right_hand",
            "R06:T0": IDLE, "R06:T1": "both_hands", "R06:T2": "both_feet",
            "R10:T0": IDLE, "R10:T1": "both_hands", "R10:T2": "both_feet",
            "R14:T0": IDLE, "R14:T1": "both_hands", "R14:T2": "both_feet",
        },
        "notes": "T1/T2 meanings depend on run number. Execution runs 3/5/7/9/11/13 must not be imported as imagery.",
    },
    "bnci2014_001": {
        "name": "BCI Competition IV 2a",
        "categories": (CATEGORY_HAND, CATEGORY_FEET),
        "paradigm": "imagery",
        "modality": "EEG",
        "subjects": 9,
        "channels": 22,
        "sample_rate": 250,
        "format": "GDF via MOABB",
        "license": "CC-BY-ND-4.0",
        "redistribute_derived": False,
        "access": "public",
        "download": "https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html",
        "citation": "Tangermann et al. (2012), doi:10.3389/fnins.2012.00055",
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
        "format": "MAT via MOABB",
        "license": "CC0-1.0",
        "redistribute_derived": True,
        "access": "public",
        "download": "https://doi.org/10.5524/100542",
        "citation": "Lee et al. (2019), doi:10.1093/gigascience/giz002",
        "importer": "moabb",
        "moabb_class": "Lee2019_MI",
        "source_label_map": {"left_hand": "left_hand", "right_hand": "right_hand"},
        "notes": "Importer uses labelled training runs; unlabelled online test runs are excluded.",
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
    np.savez_compressed(
        path, X=X, labels=labels, subjects=subjects, sessions=sessions,
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
    return match.group(1) if match else Path(path).stem


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


def import_openneuro_finger(bids_root, output_root=None, *, fs_out=250.0,
                            window_s=3.0, tmin=0.5,
                            target_channels=CYTON_MOTOR_CHANNELS,
                            bandpass=(8.0, 30.0)):
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
        marker = next((v for v in ("MARKER", "STATUS", "STI 014") if v in raw.ch_names), None)
        if marker is None:
            raise ValueError(f"{path} has no trigger/Marker channel")
        events = mne.find_events(raw, stim_channel=marker, shortest_event=1, verbose=False)
        raw, events = _resample_with_events(raw, events, fs_out)
        missing = [v for v in channels if v not in raw.ch_names]
        if missing:
            raise ValueError(f"{path} is missing requested EEG channels {missing}")
        raw.pick(channels)
        if bandpass:
            raw.filter(*bandpass, fir_design="firwin", skip_by_annotation="edge", verbose=False)
        wanted_events = {code: int(code) for code in source_map}
        epochs = mne.Epochs(raw, events, event_id=wanted_events,
                            tmin=float(tmin), tmax=float(tmin + window_s),
                            baseline=None, preload=True, verbose=False)
        data = epochs.get_data(copy=True)[..., :n_samples]
        if not len(data):
            continue
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
                       "resampled_hz": float(fs_out), "channel_subset": channels,
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
                              bandpass=(8.0, 30.0)):
    """Import only EEGMMIDB imagery runs and split hand/feet archives."""
    mne = _require_mne()
    if isinstance(paths, (str, Path)):
        paths = sorted(glob.glob(str(paths), recursive=True))
    paths = [Path(v) for v in paths]
    if not paths:
        raise FileNotFoundError("no PhysioNet EDF files given")
    channels = [str(v).upper() for v in target_channels]
    n_samples = round(window_s * fs_out)
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
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        raw, events = _resample_with_events(raw, events, fs_out)
        missing = [v for v in channels if v not in raw.ch_names]
        if missing:
            raise ValueError(f"{path} is missing requested EEG channels {missing}")
        raw.pick(channels)
        if bandpass:
            raw.filter(*bandpass, fir_design="firwin", skip_by_annotation="edge", verbose=False)
        valid_event_id = {name: event_id[name] for name in label_map if name in event_id}
        if not valid_event_id:
            continue
        epochs = mne.Epochs(raw, events, event_id=valid_event_id,
                            tmin=float(tmin), tmax=float(tmin + window_s),
                            baseline=None, preload=True, verbose=False)
        reverse = {value: name for name, value in valid_event_id.items()}
        data = epochs.get_data(copy=True)[..., :n_samples]
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
                           "resampled_hz": float(fs_out), "channel_subset": channels,
                           "imagery_runs_only": sorted(_PHYSIONET_IMAGERY_MAPS)})
    if not outputs:
        raise ValueError("no usable EEGMMIDB imagery epochs extracted; provide runs 04/06/08/10/12/14")
    return outputs


def import_moabb_dataset(dataset_id, output_root=None, *, subjects=None,
                         fs_out=250.0, window_s=3.0, tmin=0.5,
                         target_channels=CYTON_MOTOR_CHANNELS,
                         bandpass=(8.0, 30.0)):
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
    events = list(entry["source_label_map"])
    fmin, fmax = bandpass if bandpass else (None, None)
    paradigm = MotorImagery(events=events, tmin=float(tmin), tmax=float(tmin + window_s),
                            fmin=fmin, fmax=fmax, resample=float(fs_out))
    epochs, source_labels, metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects, return_epochs=True)
    channels = [str(v).upper() for v in target_channels]
    available = {str(v).upper(): v for v in epochs.ch_names}
    missing = [v for v in channels if v not in available]
    if missing:
        raise ValueError(f"{dataset_id} is missing requested channels {missing}")
    epochs.pick([available[v] for v in channels])
    n_samples = round(window_s * fs_out)
    X = epochs.get_data(copy=True)[..., :n_samples].astype(np.float32)
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
                           "bandpass_hz": list(bandpass) if bandpass else None,
                           "tmin_s": float(tmin), "window_s": float(window_s),
                           "resampled_hz": float(fs_out), "channel_subset": channels,
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
