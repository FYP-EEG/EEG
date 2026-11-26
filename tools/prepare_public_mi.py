"""One-command public MI download, separation, training, selection and cleanup.

Edited by Anson
Date: 10/11/2026
Purpose: build four separate public category specialists without mixing public
         EEG into personal Cyton calibration or serving public models live.

Typical Windows command from the project root:

    python tools/prepare_public_mi.py --accept-licenses

The workflow downloads only public, automatable datasets; imports metadata-rich
category archives; evaluates candidate models with held-out subjects; publishes
one selected pretraining model per category; writes a router manifest; generates
cohort/source/literature tables and archive-backed figures; verifies every output;
and then applies an explicit safe cleanup policy. Use
``--cleanup-policy models-reports`` to retain only selected models and reports.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

from ML.motor_intent import MI_CATEGORIES, normalize_category
from ML.public_mi import (
    CYTON_MOTOR_CHANNELS,
    PUBLIC_DATA_FORMATS,
    PUBLIC_MI_DATASETS,
    get_public_dataset,
    import_moabb_dataset,
    import_openneuro_finger,
    import_physionet_imagery,
    load_standardized_archive,
    public_archive_path,
    public_channel_plan,
    public_model_dir,
    public_native_sfreqs,
)
from ML.train_public_mi import PUBLIC_PIPELINES, train_public_category

AUTOMATED_DATASETS = (
    "openneuro_ds008446",
    "physionet_eegmmidb",
    "bnci2014_001",
    "ofner2017",
    "cho2017",
    "lee2019_mi",
)
MOABB_DATASETS = tuple(
    dataset_id for dataset_id in AUTOMATED_DATASETS
    if PUBLIC_MI_DATASETS[dataset_id]["importer"] == "moabb"
)
PHYSIONET_IMAGERY_RUNS = (4, 6, 8, 10, 12, 14)
# Conservative planning estimates, not download claims. The subject limit scales
# these values; 1.35x leaves room for processed archives and candidate models.
ESTIMATED_FULL_DOWNLOAD_GB = {
    "openneuro_ds008446": 5.5,
    "physionet_eegmmidb": 1.7,
    # Verified from current NEMAR sourcedata manifests. Ofner downloads both
    # source sessions before its loader keeps imagery; Lee contains two large
    # per-subject MATLAB files.
    "bnci2014_001": 1.0,
    "ofner2017": 28.0,
    "cho2017": 11.0,
    "lee2019_mi": 65.0,
}
DEFAULT_WORK_ROOT = ROOT / "dataset" / "public" / "work"
DEFAULT_PROCESSED_ROOT = ROOT / "dataset" / "public" / "processed"
DEFAULT_MODEL_ROOT = ROOT / "models" / "public"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "public_mi"
LOCK_NAME = ".prepare_public_mi.lock"


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _comma_values(value):
    return tuple(part.strip().lower() for part in str(value).split(",") if part.strip())


def resolve_bandpass(value):
    """Parse the importer passband; the workflow default is broad 1--40 Hz."""
    try:
        values = tuple(float(part.strip()) for part in str(value).split(","))
    except ValueError as exc:
        raise ValueError("--import-bandpass must be LOW,HIGH in Hz") from exc
    if len(values) != 2 or values[0] <= 0 or values[0] >= values[1]:
        raise ValueError("--import-bandpass must satisfy 0 < LOW < HIGH")
    return values


def resolve_datasets(value):
    requested = _comma_values(value)
    if not requested or requested == ("all",):
        return AUTOMATED_DATASETS
    unknown = set(requested) - set(AUTOMATED_DATASETS)
    if unknown:
        raise ValueError(
            f"unsupported automated datasets {sorted(unknown)}; expected: "
            f"{', '.join(AUTOMATED_DATASETS)}")
    return tuple(dataset_id for dataset_id in AUTOMATED_DATASETS
                 if dataset_id in requested)


def resolve_pipelines(value):
    requested = _comma_values(value)
    if not requested or requested == ("auto",):
        result = ["riemann_lr", "riemann_svm", "fbcsp_lda"]
        if find_spec("torch") is not None and find_spec("braindecode") is not None:
            result.extend(("eegconformer", "tcformer"))
        return tuple(result)
    unknown = set(requested) - set(PUBLIC_PIPELINES)
    if unknown:
        raise ValueError(
            f"unknown pipelines {sorted(unknown)}; expected: {PUBLIC_PIPELINES}")
    return requested


def selected_subjects(dataset_id, limit=None):
    count = int(PUBLIC_MI_DATASETS[dataset_id]["subjects"])
    values = list(range(1, count + 1))
    if limit:
        values = values[:min(int(limit), count)]
    return values


def physionet_download_specs(subject_limit=None):
    """Only imagery EDFs; execution/baseline files are never downloaded."""
    specs = []
    for subject in selected_subjects("physionet_eegmmidb", subject_limit):
        subject_name = f"S{subject:03d}"
        for run in PHYSIONET_IMAGERY_RUNS:
            relative = f"{subject_name}/{subject_name}R{run:02d}.edf"
            specs.append({
                "relative": relative,
                "url": f"https://physionet.org/files/eegmmidb/1.0.0/{relative}",
            })
    return specs


def build_plan(dataset_ids, pipelines, subject_limit=None,
               import_bandpass=(1.0, 40.0)):
    import_bandpass = tuple(map(float, import_bandpass))
    minimum_nyquist = min(
        min(public_native_sfreqs(dataset_id)) / 2.0
        for dataset_id in dataset_ids)
    if import_bandpass[1] >= minimum_nyquist:
        raise ValueError(
            f"import high cutoff {import_bandpass[1]:g} Hz must be below the "
            f"selected sources' minimum Nyquist frequency {minimum_nyquist:g} Hz")
    categories = {category: [] for category in MI_CATEGORIES}
    datasets = []
    for dataset_id in dataset_ids:
        entry = get_public_dataset(dataset_id)
        subjects = selected_subjects(dataset_id, subject_limit)
        format_contract = PUBLIC_DATA_FORMATS[dataset_id]
        channel_plan = public_channel_plan(dataset_id)
        row = {
            "id": dataset_id,
            "name": entry["name"],
            "categories": list(entry["categories"]),
            "license": entry["license"],
            "subjects": subjects,
            "subject_count": len(subjects),
            "importer": entry["importer"],
            "source_format": format_contract["source_format"],
            "native_sfreq": format_contract["native_sfreq"],
            "verified_native_sfreqs": list(public_native_sfreqs(dataset_id)),
            "native_eeg_channels": format_contract["eeg_channel_count"],
            "target_channels": list(channel_plan["target_channels"]),
            "interpolated_channels": list(
                channel_plan["interpolated_channels"]),
            "cohort": entry["cohort"],
        }
        datasets.append(row)
        for category in entry["categories"]:
            categories[category].append(dataset_id)
    estimated_download = sum(
        ESTIMATED_FULL_DOWNLOAD_GB[row["id"]] *
        row["subject_count"] / int(PUBLIC_MI_DATASETS[row["id"]]["subjects"])
        for row in datasets)
    return {
        "datasets": datasets,
        "categories": {key: value for key, value in categories.items() if value},
        "pipelines": list(pipelines),
        "restricted_skipped": ["zenodo_18470278"],
        "estimated_download_gb": round(estimated_download, 1),
        "estimated_peak_free_space_gb": round(estimated_download * 1.35 + 2.0, 1),
        "estimate_note": "conservative planning estimate; actual provider sizes may differ",
        "import_bandpass_hz": list(import_bandpass),
        "model_strategy": "separate category specialists; no universal all-body classifier",
    }


def dependency_errors(dataset_ids, pipelines):
    missing = []
    if find_spec("mne") is None:
        missing.append("mne (install .[public-mi])")
    if any(dataset_id in MOABB_DATASETS for dataset_id in dataset_ids):
        if find_spec("moabb") is None:
            missing.append("moabb (install .[public-mi])")
        if find_spec("pandas") is None:
            missing.append("pandas (install .[public-mi])")
    if ("openneuro_ds008446" in dataset_ids and
            find_spec("openneuro") is None and shutil.which("aws") is None):
        missing.append(
            "openneuro-py (Python 3.10+) or AWS CLI for OpenNeuro download")
    if any(name in {"eegconformer", "tcformer"} for name in pipelines):
        if find_spec("torch") is None:
            missing.append("torch (install .[transformer])")
        if find_spec("braindecode") is None:
            missing.append("braindecode (install .[transformer])")
    return missing


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_http(url, destination, expected_sha256=None, retries=3):
    """Resumable HTTP download with atomic completion and optional SHA-256."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 is None or sha256_file(destination) == expected_sha256:
            return {"path": str(destination), "status": "reused",
                    "bytes": destination.stat().st_size}
        destination.unlink()
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, int(retries) + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "bci-project-public-mi/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", response.getcode())
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if expected_sha256 and sha256_file(partial) != expected_sha256:
                raise ValueError(f"SHA-256 mismatch for {url}")
            partial.replace(destination)
            return {"path": str(destination), "status": "downloaded",
                    "bytes": destination.stat().st_size}
        except (OSError, ValueError, urllib.error.URLError) as exc:
            if attempt >= retries:
                raise RuntimeError(
                    f"download failed after {retries} attempts: {url}: {exc}") from exc
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"unreachable download state for {url}")


def _physionet_checksums(cache_dir):
    checksum_path = Path(cache_dir) / "SHA256SUMS.txt"
    _download_http(
        "https://physionet.org/files/eegmmidb/1.0.0/SHA256SUMS.txt",
        checksum_path)
    result = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            result[parts[1].lstrip("*./")] = parts[0].lower()
    return result


def download_physionet(target, subject_limit=None, workers=4):
    target = Path(target)
    specs = physionet_download_specs(subject_limit)
    checksums = _physionet_checksums(target)
    results = []

    def download_one(spec):
        expected = checksums.get(spec["relative"])
        if expected is None:
            raise ValueError(f"PhysioNet checksum missing for {spec['relative']}")
        return _download_http(
            spec["url"], target / spec["relative"], expected_sha256=expected)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(download_one, spec): spec for spec in specs}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"    PhysioNet {completed}/{len(specs)}: {Path(result['path']).name} "
                  f"({result['status']})")
    return results


def download_openneuro(target, subject_limit=None):
    subjects = selected_subjects("openneuro_ds008446", subject_limit)
    includes = [f"sub-{subject:02d}/eeg" for subject in subjects]
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"    OpenNeuro ds008446: {len(subjects)} subject(s)")
    if find_spec("openneuro") is not None:
        import openneuro
        openneuro.download(
            dataset="ds008446", target_dir=str(target), include=includes)
    elif shutil.which("aws") is not None:
        command = [
            "aws", "s3", "sync", "--no-sign-request",
            "s3://openneuro.org/ds008446", str(target), "--exclude", "*",
        ]
        for include in includes:
            command.extend(("--include", include + "/*"))
        subprocess.run(command, check=True)
    else:  # pragma: no cover - dependency preflight handles this
        raise ImportError(
            "OpenNeuro download needs openneuro-py (Python 3.10+) or AWS CLI")
    files = sorted(target.glob("sub-*/eeg/*_eeg.edf"))
    expected_minimum = 4 * len(subjects)
    if len(files) < expected_minimum:
        raise RuntimeError(
            f"OpenNeuro download has {len(files)} EDF files; expected at least {expected_minimum}")
    return files


@contextmanager
def moabb_download_directory(path):
    """Confine MOABB/MNE downloads to workflow-owned storage, then restore config."""
    import mne
    import moabb

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    original = mne.get_config("MNE_DATA", default=None)
    old_environment = os.environ.get("MNE_DATA")
    os.environ["MNE_DATA"] = str(path)
    setter = getattr(moabb, "set_download_dir", None)
    if setter is None:
        from moabb.utils import set_download_dir as setter
    setter(str(path))
    try:
        yield
    finally:
        mne.set_config("MNE_DATA", original, set_env=False)
        if old_environment is None:
            os.environ.pop("MNE_DATA", None)
        else:
            os.environ["MNE_DATA"] = old_environment


def directory_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def available_gb(path):
    probe = Path(path).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024 ** 3)


def ensure_disk_space(path, required_gb):
    available = available_gb(path)
    if available < float(required_gb):
        raise RuntimeError(
            f"insufficient free space at {path}: {available:.1f} GiB available, "
            f"planning threshold is {float(required_gb):.1f} GiB. Use another "
            "--work-root or explicitly lower --min-free-gb after checking provider sizes.")
    return available


def safe_cleanup(path, allowed_root):
    """Delete only a child of the configured workflow root."""
    path = Path(path).resolve()
    allowed_root = Path(allowed_root).resolve()
    if path == allowed_root or allowed_root not in path.parents:
        raise ValueError(f"refusing cleanup outside workflow root: {path}")
    if path.exists():
        shutil.rmtree(path)
    return True


def cleanup_processed_datasets(dataset_ids, processed_root):
    """Delete only named dataset children, never the processed-data root."""
    processed_root = Path(processed_root).resolve()
    removed = []
    bytes_removed = 0
    for dataset_id in dataset_ids:
        if dataset_id not in AUTOMATED_DATASETS:
            raise ValueError(f"refusing cleanup of unknown dataset {dataset_id!r}")
        child = processed_root / dataset_id
        if child.exists():
            size = directory_size(child)
            safe_cleanup(child, processed_root)
            bytes_removed += size
            removed.append(str(child))
    return {"dataset_directories": removed, "bytes_removed": bytes_removed}


def _archive_is_reusable(path, dataset_id, category, minimum_subjects,
                         import_bandpass=(1.0, 40.0)):
    path = Path(path)
    if not path.exists():
        return False
    try:
        data = load_standardized_archive(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    entry = get_public_dataset(dataset_id)
    expected_channels = [str(value).upper() for value in CYTON_MOTOR_CHANNELS]
    expected_bandpass = list(map(float, import_bandpass))
    actual_bandpass = data["preprocessing"].get("bandpass_hz")
    actual_bandpass = (list(map(float, actual_bandpass))
                       if actual_bandpass is not None else None)
    return (data["source_dataset"] == dataset_id and
            data["category"] == category and
            data["license"] == entry["license"] and
            data["units"] == "volts" and
            abs(float(data["fs"]) - 250.0) < 1e-9 and
            data["channel_labels"].tolist() == expected_channels and
            data["X"].shape[1:] == (len(expected_channels), 750) and
            actual_bandpass == expected_bandpass and
            len(set(data["subjects"].tolist())) >= int(minimum_subjects))


def existing_dataset_archives(dataset_id, processed_root, subject_limit=None,
                              import_bandpass=(1.0, 40.0)):
    entry = get_public_dataset(dataset_id)
    minimum_subjects = len(selected_subjects(dataset_id, subject_limit))
    paths = {
        category: public_archive_path(dataset_id, category, root=processed_root)
        for category in entry["categories"]
    }
    return paths if all(
        _archive_is_reusable(
            path, dataset_id, category, minimum_subjects, import_bandpass)
        for category, path in paths.items()) else None


def import_dataset(dataset_id, *, stage, processed_root, subject_limit=None,
                   workers=4, import_bandpass=(1.0, 40.0)):
    entry = get_public_dataset(dataset_id)
    downloads = Path(stage) / "downloads" / dataset_id
    subjects = selected_subjects(dataset_id, subject_limit)
    if entry["importer"] == "bids_finger":
        download_openneuro(downloads, subject_limit=subject_limit)
        output = import_openneuro_finger(
            downloads, output_root=processed_root, bandpass=import_bandpass)
        return {"finger": output}
    if entry["importer"] == "physionet":
        download_physionet(
            downloads, subject_limit=subject_limit, workers=workers)
        return import_physionet_imagery(
            str(downloads / "S*" / "*.edf"), output_root=processed_root,
            bandpass=import_bandpass)
    if entry["importer"] == "moabb":
        return import_moabb_dataset(
            dataset_id, output_root=processed_root, subjects=subjects,
            bandpass=import_bandpass)
    raise ValueError(f"dataset {dataset_id!r} has no automated importer")


def select_best_candidate(candidates):
    valid = [row for row in candidates if row.get("status") == "success"]
    if not valid:
        raise ValueError("no candidate model trained successfully")
    dataset_valid = [
        row for row in valid if row.get("dataset_accuracy") is not None]
    if dataset_valid:
        # Cross-provider generalization is the stronger public-data criterion.
        selected = max(dataset_valid, key=lambda row: (
            float(row["dataset_accuracy"]),
            -float(row.get("dataset_std") or 0.0),
            float(row["accuracy"]), -float(row.get("std", 0.0)),
            str(row["pipeline"])))
        selected["selection_criterion"] = (
            "highest class-complete leave-one-dataset-out balanced accuracy; "
            "lower dataset std and subject-held-out accuracy tie-break")
        return selected
    # A single-source category can only use subject-held-out evidence.
    selected = max(valid, key=lambda row: (
        float(row["accuracy"]), -float(row.get("std", 0.0)),
        str(row["pipeline"])))
    selected["selection_criterion"] = (
        "highest subject-held-out balanced accuracy; lower std tie-break")
    return selected


def _publish_selected_model(category, selected, candidates, model_root):
    source_path = Path(selected["path"])
    bundle = joblib.load(source_path)
    if not isinstance(bundle, dict) or bundle.get("kind") != "PublicMICategoryModel":
        raise ValueError(f"candidate is not a PublicMICategoryModel: {source_path}")
    if bundle.get("category") != category or bundle.get("runtime_ready") is not False:
        raise ValueError(f"candidate metadata is unsafe for category {category}")
    bundle["selection"] = {
        "selected_pipeline": selected["pipeline"],
        "criterion": selected["selection_criterion"],
        "candidates": [{key: value for key, value in row.items() if key != "path"}
                       for row in candidates],
    }
    destination_dir = public_model_dir(category, root=model_root)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "category_model.joblib"
    temporary = destination.with_name(destination.name + ".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(destination)
    info = {key: value for key, value in bundle.items() if key != "model"}
    _json_write(destination_dir / "category_model_info.json", info)
    return destination, bundle


def train_and_select_category(category, archives, pipelines, *, stage,
                              model_root, splits=5, epochs=40, batch_size=32,
                              random_state=42):
    candidates = []
    for index, pipeline in enumerate(pipelines):
        candidate_root = Path(stage) / "candidate_models" / pipeline
        print(f"    candidate {pipeline}")
        try:
            result = train_public_category(
                archives, category=category, pipeline=pipeline,
                output_root=candidate_root, n_splits=splits,
                epochs=epochs, batch_size=batch_size,
                random_state=random_state + index, verbose=True)
            candidates.append({"status": "success", **result})
        except Exception as exc:  # noqa: BLE001 - preserve other category candidates
            print(f"    WARNING: {pipeline} failed: {exc}")
            candidates.append({
                "status": "failed", "pipeline": pipeline,
                "error": f"{type(exc).__name__}: {exc}",
            })
    selected = select_best_candidate(candidates)
    path, bundle = _publish_selected_model(
        category, selected, candidates, model_root)
    return {
        "category": category,
        "selected_pipeline": selected["pipeline"],
        "accuracy": selected["accuracy"],
        "std": selected["std"],
        "dataset_accuracy": selected.get("dataset_accuracy"),
        "dataset_std": selected.get("dataset_std"),
        "dataset_folds": selected.get("dataset_folds", 0),
        "selection_criterion": selected["selection_criterion"],
        "chance": selected["chance"],
        "path": str(path),
        "runtime_ready": False,
        "labels": bundle["labels"],
        "source_datasets": bundle["source_datasets"],
        "candidates": [
            {key: value for key, value in row.items() if key != "path"}
            for row in candidates
        ],
    }


def verify_final_models(category_results):
    for category, result in category_results.items():
        path = Path(result["path"])
        if not path.exists():
            raise FileNotFoundError(f"selected {category} model missing: {path}")
        bundle = joblib.load(path)
        if (not isinstance(bundle, dict) or
                bundle.get("kind") != "PublicMICategoryModel" or
                bundle.get("category") != category or
                bundle.get("runtime_ready") is not False):
            raise ValueError(f"selected {category} model failed safety verification")
        if not bundle.get("source_datasets") or len(bundle.get("labels", [])) < 2:
            raise ValueError(f"selected {category} model lacks source/classes")
        if category == "hand" and set(bundle["labels"]) != {
                "left_hand", "right_hand"}:
            raise ValueError(
                "selected hand model is not the harmonized left/right task")
        if (category == "feet" and
                bundle.get("training_policy", {}).get("id") !=
                "source-complete-idle-versus-both-feet-v1"):
            raise ValueError(
                "selected feet model does not enforce source-complete balancing")
    return True


def _relative_to_root(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_router_manifest(category_results, plan, model_root):
    manifest = {
        "kind": "PublicMICategoryRouterManifest",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_ready": False,
        "purpose": "category pretraining/benchmark suite; personal calibration still required",
        "routing": "exact personal task set, then compatible personal category specialist",
        "universal_all_body_model": False,
        "restricted_skipped": plan["restricted_skipped"],
        "categories": {
            category: {
                "model": _relative_to_root(result["path"]),
                "pipeline": result["selected_pipeline"],
                "subject_held_out_balanced_accuracy": result["accuracy"],
                "subject_std": result["std"],
                "dataset_held_out_balanced_accuracy": result.get("dataset_accuracy"),
                "dataset_std": result.get("dataset_std"),
                "selection_criterion": result["selection_criterion"],
                "chance": result["chance"],
                "labels": result["labels"],
                "source_datasets": result["source_datasets"],
                "runtime_ready": False,
            }
            for category, result in category_results.items()
        },
    }
    path = Path(model_root) / "model_manifest.json"
    _json_write(path, manifest)
    return path, manifest


def _print_license_summary(plan):
    print("\nDatasets and licenses:")
    for row in plan["datasets"]:
        print(f"  {row['id']:<24} {row['license']:<14} "
              f"subjects={row['subject_count']}")
    print("  zenodo_18470278       SKIPPED: restricted files and unstated license")
    print("\nVerified format/channel plan:")
    print(f"  archive passband: {plan['import_bandpass_hz'][0]:g}-"
          f"{plan['import_bandpass_hz'][1]:g} Hz")
    for row in plan["datasets"]:
        interpolated = row["interpolated_channels"]
        handling = (f"interpolate {','.join(interpolated)}"
                    if interpolated else "all targets native")
        print(f"  {row['id']:<24} {row['source_format']}; {handling}")
    if any(row["id"] == "bnci2014_001" for row in plan["datasets"]):
        print("\nDerived BNCI2014-001 archives/models are for local research use; "
              "the registry blocks derived-data redistribution under CC-BY-ND.")


def _acquire_lock(work_root, force=False):
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    lock = work_root / LOCK_NAME
    if lock.exists() and not force:
        raise RuntimeError(
            f"workflow lock exists: {lock}. Another run may be active; "
            "use --force-unlock only after checking it is stale.")
    lock.write_text(json.dumps({"pid": os.getpid(), "started_at": timestamp()}),
                    encoding="utf-8")
    return lock


def run_workflow(args):
    dataset_ids = resolve_datasets(args.datasets)
    pipelines = resolve_pipelines(args.pipelines)
    import_bandpass = resolve_bandpass(args.import_bandpass)
    plan = build_plan(
        dataset_ids, pipelines, args.subjects_per_dataset, import_bandpass)
    if args.keep_downloads and args.cleanup_policy == "models-reports":
        raise ValueError(
            "--keep-downloads conflicts with --cleanup-policy models-reports")
    plan["cleanup_policy"] = args.cleanup_policy
    plan["evidence_hash_inputs"] = bool(args.hash_evidence_inputs)
    _print_license_summary(plan)
    if args.dry_run:
        print("\nDRY RUN: no downloads, training, model writes or cleanup performed.")
        print(json.dumps(plan, indent=2))
        return {"status": "dry-run", "plan": plan}
    if not args.accept_licenses:
        if not sys.stdin.isatty():
            raise RuntimeError("non-interactive use requires --accept-licenses")
        answer = input("\nContinue under these dataset licenses? Type YES: ").strip()
        if answer != "YES":
            raise RuntimeError("license acceptance not confirmed")

    missing = dependency_errors(dataset_ids, pipelines)
    if missing:
        raise RuntimeError(
            "missing optional dependencies:\n  - " + "\n  - ".join(missing) +
            "\nInstall data support with `python -m pip install -e \".[public-mi]\"` "
            "and Transformer support, if selected, with "
            "`python -m pip install -e \".[transformer]\"`.")

    work_root = Path(args.work_root)
    processed_root = Path(args.processed_root)
    model_root = Path(args.model_root)
    report_root = Path(args.report_root)
    pending_download_gb = 0.0
    for row in plan["datasets"]:
        reusable = (None if args.refresh else existing_dataset_archives(
            row["id"], processed_root, args.subjects_per_dataset,
            import_bandpass))
        if not reusable:
            pending_download_gb += (
                ESTIMATED_FULL_DOWNLOAD_GB[row["id"]] * row["subject_count"] /
                int(PUBLIC_MI_DATASETS[row["id"]]["subjects"]))
    required_gb = (float(args.min_free_gb) if args.min_free_gb is not None
                   else pending_download_gb * 1.35 + 2.0)
    free_gb = ensure_disk_space(work_root, required_gb)
    print(f"\nDisk preflight: {free_gb:.1f} GiB free; "
          f"planning threshold {required_gb:.1f} GiB")
    plan["pending_estimated_download_gb"] = round(pending_download_gb, 1)
    plan["disk_free_gb_at_start"] = round(free_gb, 1)
    plan["disk_threshold_gb"] = round(required_gb, 1)
    run_id = args.run_id or "auto"
    stage = work_root / run_id
    lock = _acquire_lock(work_root, force=args.force_unlock)
    report = {
        "kind": "PublicMIAutomationReport", "version": 1,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running", "plan": plan, "stage": str(stage),
        "datasets": {}, "categories": {}, "cleanup": {},
    }
    report_path = report_root / f"public_mi_workflow_{timestamp()}.json"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        archives_by_category = {category: [] for category in MI_CATEGORIES}
        moabb_selected = any(value in MOABB_DATASETS for value in dataset_ids)
        context = (moabb_download_directory(stage / "moabb_cache")
                   if moabb_selected else _null_context())
        with context:
            for dataset_id in dataset_ids:
                print(f"\n[{dataset_id}]")
                reused = None if args.refresh else existing_dataset_archives(
                    dataset_id, processed_root, args.subjects_per_dataset,
                    import_bandpass)
                if reused:
                    outputs = reused
                    state = "reused-processed"
                    print("  using verified processed archive(s)")
                else:
                    outputs = import_dataset(
                        dataset_id, stage=stage,
                        processed_root=processed_root,
                        subject_limit=args.subjects_per_dataset,
                        workers=args.workers,
                        import_bandpass=import_bandpass)
                    state = "downloaded-imported"
                checked_outputs = {}
                for category, path in outputs.items():
                    category = normalize_category(category)
                    archive = load_standardized_archive(path)
                    if archive["source_dataset"] != dataset_id:
                        raise ValueError(f"archive source mismatch for {path}")
                    archives_by_category[category].append(str(path))
                    checked_outputs[category] = str(path)
                report["datasets"][dataset_id] = {
                    "status": state, "archives": checked_outputs,
                }
                _json_write(report_path, report)

        populated = {key: value for key, value in archives_by_category.items() if value}
        expected_categories = set(plan["categories"])
        if set(populated) != expected_categories:
            raise ValueError(
                f"category separation incomplete: expected {sorted(expected_categories)}, "
                f"found {sorted(populated)}")

        print("\nTraining and selecting separate category specialists")
        for category in MI_CATEGORIES:
            archives = populated.get(category)
            if not archives:
                continue
            print(f"\n[{category}] archives={len(archives)}")
            result = train_and_select_category(
                category, archives, pipelines, stage=stage,
                model_root=model_root, splits=args.splits,
                epochs=args.epochs, batch_size=args.batch_size,
                random_state=args.seed)
            report["categories"][category] = result
            _json_write(report_path, report)

        verify_final_models(report["categories"])
        manifest_path, manifest = write_router_manifest(
            report["categories"], plan, model_root)
        report["manifest"] = str(manifest_path)
        report["manifest_verified_categories"] = sorted(manifest["categories"])

        print("\nGenerating archive-backed evidence, cohort tables and figures")
        from ML.public_mi_evidence import generate_public_mi_evidence
        evidence_root = report_root / "evidence"
        evidence, evidence_md, evidence_json, evidence_csv = \
            generate_public_mi_evidence(
                processed_root, evidence_root, model_root=model_root,
                categories=tuple(report["categories"]), n_splits=args.splits,
                seed=args.seed, content_hash=args.hash_evidence_inputs,
                command=" ".join(sys.argv), project_root=ROOT)
        failed_evidence = {
            category: row.get("status")
            for category, row in evidence["categories"].items()
            if row.get("status") != "success"
        }
        if failed_evidence:
            raise RuntimeError(
                f"evidence generation did not complete every category: "
                f"{failed_evidence}")
        report["evidence"] = {
            "markdown": str(evidence_md), "json": str(evidence_json),
            "source_literature_csv": str(evidence_csv),
            "archive_inventory_csv": str(
                evidence_root / "dataset_archive_inventory.csv"),
            "category_status": {
                key: value["status"]
                for key, value in evidence["categories"].items()},
        }
        report["status"] = "success"
        report["finished_at"] = datetime.now().isoformat(timespec="seconds")

        cleanup = {
            "policy": args.cleanup_policy,
            "selected_models_preserved": True,
            "reports_and_figures_preserved": True,
            "personal_data_preserved": True,
        }
        if args.keep_downloads:
            cleanup.update({
                "staging_removed": False, "reason": "--keep-downloads",
                "stage": str(stage), "staging_bytes_retained": directory_size(stage),
            })
        else:
            staging_bytes = directory_size(stage)
            safe_cleanup(stage, work_root)
            cleanup.update({
                "staging_removed": True,
                "staging_scope": (
                    "workflow-owned downloads, MOABB cache and candidate models"),
                "staging_bytes_removed": staging_bytes,
            })
        if args.cleanup_policy == "models-reports":
            processed_cleanup = cleanup_processed_datasets(
                dataset_ids, processed_root)
            cleanup.update({
                "processed_archives_preserved": False,
                "processed_dataset_directories_removed":
                    processed_cleanup["dataset_directories"],
                "processed_bytes_removed": processed_cleanup["bytes_removed"],
            })
        else:
            cleanup["processed_archives_preserved"] = True
        cleanup["performed"] = bool(
            cleanup.get("staging_removed") or
            cleanup.get("processed_dataset_directories_removed"))
        report["cleanup"] = cleanup
        _json_write(report_path, report)
        _json_write(report_root / "latest_workflow.json", report)
        print("\nSUCCESS")
        print(f"  manifest: {manifest_path}")
        print(f"  report:   {report_path}")
        print(f"  cleanup:  {args.cleanup_policy} completed safely")
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["finished_at"] = datetime.now().isoformat(timespec="seconds")
        report["error"] = f"{type(exc).__name__}: {exc}"
        if args.cleanup_on_failure:
            removed = directory_size(stage)
            safe_cleanup(stage, work_root)
            report["cleanup"] = {
                "performed": True, "on_failure": True,
                "bytes_removed": removed,
            }
        else:
            report["cleanup"] = {
                "performed": False,
                "reason": "failure staging retained for resumable diagnosis",
                "stage": str(stage),
            }
        _json_write(report_path, report)
        _json_write(report_root / "latest_workflow.json", report)
        raise
    finally:
        lock.unlink(missing_ok=True)


@contextmanager
def _null_context():
    yield


def make_parser():
    parser = argparse.ArgumentParser(
        description="Download, separate, train, select and safely clean public MI datasets")
    parser.add_argument(
        "--datasets", default="all",
        help="comma-separated registry IDs or all (restricted data is always skipped)")
    parser.add_argument(
        "--pipelines", default="auto",
        help=(
            "auto or comma-separated riemann_lr,riemann_svm,fbcsp_lda,"
            "eegconformer,tcformer"))
    parser.add_argument(
        "--subjects-per-dataset", type=int, default=None,
        help="limit each dataset for a pilot; default uses every subject")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--import-bandpass", default="1,40", metavar="LOW,HIGH",
        help=(
            "archive import passband in Hz; default 1,40. Use 0.3,40 only "
            "with separate processed/model roots for full Ofner MRCP experiments"))
    parser.add_argument(
        "--min-free-gb", type=float, default=None,
        help="override conservative automatic free-space threshold")
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--run-id", default="auto")
    parser.add_argument(
        "--accept-licenses", action="store_true",
        help="required for unattended download; confirms displayed source licenses")
    parser.add_argument(
        "--refresh", action="store_true",
        help="reimport instead of reusing verified processed archives; resumable staging files may still be reused")
    parser.add_argument(
        "--keep-downloads", action="store_true",
        help="keep workflow-owned downloads/cache after successful verification")
    parser.add_argument(
        "--cleanup-policy", choices=("staging-only", "models-reports"),
        default="staging-only",
        help=(
            "staging-only keeps standardized archives; models-reports also "
            "deletes processed dataset children after verified evidence generation"))
    parser.add_argument(
        "--hash-evidence-inputs", action="store_true",
        help="SHA-256 every processed archive in the evidence report before cleanup")
    parser.add_argument(
        "--cleanup-on-failure", action="store_true",
        help="delete staging after failure; default retains resumable diagnostic files")
    parser.add_argument("--force-unlock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    run_workflow(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
