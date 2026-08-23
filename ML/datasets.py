"""
Author: Anson
Created on: 23/8/2026
Purpose: Loaders that turn PUBLIC datasets and YOUR OWN recordings into the one
         array shape the rest of the project uses.

WHY THIS EXISTS
---------------
Training happens in three places and they all produced different shapes:

    Tsinghua Benchmark .mat   (64, 1500, 40, 6)   channels, time, target, block
    PhysioNet EEGMMIDB .edf   raw continuous, 160 Hz, 64 channels
    Your calibration .npz     (n_windows, 8, 750)

Every classifier in this project expects **(trials, channels, samples)** at 250 Hz.
Without a common loader each notebook re-implements the reshaping, which is exactly
how the 64-channel indices ended up hard-coded into an 8-channel pipeline.

These functions are also what let you TRAIN ON PUBLIC DATA AND FINE-TUNE ON YOUR OWN:
both sources come back in the same format, so they can be concatenated.

    from ML.datasets import load_benchmark_mat, load_calibration_npz, resample_to

    Xp, yp, meta = load_benchmark_mat("S1.mat", freq_map={10: 0, 20: 1})
    Xo, yo, _    = load_calibration_npz("dataset/calib_S01_*.npz")

Note on channels: public 64-channel data must be reduced to the 8 electrodes you
actually own before it can be mixed with your recordings. `pick_channels()` does
that explicitly rather than silently.
"""

import glob
import os

import numpy as np

from ML import montage as montage_mod


# --------------------------------------------------------------------- helpers
def resample_to(X, fs_in, fs_out):
    """Resample (…, samples) along the last axis using FFT resampling."""
    if fs_in == fs_out:
        return X
    from scipy.signal import resample
    n_out = int(round(X.shape[-1] * fs_out / fs_in))
    return resample(X, n_out, axis=-1)


def pick_channels(X, indices):
    """Reduce a dense montage to the electrodes you physically have."""
    idx = list(indices)
    if X.shape[-2] <= max(idx):
        raise ValueError(
            f"cannot pick channels {idx} from data with {X.shape[-2]} channels")
    return X[..., idx, :]


def crop_or_pad(X, n_samples):
    """Force the time axis to exactly n_samples."""
    cur = X.shape[-1]
    if cur == n_samples:
        return X
    if cur > n_samples:
        return X[..., :n_samples]
    pad = [(0, 0)] * X.ndim
    pad[-1] = (0, n_samples - cur)
    return np.pad(X, pad, mode="edge")


def _resolve(path):
    hits = sorted(glob.glob(str(path)))
    if not hits:
        raise FileNotFoundError(f"No file matched {path!r}")
    return hits[-1]


# ------------------------------------------------------- Tsinghua Benchmark
def load_freq_phase(path):
    """Read Freq_Phase.mat -> (freqs, phases) as plain lists.

    ALWAYS use this instead of hard-coding target indices. The Benchmark's 40
    targets are laid out column-major (8.0-15.8 Hz in 0.2 Hz steps) and it is very
    easy to guess the wrong index -- a wrong map trains on mislabelled data and the
    resulting accuracy number is meaningless.
    """
    import scipy.io
    mat = scipy.io.loadmat(_resolve(path))
    if "freqs" not in mat:
        raise ValueError(f"{path} has no 'freqs' key -- is this really Freq_Phase.mat?")
    freqs = np.asarray(mat["freqs"]).ravel().astype(float).tolist()
    phases = (np.asarray(mat["phases"]).ravel().astype(float).tolist()
              if "phases" in mat else [0.0] * len(freqs))
    return freqs, phases


def build_freq_map(freq_phase_path, target_freqs, tol=0.05):
    """Map YOUR stimulus frequencies onto Benchmark target indices.

    :return: (freq_map, missing) where freq_map is {dataset_index: class_index}
             and `missing` lists the frequencies absent from the dataset.

    Frequencies the Benchmark does not contain (e.g. 20 Hz) are reported, not
    silently dropped -- you cannot pre-train a class that has no public data.
    """
    freqs, _ = load_freq_phase(freq_phase_path)
    freq_map, missing = {}, []
    for cls, want in enumerate(target_freqs):
        best, err = None, 1e9
        for i, f in enumerate(freqs):
            d = abs(f - want)
            if d < err:
                best, err = i, d
        if best is not None and err <= tol:
            freq_map[best] = cls
        else:
            missing.append(float(want))
    return freq_map, missing



def load_benchmark_mat(path, freq_map=None, fs=250, window=750, cue_offset=160,
                       channels=None):
    """Load one subject from the Tsinghua SSVEP Benchmark dataset.

    :param path: path to S<n>.mat (globs allowed)
    :param freq_map: {dataset_target_index: your_class_index}. Verify these against
                     Freq_Phase.mat -- a wrong map silently trains on the wrong labels.
    :param cue_offset: samples to skip (0.5 s dataset cue + ~140 ms visual latency)
    :param channels: electrode indices to keep; None keeps all 64
    :return: (X, y, meta) with X shaped (trials, channels, window)
    """
    import scipy.io
    path = _resolve(path)
    mat = scipy.io.loadmat(path)
    if "data" not in mat:
        raise ValueError(f"{os.path.basename(path)} has no 'data' key "
                         f"(Freq_Phase.mat is metadata, not trials)")
    arr = mat["data"]                       # (64, 1500, 40, 6)
    if arr.ndim != 4:
        raise ValueError(f"expected 4-D benchmark array, got {arr.shape}")

    if freq_map is None:
        raise ValueError(
            "freq_map is required: {dataset_target_index: class_index}. "
            "Check Freq_Phase.mat to confirm which index is which frequency.")

    X, y = [], []
    for tgt_idx, cls in freq_map.items():
        for block in range(arr.shape[3]):
            seg = arr[:, cue_offset:cue_offset + window, tgt_idx, block]
            if seg.shape[1] < window:
                continue
            X.append(seg)
            y.append(cls)
    if not X:
        raise ValueError("no trials extracted -- check freq_map and cue_offset")

    X = np.asarray(X, dtype=np.float64)
    if channels is not None:
        X = pick_channels(X, channels)
    return X, np.asarray(y), {"source": "tsinghua_benchmark", "fs": fs,
                              "file": os.path.basename(path),
                              "n_channels": X.shape[1]}


def load_benchmark_dir(directory, freq_map=None, **kw):
    """Load and concatenate every S*.mat in a directory (skips Freq_Phase.mat)."""
    files = [f for f in sorted(glob.glob(os.path.join(str(directory), "*.mat")))
             if "freq_phase" not in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(f"no S*.mat files in {directory}")
    Xs, ys, subs = [], [], []
    for i, f in enumerate(files):
        try:
            X, y, _ = load_benchmark_mat(f, freq_map=freq_map, **kw)
        except Exception:
            continue
        Xs.append(X)
        ys.append(y)
        subs.append(np.full(len(y), i))
    if not Xs:
        raise ValueError("no usable subjects loaded")
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(subs),
            {"source": "tsinghua_benchmark", "n_subjects": len(Xs)})


# --------------------------------------------------------- PhysioNet EEGMMIDB
def load_physionet_edf(paths, target_channels=None, fs_out=250.0, window=750,
                       tmin=0.5, tmax=3.5, bandpass=(8.0, 30.0)):
    """Load PhysioNet Motor Imagery runs (requires `mne`).

    :param paths: list of .edf paths, or a glob
    :return: (X, y, subjects, meta); y: 0 = left fist (T1), 1 = right fist (T2)
    """
    try:
        import mne
    except ImportError as exc:               # pragma: no cover
        raise ImportError(
            "load_physionet_edf needs mne: pip install mne") from exc
    mne.set_log_level("ERROR")

    if isinstance(paths, str):
        paths = sorted(glob.glob(paths))
    if not paths:
        raise FileNotFoundError("no EDF files given")
    target_channels = target_channels or ['FC3', 'FC4', 'C3', 'CZ',
                                          'C4', 'CP3', 'CP4', 'PZ']

    Xs, ys, subs = [], [], []
    for si, fp in enumerate(paths):
        try:
            raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
            if raw.info["sfreq"] != fs_out:
                raw.resample(fs_out, verbose=False)
            if bandpass:
                raw.filter(*bandpass, fir_design="firwin",
                           skip_by_annotation="edge", verbose=False)
            raw.rename_channels(lambda c: c.strip(".").upper())
            picks = [c for c in target_channels if c in raw.ch_names]
            if len(picks) != len(target_channels):
                continue
            raw.pick(picks)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            ev = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
            if not ev:
                continue
            ep = mne.Epochs(raw, events, event_id=ev, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True, verbose=False)
            data = ep.get_data()
            if data.shape[-1] < window:
                continue
            data = data[:, :, :window]
            t1 = ev.get("T1")
            labels = np.where(ep.events[:, -1] == t1, 0, 1)
            Xs.append(data.astype(np.float64))
            ys.append(labels)
            subs.append(np.full(len(labels), si))
        except Exception:
            continue

    if not Xs:
        raise ValueError("no usable epochs extracted from the EDF files")
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(subs),
            {"source": "physionet_eegmmidb", "fs": fs_out,
             "channels": target_channels, "units": "volts"})


# ------------------------------------------------------------- own recordings
def load_calibration_npz(path, include=("target_0", "target_1")):
    """Load YOUR OWN Step 3 calibration recording.

    :return: (X, y, meta). meta carries `groups` (recording blocks -- use these for
             leakage-safe CV), `idle` windows, and `labels`.
    """
    path = _resolve(path)
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    labels = d["labels"].astype(str)

    mask = np.isin(labels, list(include))
    y = np.array([list(include).index(l) if l in include else -1
                  for l in labels[mask]])

    groups = np.zeros(len(labels), dtype=int)
    g = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            g += 1
        groups[i] = g

    return X[mask], y, {
        "source": "own_calibration", "file": os.path.basename(path),
        "fs": int(d["fs"]), "montage": str(d["montage"]),
        "target_freqs": [float(f) for f in d["target_freqs"]],
        "groups": groups[mask],
        "labels": labels[mask],
        "idle": X[np.isin(labels, ["idle", "idle_distracted"])],
        "artifact": X[labels == "artifact"],
    }


def combine(*datasets, montage_name="cyton8_ssvep"):
    """Concatenate datasets that have been reduced to the same channel count.

    Use to pre-train on public data and fine-tune on your own. Raises rather than
    broadcasting if the shapes disagree -- silent mismatches here are how a model
    ends up trained on the wrong electrodes.
    """
    Xs, ys, srcs = [], [], []
    ref = None
    for i, (X, y) in enumerate(datasets):
        X = np.asarray(X)
        if ref is None:
            ref = X.shape[1:]
        elif X.shape[1:] != ref:
            raise ValueError(
                f"dataset {i} has shape {X.shape[1:]} but expected {ref}. "
                f"Use pick_channels()/resample_to()/crop_or_pad() first.")
        Xs.append(X)
        ys.append(np.asarray(y))
        srcs.append(np.full(len(y), i))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(srcs)
