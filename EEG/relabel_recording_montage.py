"""
Author: Anson
Created on: 12/11/2026
Purpose: safely correct montage metadata on existing NPZ recordings.

This never changes EEG samples, trial labels or class indices. By default it
writes a new file. Use --in-place only after inspecting the preview output; an
`.original.npz` backup is retained beside every replaced file.

Example for the reported lab channel order:
    python EEG/relabel_recording_montage.py \
      --input "dataset/motor_intent/calib_brian_*.npz" \
      --montage cyton8_fp_occipital
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.montage import MONTAGES, get


def relabel(path, montage_name, in_place=False):
    path = Path(path)
    montage = get(montage_name)
    with np.load(path, allow_pickle=True) as data:
        payload = {name: data[name] for name in data.files}
    X = np.asarray(payload.get("X"))
    if X.ndim != 3:
        raise ValueError(f"{path}: expected X=(trials,channels,samples), got {X.shape}")
    if X.shape[1] != montage["n_channels"]:
        raise ValueError(
            f"{path}: data has {X.shape[1]} channels but {montage_name} expects "
            f"{montage['n_channels']}")

    old_montage = str(np.asarray(payload.get("montage", "unknown")).item())
    old_labels = (np.asarray(payload.get("channel_labels", [])).astype(str).tolist())
    payload["montage"] = np.asarray(montage_name)
    payload["channel_labels"] = np.asarray(montage["labels"])

    if in_place:
        backup = path.with_name(path.stem + ".original.npz")
        if backup.exists():
            raise FileExistsError(f"backup already exists: {backup}")
        path.replace(backup)
        output = path
    else:
        backup = None
        output = path.parent / "montage_fixed" / path.name
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
    np.savez_compressed(output, **payload)

    print(f"\n{path}")
    print(f"  old montage : {old_montage}")
    print(f"  old labels  : {old_labels}")
    print(f"  new montage : {montage_name}")
    print(f"  new labels  : {montage['labels']}")
    print(f"  EEG shape   : {X.shape} (samples unchanged)")
    print(f"  output      : {output}")
    if backup:
        print(f"  backup      : {backup}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Correct montage/channel-label metadata without changing EEG")
    parser.add_argument("--input", nargs="+", required=True,
                        help="NPZ path(s) or glob pattern(s)")
    parser.add_argument("--montage", required=True, choices=sorted(MONTAGES))
    parser.add_argument("--in-place", action="store_true",
                        help="replace each input but retain .original.npz backup")
    args = parser.parse_args()

    files = []
    for pattern in args.input:
        matches = sorted(glob.glob(pattern))
        if Path(pattern).is_file() and pattern not in matches:
            matches.append(pattern)
        files.extend(matches)
    unique = []
    seen = set()
    for value in files:
        resolved = str(Path(value).resolve())
        if resolved not in seen:
            unique.append(Path(value)); seen.add(resolved)
    if not unique:
        print("No NPZ files matched.")
        return 1
    for path in unique:
        relabel(path, args.montage, in_place=args.in_place)
    print(f"\nUpdated {len(unique)} recording(s). Verify the new files before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
