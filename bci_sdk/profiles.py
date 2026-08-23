"""
Author: Anson
Created on: 23/8/2026
Purpose: Per-user profile storage.

Calibration is PER USER, not per install: the confidence threshold depends on an
individual's alpha amplitude, skull thickness, and exactly where the electrodes sat
today. A threshold fitted to one person will make the speller either unresponsive or
trigger-happy for the next.

Layout on disk (mirrors the Drive layout so recordings can be synced verbatim):

    profiles/
      anson/
        profile.json            <- fitted threshold + metadata
        calib_20260823_1042.npz <- the recording it came from
      brian/
        profile.json
        ...

`UserProfile` is deliberately plain JSON + npz so a profile can be emailed, committed,
or dropped into Google Drive without any custom tooling.
"""

import glob
import json
import os
import time
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "profiles"


def _slug(name):
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name).strip())
    return keep.lower() or "user"


class UserProfile:
    """One person's calibration state."""

    def __init__(self, name, root=None):
        self.name = str(name).strip()
        self.slug = _slug(name)
        self.root = Path(root) if root else DEFAULT_ROOT
        self.dir = self.root / self.slug
        self.path = self.dir / "profile.json"
        self.data = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                self.data = {}

    # ---------------------------------------------------------------- state
    @property
    def exists(self):
        return self.path.exists()

    @property
    def is_calibrated(self):
        return bool(self.data.get("confidence_threshold"))

    @property
    def threshold(self):
        return self.data.get("confidence_threshold")

    @property
    def target_freqs(self):
        return tuple(self.data.get("target_freqs", (15.0, 20.0)))

    @property
    def age_days(self):
        ts = self.data.get("calibrated_at")
        return None if not ts else (time.time() - ts) / 86400.0

    def is_stale(self, max_age_days=7.0):
        """Electrode placement drifts between sessions; recalibrate periodically."""
        a = self.age_days
        return a is None or a > max_age_days

    # ------------------------------------------------------------------ io
    def save(self, **fields):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data.update(fields)
        self.data.setdefault("name", self.name)
        self.data["calibrated_at"] = self.data.get("calibrated_at") or time.time()
        self.path.write_text(json.dumps(self.data, indent=2))
        return self.path

    def update_from_analysis(self, profile_dict, recording_path=None):
        """Store the output of ML.calibration.analyse()."""
        self.data.update({
            "confidence_threshold": profile_dict.get("confidence_threshold"),
            "target_freqs": profile_dict.get("target_freqs", list(self.target_freqs)),
            "montage": profile_dict.get("montage", "cyton8_ssvep"),
            "sample_freq": profile_dict.get("sample_freq", 250),
            "debounce_window": profile_dict.get("debounce_window", 3),
            "metrics": profile_dict.get("metrics", {}),
            "idle_minutes": profile_dict.get("idle_minutes"),
            "calibrated_at": time.time(),
        })
        if recording_path:
            self.data["recording"] = str(recording_path)
        return self.save()

    def recording_path(self, stamp=None):
        """Where this user's next recording should be written."""
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
        return self.dir / f"calib_{self.slug}_{stamp}.npz"

    def latest_recording(self):
        hits = sorted(glob.glob(str(self.dir / "calib_*.npz")))
        return hits[-1] if hits else None

    def as_session_kwargs(self):
        """Kwargs to hand straight to BCISession."""
        if not self.is_calibrated:
            return {}
        return {"profile": str(self.path)}

    def __repr__(self):
        state = "calibrated" if self.is_calibrated else "uncalibrated"
        return f"<UserProfile {self.name!r} {state}>"


def list_users(root=None):
    """All known users, most recently calibrated first."""
    root = Path(root) if root else DEFAULT_ROOT
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "profile.json").exists():
            try:
                name = json.loads((d / "profile.json").read_text()).get("name", d.name)
            except Exception:
                name = d.name
            out.append(UserProfile(name, root=root))
    out.sort(key=lambda p: p.data.get("calibrated_at", 0), reverse=True)
    return out
