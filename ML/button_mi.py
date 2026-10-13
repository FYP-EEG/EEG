"""
Author: Anson
Created on: 13/10/2026
Purpose: experimental direct-button motor-imagery model and label contract.

This module is deliberately separate from ML/local_model.py. The established
motor-imagery path predicts LEFT or RIGHT and remains unchanged. A button-MI
model instead predicts the name of one button directly:

    imagine reaching to and pressing ROCK -> "rock"

This is an experimental multi-class movement-imagery task, not semantic thought
reading. It has the best chance when each button occupies a different screen
position and the participant imagines the complete reach-and-press movement.
A model is specific to one user, application and button layout; there is no
public-data starter model for arbitrary buttons.
"""

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "profiles"
NO_ACTION = -1
BUTTON_LABEL_PREFIX = "button::"


def _slug(value):
    """Filesystem-safe user/application key."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower())
    value = value.strip("._-")
    if not value:
        raise ValueError("user/app name must contain a letter or number")
    return value


def button_profile_dir(user, app, root=None):
    """Directory isolated from the existing LEFT/RIGHT MI profile."""
    base = Path(root) if root is not None else PROFILES
    return base / _slug(user) / "button_mi" / _slug(app)


def make_button_label(index, name):
    """Stable label stored in calibration .npz files.

    The zero-padded screen-order index makes class ordering deterministic while
    the final field carries the exact Button.name used for live dispatch.
    """
    name = str(name).strip()
    if not name:
        raise ValueError("button name cannot be empty")
    return f"{BUTTON_LABEL_PREFIX}{int(index):03d}::{name}"


def parse_button_label(label):
    """Return ``(screen_index, button_name)`` from a stored label."""
    text = str(label)
    if not text.startswith(BUTTON_LABEL_PREFIX):
        raise ValueError(f"not a button-MI label: {text!r}")
    parts = text.split("::", 2)
    if len(parts) != 3 or not parts[2]:
        raise ValueError(f"invalid button-MI label: {text!r}")
    try:
        index = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid button index in {text!r}") from exc
    return index, parts[2]


class ButtonMIModel:
    """Serve an app-specific, multi-class direct-button model locally."""

    def __init__(self, model, class_map, sample_freq=250, montage=None,
                 confidence_threshold=0.55, user=None, app=None, meta=None):
        self.model = model
        self.class_map = {int(k): str(v) for k, v in dict(class_map).items()}
        self.sample_freq = int(sample_freq)
        self.montage = montage
        self.confidence_threshold = float(confidence_threshold)
        self.user = user
        self.app = app
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, user, app, root=None, confidence_threshold=None):
        """Load ``button_model.joblib`` for exactly this user and app.

        No fallback to the LEFT/RIGHT starter model is allowed: that model has
        different classes and cannot recognise application-specific buttons.
        """
        import joblib

        directory = button_profile_dir(user, app, root=root)
        path = directory / "button_model.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"no direct-button model for user={user!r}, app={app!r}: {path}\n"
                f"record with `python EEG/data_record.py --file {app} "
                f"--button-mi --subject {user} ...`, then train with "
                f"`python ML/train_button_mi.py {user} --app {app}`")
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get("kind") != "ButtonMILocalModel":
            raise ValueError(f"{path} is not a direct-button MI model")

        threshold = (bundle.get("confidence_threshold", 0.55)
                     if confidence_threshold is None
                     else confidence_threshold)
        return cls(
            model=bundle["model"],
            class_map=bundle["class_map"],
            sample_freq=bundle.get("sample_freq", 250),
            montage=bundle.get("montage"),
            confidence_threshold=threshold,
            user=str(user), app=str(app),
            meta={k: v for k, v in bundle.items() if k != "model"},
        )

    def _prepare(self, window):
        x = np.asarray(window, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (channels, samples), got {x.shape}")
        return x[np.newaxis, ...]

    def predict_proba(self, window):
        """Return ``(class_index, confidence)`` or ``(NO_ACTION, confidence)``."""
        x = self._prepare(window)
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(x))[0]
            pos = int(np.argmax(probs))
            classes = getattr(self.model, "classes_", np.arange(len(probs)))
            label = int(classes[pos])
            confidence = float(probs[pos])
        else:
            label = int(self.model.predict(x)[0])
            confidence = 1.0
        if confidence < self.confidence_threshold:
            return NO_ACTION, confidence
        return label, confidence

    def predict_command(self, window):
        """Return ``(button_name | None, confidence)``."""
        label, confidence = self.predict_proba(window)
        if label == NO_ACTION:
            return None, confidence
        if label not in self.class_map:
            raise ValueError(f"model predicted unknown button class {label}")
        return self.class_map[label], confidence

    def describe(self):
        return {
            "kind": "ButtonMILocalModel",
            "user": self.user,
            "app": self.app,
            "pipeline": self.meta.get("pipeline"),
            "buttons": [self.class_map[k] for k in sorted(self.class_map)],
            "cv_accuracy": self.meta.get("cv_accuracy"),
            "cv_reliable": self.meta.get("cv_reliable"),
            "chance_accuracy": self.meta.get("chance_accuracy"),
            "n_trials": self.meta.get("n_trials"),
            "n_sessions": self.meta.get("n_sessions"),
            "sample_freq": self.sample_freq,
            "montage": self.montage,
            "confidence_threshold": self.confidence_threshold,
            "trained_at": self.meta.get("trained_at"),
        }

    def __repr__(self):
        d = self.describe()
        acc = d["cv_accuracy"]
        acc = f"{acc:.1%}" if isinstance(acc, (float, np.floating)) else "?"
        return (f"<ButtonMIModel app={self.app!r} buttons={len(self.class_map)} "
                f"pipeline={d['pipeline']} cv={acc} "
                f"thr={self.confidence_threshold:.2f}>")
