"""
Author: Anson
Created on: 19/10/2026
Purpose: reusable motor-intent vocabulary, mapping validation and local model.

The classifier predicts a physical motor intention (for example right_index or
both_feet). Applications map that stable intention to Rock, jump, reload, or any
other callback. Models are stored by user + task set, so the same calibrated
intentions can be reused by several applications without retraining.

This module is additive. The established LEFT/RIGHT model and the experimental
button-labelled model keep their existing files and behaviour.
"""

import hashlib
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "profiles"
NO_ACTION = -1
INTENT_LABEL_PREFIX = "intent::"
IDLE = "idle"


class MotorIntent:
    LEFT_THUMB = "left_thumb"
    LEFT_INDEX = "left_index"
    LEFT_MIDDLE = "left_middle"
    LEFT_RING = "left_ring"
    LEFT_PINKY = "left_pinky"
    RIGHT_THUMB = "right_thumb"
    RIGHT_INDEX = "right_index"
    RIGHT_MIDDLE = "right_middle"
    RIGHT_RING = "right_ring"
    RIGHT_PINKY = "right_pinky"

    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    BOTH_HANDS = "both_hands"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    BOTH_FEET = "both_feet"
    IDLE = IDLE


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_INTENTS = tuple(
    f"{side}_{finger}" for side in ("left", "right") for finger in FINGER_NAMES)
BODY_INTENTS = ("left_hand", "right_hand", "both_hands",
                "left_foot", "right_foot", "both_feet")
VALID_INTENTS = FINGER_INTENTS + BODY_INTENTS + (IDLE,)
_CANONICAL_ORDER = {name: i for i, name in enumerate(VALID_INTENTS)}

_ALIASES = {
    "left_little": "left_pinky", "right_little": "right_pinky",
    "left_little_finger": "left_pinky", "right_little_finger": "right_pinky",
    "left_index_finger": "left_index", "right_index_finger": "right_index",
    "left_middle_finger": "left_middle", "right_middle_finger": "right_middle",
    "left_ring_finger": "left_ring", "right_ring_finger": "right_ring",
    "left_thumb_finger": "left_thumb", "right_thumb_finger": "right_thumb",
    "feet": "both_feet", "both_foot": "both_feet", "rest": IDLE,
}


def _slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower())
    value = value.strip("._-")
    if not value:
        raise ValueError("name must contain a letter or number")
    return value


def normalize_intent(value, allow_idle=True):
    """Normalize an intent string and reject unsupported/invented classes."""
    name = str(getattr(value, "value", value)).strip().lower()
    name = re.sub(r"[\s-]+", "_", name)
    name = _ALIASES.get(name, name)
    allowed = VALID_INTENTS if allow_idle else VALID_INTENTS[:-1]
    if name not in allowed:
        raise ValueError(
            f"unknown motor intent {value!r}; expected one of: "
            f"{', '.join(allowed)}")
    return name


def canonical_intents(intents, allow_idle=False):
    """Unique canonical task set, independent of developer mapping order."""
    clean = [normalize_intent(v, allow_idle=allow_idle) for v in intents]
    if not allow_idle and IDLE in clean:
        raise ValueError("idle is an internal no-action class, not an app action")
    if len(clean) != len(set(clean)):
        raise ValueError("each application action needs a unique motor intention")
    if not clean:
        raise ValueError("at least one motor intention is required")
    return tuple(sorted(clean, key=lambda v: _CANONICAL_ORDER[v]))


def make_intent_label(intent):
    return INTENT_LABEL_PREFIX + normalize_intent(intent, allow_idle=True)


def parse_intent_label(label):
    text = str(label)
    if not text.startswith(INTENT_LABEL_PREFIX):
        raise ValueError(f"not a motor-intent label: {text!r}")
    return normalize_intent(text[len(INTENT_LABEL_PREFIX):], allow_idle=True)


def parse_intent_map(value):
    """Parse ``action=intent,...`` or validate an existing mapping."""
    if isinstance(value, MotorIntentMap):
        return dict(value.action_to_intent)
    if isinstance(value, dict):
        items = value.items()
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("intent map is empty")
        pairs = []
        for part in text.split(","):
            if "=" not in part:
                raise ValueError(
                    f"invalid intent mapping {part!r}; use action=intent")
            action, intent = part.split("=", 1)
            pairs.append((action, intent))
        items = pairs

    result = {}
    for raw_action, raw_intent in items:
        action = str(raw_action).strip()
        if not action:
            raise ValueError("application action name cannot be empty")
        if action in result:
            raise ValueError(f"duplicate application action {action!r}")
        result[action] = normalize_intent(raw_intent, allow_idle=False)
    canonical_intents(result.values())  # validates nonempty + unique
    return result


def task_set_key(intents):
    """Stable, reusable directory key for an unordered set of intentions."""
    values = canonical_intents(intents)
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:8]
    if len(values) <= 3:
        readable = "-".join(values)
    elif set(values) == set(FINGER_INTENTS):
        readable = "all-10-fingers"
    else:
        readable = f"{len(values)}-intents"
    return f"{readable}-{digest}"


def intent_profile_dir(user, intents, root=None):
    base = Path(root) if root is not None else PROFILES
    return base / _slug(user) / "motor_intent" / task_set_key(intents)


def adjacent_finger_pairs(intents):
    values = set(canonical_intents(intents))
    pairs = []
    for side in ("left", "right"):
        for a, b in zip(FINGER_NAMES, FINGER_NAMES[1:]):
            ia, ib = f"{side}_{a}", f"{side}_{b}"
            if ia in values and ib in values:
                pairs.append((ia, ib))
    return pairs


def assess_intent_set(intents):
    """Neurophysiology-aware warnings; API scale is not an accuracy claim."""
    values = canonical_intents(intents)
    fingers = [v for v in values if v in FINGER_INTENTS]
    adjacent = adjacent_finger_pairs(values)
    n = len(values)
    if n == 1:
        level = "BASIC"
    elif n == 2:
        level = "PILOT"
    elif n == 3:
        level = "EXPERIMENTAL"
    elif n <= 5:
        level = "ADVANCED"
    else:
        level = "RESEARCH_ONLY"

    warnings = []
    if n == 1:
        warnings.append(
            "one-intent control requires recorded idle trials so the model can "
            "distinguish the command from no action")
    if adjacent:
        warnings.append(
            "adjacent same-hand fingers have strongly overlapping cortical "
            "representations: " + ", ".join(f"{a}/{b}" for a, b in adjacent))
    if len(fingers) >= 4:
        warnings.append(
            f"{len(fingers)} finger classes is an advanced fine-MI task; "
            "accuracy normally falls as similar classes are added")
    if len(fingers) >= 6:
        warnings.append(
            "6-10 finger control is research-only for low-density scalp EEG; "
            "software support does not guarantee neurological separability")
    for side in ("left", "right"):
        same = [v for v in fingers if v.startswith(side + "_")]
        if len(same) >= 3:
            warnings.append(
                f"{len(same)} classes use the {side} hand; consider spreading "
                "commands across both hands when the application permits")
    return {"intents": values, "count": n, "finger_count": len(fingers),
            "difficulty": level, "adjacent_pairs": adjacent,
            "warnings": warnings, "task_set": task_set_key(values)}


def intent_display_name(intent):
    name = normalize_intent(intent, allow_idle=True)
    if name == IDLE:
        return "REST / NO MOVEMENT"
    return name.replace("_", " ").upper()


def intent_instruction(intent, action=None):
    name = normalize_intent(intent, allow_idle=True)
    if name == IDLE:
        return "REST. Keep still and do not imagine any movement."
    target = f" to activate '{action}'" if action else ""
    if name in FINGER_INTENTS:
        side, finger = name.split("_", 1)
        return (f"Kinesthetically imagine repeatedly PRESSING with your "
                f"{side.upper()} {finger.upper()} finger{target}. Do not move.")
    if name == "both_feet":
        return "Kinesthetically imagine JUMPING with BOTH FEET. Do not move."
    if name.endswith("_hand"):
        side = name.split("_", 1)[0].upper()
        return f"Kinesthetically imagine moving your {side} HAND{target}. Do not move."
    if name == "both_hands":
        return "Kinesthetically imagine moving BOTH HANDS. Do not move."
    if name.endswith("_foot"):
        side = name.split("_", 1)[0].upper()
        return f"Kinesthetically imagine moving your {side} FOOT{target}. Do not move."
    return f"Imagine {name.replace('_', ' ')}{target}. Do not move."


class MotorIntentMap:
    """Developer-facing action -> reusable motor intention mapping."""

    def __init__(self, mapping):
        self.action_to_intent = parse_intent_map(mapping)
        self.intent_to_action = {v: k for k, v in self.action_to_intent.items()}
        self.intents = canonical_intents(self.action_to_intent.values())
        self.assessment = assess_intent_set(self.intents)

    @property
    def warnings(self):
        return list(self.assessment["warnings"])

    @property
    def task_set(self):
        return self.assessment["task_set"]

    def action_for(self, intent):
        return self.intent_to_action.get(normalize_intent(intent, allow_idle=False))

    def report(self):
        return {"mapping": dict(self.action_to_intent), **self.assessment}

    def __len__(self):
        return len(self.action_to_intent)

    def __repr__(self):
        return (f"<MotorIntentMap {len(self)} actions "
                f"difficulty={self.assessment['difficulty']}>")


class MotorIntentModel:
    """Serve a reusable, locally trained motor-intent task-set model."""

    def __init__(self, model, class_map, intents, sample_freq=250, montage=None,
                 confidence_threshold=0.55, user=None, meta=None):
        self.model = model
        self.class_map = {int(k): str(v) for k, v in dict(class_map).items()}
        self.intents = canonical_intents(intents)
        self.sample_freq = int(sample_freq)
        self.montage = montage
        self.confidence_threshold = float(confidence_threshold)
        self.user = user
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, user, intents, root=None, confidence_threshold=None):
        import joblib

        values = canonical_intents(intents)
        directory = intent_profile_dir(user, values, root=root)
        path = directory / "intent_model.joblib"
        if not path.exists():
            joined = ",".join(values)
            raise FileNotFoundError(
                f"no reusable motor-intent model for user={user!r}, "
                f"intents={values!r}: {path}\n"
                f"record with `python EEG/data_record.py --file <app> "
                f"--intent-mi --subject {user} ...`, then train with "
                f"`python ML/train_intent_mi.py {user} --intents {joined}`")
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get("kind") != "MotorIntentModel":
            raise ValueError(f"{path} is not a reusable motor-intent model")
        stored = canonical_intents(bundle.get("intents", ()))
        if stored != values:
            raise ValueError(f"model task set {stored!r} does not match {values!r}")
        threshold = (bundle.get("confidence_threshold", 0.55)
                     if confidence_threshold is None else confidence_threshold)
        return cls(bundle["model"], bundle["class_map"], stored,
                   sample_freq=bundle.get("sample_freq", 250),
                   montage=bundle.get("montage"),
                   confidence_threshold=threshold, user=str(user),
                   meta={k: v for k, v in bundle.items() if k != "model"})

    def _prepare(self, window):
        x = np.asarray(window, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (channels, samples), got {x.shape}")
        return x[np.newaxis, ...]

    def predict_proba(self, window):
        x = self._prepare(window)
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(x))[0]
            pos = int(np.argmax(probs))
            classes = getattr(self.model, "classes_", np.arange(len(probs)))
            label, confidence = int(classes[pos]), float(probs[pos])
        else:
            label, confidence = int(self.model.predict(x)[0]), 1.0
        if confidence < self.confidence_threshold:
            return NO_ACTION, confidence
        return label, confidence

    def predict_command(self, window):
        label, confidence = self.predict_proba(window)
        if label == NO_ACTION:
            return None, confidence
        intent = self.class_map.get(label)
        if intent is None:
            raise ValueError(f"model predicted unknown motor-intent class {label}")
        if intent == IDLE:
            return None, confidence
        return intent, confidence

    def describe(self):
        return {
            "kind": "MotorIntentModel", "user": self.user,
            "pipeline": self.meta.get("pipeline"), "intents": list(self.intents),
            "classes": [self.class_map[k] for k in sorted(self.class_map)],
            "task_set": task_set_key(self.intents),
            "cv_accuracy": self.meta.get("cv_accuracy"),
            "cv_reliable": self.meta.get("cv_reliable"),
            "chance_accuracy": self.meta.get("chance_accuracy"),
            "n_trials": self.meta.get("n_trials"),
            "n_sessions": self.meta.get("n_sessions"),
            "sample_freq": self.sample_freq, "montage": self.montage,
            "confidence_threshold": self.confidence_threshold,
            "trained_at": self.meta.get("trained_at"),
        }

    def __repr__(self):
        d = self.describe()
        acc = d["cv_accuracy"]
        acc = f"{acc:.1%}" if isinstance(acc, (float, np.floating)) else "?"
        return (f"<MotorIntentModel intents={len(self.intents)} "
                f"pipeline={d['pipeline']} cv={acc} "
                f"thr={self.confidence_threshold:.2f}>")
