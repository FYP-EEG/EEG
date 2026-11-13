"""
Author: Anson
Created on: 19/10/2026
Purpose: gameplay prediction feedback logging and live monitor data export.

Controls used by the three pygame demos while GAME is active:
    Space       pause/resume gameplay and BCI dispatch
    Left click  mark the latest classified prediction correct
    Right click mark the latest classified prediction incorrect

The separate ``example/performance_monitor.py`` process reads the files written
here. No gameplay EEG is treated as training truth unless the user explicitly
marks it; this module only records feedback and never retrains automatically.
"""

import glob
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = ROOT / "runtime_logs"
DEFAULT_CHANNELS = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Pz")


def _slug(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown").strip())
    return text.strip("._-") or "unknown"


def _name(value):
    if value is None:
        return None
    return str(getattr(value, "name", value))


class PredictionPerformanceLogger:
    """Persist model outputs, manual correctness marks and the latest EEG."""

    def __init__(self, app, user=None, root=None, sample_freq=250,
                 channels=DEFAULT_CHANNELS, eeg_write_interval=.25):
        self.app = _slug(app)
        self.user = str(user or "anonymous")
        self.sample_freq = int(sample_freq)
        self.channels = list(channels)
        self.root = Path(root) if root else DEFAULT_LOG_ROOT
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = self.root / self.app / stamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.session_dir / "events.jsonl"
        self.status_path = self.session_dir / "status.json"
        self.eeg_path = self.session_dir / "latest_eeg.npy"
        self._events = self.events_path.open("a", encoding="utf-8", buffering=1)
        self.created = datetime.now().isoformat(timespec="seconds")
        self.started = time.monotonic()
        self.eeg_write_interval = float(eeg_write_interval)
        self._last_eeg_write = -1e18
        self._next_prediction_id = 1
        self._feedback_prediction_ids = set()
        self.latest_prediction = None
        self.last_feedback = None
        self.paused = False
        self.stats = {
            "windows": 0, "classified": 0, "dispatched": 0,
            "rejected": 0, "blink": 0, "feedback_total": 0,
            "feedback_correct": 0, "feedback_incorrect": 0,
        }
        self._write_pointer()
        self._write_status()

    def _atomic_json(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _write_pointer(self):
        payload = {"session_dir": str(self.session_dir), "app": self.app,
                   "user": self.user, "created": self.created}
        self._atomic_json(self.root / "latest_session.json", payload)
        self._atomic_json(self.root / self.app / "latest_session.json", payload)

    def _append(self, payload):
        self._events.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _status(self):
        total = self.stats["feedback_total"]
        return {
            "app": self.app, "user": self.user, "created": self.created,
            "session_dir": str(self.session_dir),
            "elapsed_s": round(time.monotonic() - self.started, 3),
            "sample_freq": self.sample_freq, "channels": self.channels,
            "paused": self.paused, "stats": dict(self.stats),
            "manual_accuracy": (self.stats["feedback_correct"] / total
                                if total else None),
            "latest_prediction": self.latest_prediction,
            "last_feedback": self.last_feedback,
            "latest_eeg": str(self.eeg_path),
            "updated": datetime.now().isoformat(timespec="milliseconds"),
        }

    def _write_status(self):
        self._atomic_json(self.status_path, self._status())

    def _write_eeg(self, window):
        now = time.monotonic()
        if now - self._last_eeg_write < self.eeg_write_interval:
            return
        array = np.asarray(window, dtype=np.float32)
        tmp = self.eeg_path.with_suffix(".npy.tmp")
        with tmp.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        tmp.replace(self.eeg_path)
        self._last_eeg_write = now

    def record(self, window, result):
        """Record one classified live window and update dashboard files."""
        if result is None:
            return None
        self.stats["windows"] += 1
        state = str(result.get("state", "UNKNOWN"))
        command = result.get("command")
        if command is not None:
            self.stats["classified"] += 1
        if state == "DISPATCHED":
            self.stats["dispatched"] += 1
        elif state == "REJECTED":
            self.stats["rejected"] += 1
        elif state == "BLINK":
            self.stats["blink"] += 1

        prediction_id = None
        if command is not None:
            prediction_id = self._next_prediction_id
            self._next_prediction_id += 1
            self.latest_prediction = {
                "id": prediction_id, "command": _name(command),
                "selected": _name(result.get("selected")),
                "confidence": float(result.get("confidence", 0.0)),
                "state": state,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            }

        event = {
            "kind": "prediction", "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": round(time.monotonic() - self.started, 6),
            "state": state, "command": _name(command),
            "selected": _name(result.get("selected")),
            "confidence": float(result.get("confidence", 0.0)),
            "candidates": [_name(value)
                           for value in result.get("candidates", [])],
        }
        self._append(event)
        self._write_eeg(window)
        self._write_status()
        return event

    def mark_feedback(self, correct):
        """Attach one explicit human mark to the latest classified output."""
        prediction = self.latest_prediction
        if prediction is None:
            return False
        prediction_id = int(prediction["id"])
        if prediction_id in self._feedback_prediction_ids:
            return False
        self._feedback_prediction_ids.add(prediction_id)
        correct = bool(correct)
        self.stats["feedback_total"] += 1
        key = "feedback_correct" if correct else "feedback_incorrect"
        self.stats[key] += 1
        self.last_feedback = {
            "prediction_id": prediction_id, "correct": correct,
            "command": prediction["command"],
            "selected": prediction.get("selected"),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        }
        self._append({"kind": "feedback", **self.last_feedback})
        self._write_status()
        return True

    def set_paused(self, paused):
        self.paused = bool(paused)
        self._append({
            "kind": "pause", "paused": self.paused,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        })
        self._write_status()

    def close(self):
        try:
            self._write_status()
            self._events.close()
        except Exception:
            pass


class TrainingDataPerformance:
    """Inspect saved calibration epochs, quality screening and predictions.

    ``recordings`` accepts paths, directories or glob patterns. An optional
    saved ``intent_model.joblib`` adds intended-versus-predicted rows. Those
    predictions are explicitly training-data previews, not held-out evidence.
    """

    def __init__(self, recordings, model_path=None, screening_kw=None):
        patterns = ([recordings] if isinstance(recordings, (str, Path))
                    else list(recordings))
        files = []
        for value in patterns:
            path = Path(value)
            if path.is_dir():
                matches = sorted(path.glob("calib_*.npz"))
            else:
                matches = [Path(item) for item in sorted(glob.glob(str(value)))]
                if path.is_file() and path not in matches:
                    matches.append(path)
            files.extend(matches)
        # Stable deduplication is important when a directory and glob overlap.
        self.files = []
        seen = set()
        for path in files:
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                self.files.append(Path(path))
                seen.add(resolved)
        if not self.files:
            raise FileNotFoundError(
                "no training NPZ files matched: "
                + ", ".join(str(value) for value in patterns))

        Xs, labels_all, session_ids, subjects = [], [], [], []
        fs_values, montages, channel_label_sets = set(), set(), set()
        for session_id, path in enumerate(self.files):
            with np.load(path, allow_pickle=True) as data:
                X = np.asarray(data["X"], dtype=np.float64)
                labels = np.asarray(data["labels"]).astype(str)
                if X.ndim != 3 or X.shape[0] != len(labels):
                    raise ValueError(
                        f"{path} has inconsistent X/labels shapes: "
                        f"{X.shape} and {labels.shape}")
                Xs.append(X)
                labels_all.extend(labels.tolist())
                session_ids.extend([session_id] * len(X))
                fs_values.add(int(np.asarray(data["fs"]).item()))
                montages.add(str(np.asarray(data["montage"]).item()))
                if "channel_labels" in data:
                    channel_label_sets.add(tuple(
                        np.asarray(data["channel_labels"]).astype(str).tolist()))
                subject = (str(np.asarray(data["subject"]).item())
                           if "subject" in data else "unknown")
                subjects.extend([subject] * len(X))
        if len(fs_values) != 1:
            raise ValueError(f"training files use different sample rates: {fs_values}")
        if len(montages) != 1:
            raise ValueError(f"training files use different montages: {montages}")

        self.X = np.concatenate(Xs)
        self.labels = np.asarray(labels_all, dtype=str)
        self.sessions = np.asarray(session_ids, dtype=int)
        self.subjects = np.asarray(subjects, dtype=str)
        self.fs = fs_values.pop()
        self.montage = montages.pop()
        if len(channel_label_sets) > 1:
            raise ValueError("training files use different channel-label orders")
        if channel_label_sets:
            self.channel_labels = list(channel_label_sets.pop())
        else:
            try:
                from ML.montage import get
                self.channel_labels = list(get(self.montage).get("labels", []))
            except Exception:
                self.channel_labels = [f"CH{index + 1}"
                                       for index in range(self.X.shape[1])]
        if len(self.channel_labels) != self.X.shape[1]:
            self.channel_labels = [f"CH{index + 1}"
                                   for index in range(self.X.shape[1])]
        self.class_names = sorted(set(self.labels.tolist()))
        self.class_counts = {
            name: int((self.labels == name).sum()) for name in self.class_names}

        from ML.quality import screen_trials
        self.screening_kw = {
            "amp_uv": 400.0, "jump_uv": 75.0,
            "flat_uv": .05, "z_max": 4.0,
            **dict(screening_kw or {}),
        }
        self.screening = screen_trials(
            self.X, self.labels, fs=self.fs, verbose=False,
            **self.screening_kw)
        self.keep = np.asarray(self.screening["keep"], dtype=bool)
        self.drop_reasons = []
        for index in range(len(self.X)):
            self.drop_reasons.append([
                name for name, mask in self.screening["reasons"].items()
                if bool(mask[index])])
        self.retained_counts = {
            name: int(((self.labels == name) & self.keep).sum())
            for name in self.class_names}

        # Per-channel attribution for troubleshooting all-trial rejection. The
        # generic screener rejects a trial when ANY channel fails; these values
        # reveal whether one dead/noisy electrode is responsible for all trials
        # or the complete recording has a scaling/reference problem.
        centred = self.X - self.X.mean(axis=2, keepdims=True)
        peak_by_trial_channel = np.max(np.abs(centred), axis=2)
        std_by_trial_channel = np.std(self.X, axis=2)
        jump_by_trial_channel = np.max(np.abs(np.diff(self.X, axis=2)), axis=2)
        self.channel_diagnostics = []
        for channel in range(self.X.shape[1]):
            peak = peak_by_trial_channel[:, channel]
            std = std_by_trial_channel[:, channel]
            jump = jump_by_trial_channel[:, channel]
            self.channel_diagnostics.append({
                "index": channel,
                "label": self.channel_labels[channel],
                "peak_median_uv": float(np.median(peak)),
                "peak_p95_uv": float(np.percentile(peak, 95)),
                "peak_max_uv": float(np.max(peak)),
                "amplitude_fail_trials": int(
                    (peak > self.screening_kw["amp_uv"]).sum()),
                "std_median_uv": float(np.median(std)),
                "std_min_uv": float(np.min(std)),
                "std_max_uv": float(np.max(std)),
                "flat_fail_trials": int(
                    (std < self.screening_kw["flat_uv"]).sum()),
                "jump_median_uv": float(np.median(jump)),
                "jump_p95_uv": float(np.percentile(jump, 95)),
                "jump_max_uv": float(np.max(jump)),
                "jump_fail_trials": int(
                    (jump > self.screening_kw["jump_uv"]).sum()),
            })
        self.trial_signal_metrics = {
            "peak_median_uv": float(np.median(
                np.max(peak_by_trial_channel, axis=1))),
            "peak_max_uv": float(np.max(peak_by_trial_channel)),
            "jump_median_uv": float(np.median(
                np.max(jump_by_trial_channel, axis=1))),
            "jump_max_uv": float(np.max(jump_by_trial_channel)),
        }

        self.model_path = str(model_path) if model_path else None
        self.predictions = None
        self.training_accuracy = None
        self.balanced_accuracy = None
        self.confusion_matrix = None
        self.model_error = None
        if model_path:
            self._predict(model_path)

    @staticmethod
    def _physical_label(value):
        text = str(value)
        return text.split("::", 1)[1] if text.startswith("intent::") else text

    def _predict(self, model_path):
        try:
            import joblib
            bundle = joblib.load(model_path)
            if not isinstance(bundle, dict) or "model" not in bundle:
                raise ValueError("model file is not a supported joblib bundle")
            model = bundle["model"]
            raw = np.asarray(model.predict(self.X), dtype=int)
            class_map = {int(k): str(v)
                         for k, v in dict(bundle.get("class_map", {})).items()}
            self.predictions = np.asarray(
                [class_map.get(int(index), f"class_{index}") for index in raw],
                dtype=str)
            intended = np.asarray(
                [self._physical_label(value) for value in self.labels], dtype=str)
            self.training_accuracy = float(np.mean(self.predictions == intended))
            names = sorted(set(intended.tolist()) | set(self.predictions.tolist()))
            recalls, matrix = [], []
            for true_name in names:
                true_mask = intended == true_name
                row = [int((true_mask & (self.predictions == predicted)).sum())
                       for predicted in names]
                matrix.append(row)
                recalls.append(row[names.index(true_name)] / max(1, int(true_mask.sum())))
            self.balanced_accuracy = float(np.mean(recalls)) if recalls else None
            self.confusion_matrix = {"classes": names, "matrix": matrix}
        except Exception as exc:
            self.model_error = f"{type(exc).__name__}: {exc}"

    def trial(self, index):
        if not len(self.X):
            raise IndexError("training dataset is empty")
        index = int(index) % len(self.X)
        predicted = (None if self.predictions is None
                     else str(self.predictions[index]))
        return {
            "index": index, "window": self.X[index],
            "label": str(self.labels[index]),
            "intended": self._physical_label(self.labels[index]),
            "predicted": predicted,
            "correct": (None if predicted is None else
                        predicted == self._physical_label(self.labels[index])),
            "keep": bool(self.keep[index]),
            "reasons": list(self.drop_reasons[index]),
            "subject": str(self.subjects[index]),
            "session": int(self.sessions[index]),
            "file": str(self.files[int(self.sessions[index])]),
        }

    def report(self):
        return {
            "files": [str(path) for path in self.files],
            "trials": int(len(self.X)), "sessions": int(len(self.files)),
            "subjects": sorted(set(self.subjects.tolist())),
            "fs": int(self.fs), "montage": self.montage,
            "channel_labels": list(self.channel_labels),
            "shape": list(self.X.shape),
            "kept": int(self.keep.sum()),
            "dropped": int((~self.keep).sum()),
            "drop_rate": float((~self.keep).mean()) if len(self.keep) else 0.0,
            "drop_reason_counts": {
                name: int(np.asarray(mask, dtype=bool).sum())
                for name, mask in self.screening["reasons"].items()
            },
            "screening_thresholds": dict(self.screening_kw),
            "trial_signal_metrics": dict(self.trial_signal_metrics),
            "channel_diagnostics": list(self.channel_diagnostics),
            "class_counts": dict(self.class_counts),
            "retained_counts": dict(self.retained_counts),
            "model": self.model_path, "model_error": self.model_error,
            "training_accuracy": self.training_accuracy,
            "training_balanced_accuracy": self.balanced_accuracy,
            "confusion_matrix": self.confusion_matrix,
            "warning": ("model metrics use training recordings and are not "
                        "held-session or held-subject validation"),
        }


class GameplayMonitor:
    """Reusable controls and logger used by every pygame GAME page."""

    def __init__(self, app, user=None, root=None, sample_freq=250,
                 enabled=True, feedback_lockout_s=3.2):
        self.enabled = bool(enabled)
        self.logger = (PredictionPerformanceLogger(
            app, user=user, root=root, sample_freq=sample_freq)
            if self.enabled else None)
        self.paused = False
        self.feedback_lockout_s = float(feedback_lockout_s)
        self._suppress_until = -1e18
        self.message = ""
        self.message_at = 0

    def _message(self, text):
        self.message = str(text)
        self.message_at = time.monotonic()
        print(f"[Performance] {self.message}")

    def handle_event(self, event, bci=None):
        """Return True when a pause/feedback control consumed the event."""
        if not self.enabled:
            return False
        import pygame
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.paused = not self.paused
            self.logger.set_paused(self.paused)
            if not self.paused and bci is not None:
                # Discard queued stale windows accumulated while paused, then
                # wait one full 3-second model window so the physical key press
                # cannot become part of the next EEG prediction.
                try:
                    bci.bridge.poll_all()
                except Exception:
                    pass
                self._suppress_until = (time.monotonic()
                                        + self.feedback_lockout_s)
            self._message("PAUSED" if self.paused else "RESUMED")
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            if bci is None:
                return False
            correct = event.button == 1
            if self.logger.mark_feedback(correct):
                self._message("MARKED CORRECT" if correct else "MARKED INCORRECT")
            else:
                self._message("NO UNMARKED PREDICTION TO LABEL")
            # Mouse movement/click EMG can contaminate overlapping EEG windows.
            # Suppress classification for one complete model window and drain
            # those windows instead of letting feedback make accuracy look good.
            self._suppress_until = time.monotonic() + self.feedback_lockout_s
            return True
        return False

    def poll(self, bci):
        if bci is None or self.paused:
            return None
        if not self.enabled:
            return bci.poll()
        if time.monotonic() < self._suppress_until:
            try:
                bci.bridge.poll_all()
            except Exception:
                pass
            return None
        result = bci.poll()
        if result is not None and getattr(bci, "last_window", None) is not None:
            self.logger.record(bci.last_window, result)
        return result

    def draw(self, surface):
        if not self.enabled:
            return
        import pygame
        if self.paused:
            veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            veil.fill((3, 8, 12, 185))
            surface.blit(veil, (0, 0))
            font = pygame.font.SysFont("dejavusans,arial", 38, bold=True)
            small = pygame.font.SysFont("dejavusans,arial", 15)
            text = font.render("PAUSED", True, (244, 201, 93))
            surface.blit(text, text.get_rect(center=surface.get_rect().center))
            hint = small.render("Space: resume", True, (220, 232, 235))
            surface.blit(hint, hint.get_rect(
                center=(surface.get_width() // 2,
                        surface.get_height() // 2 + 44)))
        elif self.message and time.monotonic() - self.message_at < 1.2:
            font = pygame.font.SysFont("dejavusans,arial", 15, bold=True)
            text = font.render(self.message, True, (244, 201, 93))
            bg = text.get_rect(topright=(surface.get_width() - 16, 14)).inflate(18, 10)
            pygame.draw.rect(surface, (12, 25, 34), bg, border_radius=7)
            surface.blit(text, text.get_rect(center=bg.center))

    def close(self):
        if self.logger is not None:
            self.logger.close()
