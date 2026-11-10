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
