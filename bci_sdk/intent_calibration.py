"""
Author: Anson
Created on: 19/10/2026
Purpose: reusable in-app calibration tab for motor-intent pygame demos.

The panel configures action -> physical-intention mappings, records a balanced
block from a Cyton, saves the shared .npz format, and automatically trains the
reusable task-set model. Dry-run rehearses timing/UI but deliberately does not
save or train because BrainFlow's synthetic signal contains no finger MI.
"""

import random
import threading
import time
from pathlib import Path

import numpy as np

from ML.motor_intent import (BODY_INTENTS, FINGER_INTENTS, IDLE,
                             MotorIntentMap, assess_intent_set,
                             canonical_intents, intent_display_name,
                             intent_instruction, intent_profile_dir,
                             make_intent_label, normalize_intent)


class _Capture:
    def __init__(self, backend):
        self.backend = backend
        info = backend.info()
        self.fs = int(info["fs"])
        self.n_channels = int(info["n_channels"])
        self._chunks, self._n = [], 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def n_samples(self):
        return self._n

    def start(self):
        self.backend.start()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="IntentCalibrationCapture")
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                chunk = self.backend.read_available()
            except Exception:
                chunk = None
            if chunk is not None and chunk.size:
                with self._lock:
                    self._chunks.append(np.asarray(chunk, dtype=np.float32))
                    self._n += chunk.shape[1]
                time.sleep(.001)
            else:
                time.sleep(.005)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self.backend.stop()
        except Exception:
            pass
        with self._lock:
            if not self._chunks:
                return np.zeros((self.n_channels, 0), dtype=np.float32)
            return np.concatenate(self._chunks, axis=1)


class IntentCalibrationPanel:
    """Drop-in pygame calibration UI shared by all intent-mi examples.

    Call ``handle_event()``, ``update()`` and ``draw()`` once per frame. F2 or
    the CALIBRATE tab opens it. Mapping changes happen before recording with
    Up/Down and Left/Right; Enter starts a block. A real Cyton block is saved and
    trained automatically. Fewer than three blocks are marked unreliable.
    """

    SETUP, REST, CUE, FINISH, TRAINING, RESULT, ERROR = (
        "SETUP", "REST", "CUE", "FINISH", "TRAINING", "RESULT", "ERROR")

    def __init__(self, targets, intent_map, user=None, serial_port=None,
                 source="dry-run", montage="cyton8_motor", action_trials=30,
                 idle_trials=10, cue_s=4.0, rest_s=2.0, root=None,
                 allowed_intents=None, enabled=True, on_before_record=None,
                 on_model_ready=None):
        import pygame
        self.pg = pygame
        self.targets = list(targets)
        self._by_action = {str(t.name): t for t in self.targets}
        self.mapping = MotorIntentMap(intent_map)
        missing = set(self.mapping.action_to_intent) - set(self._by_action)
        if missing:
            raise ValueError(f"calibration targets missing actions: {sorted(missing)}")
        self.user = str(user or "").strip()
        self.serial_port = serial_port or ""
        self.source = source
        self.montage = montage
        self.action_trials = max(1, int(action_trials))
        self.idle_trials = max(0, int(idle_trials))
        self.cue_ms = int(float(cue_s) * 1000)
        self.rest_ms = int(float(rest_s) * 1000)
        self.root = Path(root) if root else None
        self.allowed_intents = tuple(
            normalize_intent(v, allow_idle=False)
            for v in (allowed_intents or FINGER_INTENTS))
        self.enabled = bool(enabled)
        self.on_before_record = on_before_record
        self.on_model_ready = on_model_ready

        self.visible = False
        self.state = self.SETUP
        self.selected_row = 0
        self.tab_rect = pygame.Rect(0, 0, 118, 34)
        self.start_rect = pygame.Rect(0, 0, 220, 44)
        self.source_rect = pygame.Rect(0, 0, 160, 34)
        self.font = self.small = self.big = None
        self.message = ""
        self.error = ""
        self.result = None
        self._worker = None
        self._capture = None
        self._stream = None
        self._order = []
        self._log = []
        self._trial = 0
        self._target = None
        self._intent = None
        self._cue_sample = 0
        self._phase_end = 0.0
        self._done_at = 0.0

    # ------------------------------------------------------------ public API
    @property
    def is_recording(self):
        return self.state in (self.REST, self.CUE, self.FINISH, self.TRAINING)

    @property
    def intent_map(self):
        return dict(self.mapping.action_to_intent)

    def toggle(self):
        if not self.enabled or self.is_recording:
            return
        self.visible = not self.visible
        if self.visible and self.state not in (self.RESULT, self.ERROR):
            self.state = self.SETUP

    def handle_event(self, event):
        """Return True when the panel consumed the event."""
        if not self.enabled:
            return False
        pg = self.pg
        if event.type == pg.QUIT:
            return False
        if event.type == pg.KEYDOWN and event.key == pg.K_F2:
            self.toggle(); return True
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.tab_rect.collidepoint(event.pos):
                self.toggle(); return True
            if not self.visible:
                return False
            if self.state == self.SETUP:
                if self.start_rect.collidepoint(event.pos):
                    self.start(); return True
                if self.source_rect.collidepoint(event.pos):
                    self._toggle_source(); return True
                row = self._row_at(event.pos)
                if row is not None:
                    self.selected_row = row; return True
            return True
        if not self.visible:
            return False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                if self.is_recording:
                    self.abort("Calibration aborted by user")
                else:
                    self.visible = False
                return True
            if self.state in (self.RESULT, self.ERROR):
                if event.key in (pg.K_RETURN, pg.K_SPACE):
                    self.state = self.SETUP; self.result = None; self.error = ""
                return True
            if self.state == self.SETUP:
                if event.key == pg.K_UP:
                    self.selected_row = (self.selected_row - 1) % len(self.targets)
                elif event.key == pg.K_DOWN:
                    self.selected_row = (self.selected_row + 1) % len(self.targets)
                elif event.key == pg.K_LEFT:
                    self._cycle_intent(-1)
                elif event.key == pg.K_RIGHT:
                    self._cycle_intent(1)
                elif event.key == pg.K_d:
                    self._toggle_source()
                elif event.key in (pg.K_RETURN, pg.K_SPACE):
                    self.start()
                return True
            return True
        return self.visible

    def update(self):
        if not self.visible:
            return
        now = time.monotonic() * 1000.0
        if self.state == self.REST and now >= self._phase_end:
            if self._trial >= len(self._order):
                self.state = self.FINISH
                self._done_at = now
                self.message = "Finalising recording..."
            else:
                self._start_cue(now)
        elif self.state == self.CUE and now >= self._phase_end:
            self._end_cue(now)
        elif self.state == self.FINISH and now - self._done_at >= 400:
            self._begin_finish_worker()
        elif self.state == self.TRAINING and self._worker is not None and \
                not self._worker.is_alive():
            self._worker = None
            if self.error:
                self.state = self.ERROR
            else:
                self.state = self.RESULT
                if self.on_model_ready and self.result:
                    try:
                        self.on_model_ready(self.result)
                    except Exception as exc:
                        self.error = f"model-ready callback failed: {exc}"
                        self.state = self.ERROR

    def draw(self, surface):
        if not self.enabled:
            return
        self._fonts()
        w, h = surface.get_size()
        self.tab_rect = self.pg.Rect(w - 132, 10, 118, 34)
        tab_col = (71, 215, 208) if self.visible else (42, 60, 70)
        self.pg.draw.rect(surface, (10, 20, 29), self.tab_rect, border_radius=9)
        self.pg.draw.rect(surface, tab_col, self.tab_rect, 2, border_radius=9)
        tab = self.small.render("F2  CALIBRATE", True, (230, 242, 244))
        surface.blit(tab, tab.get_rect(center=self.tab_rect.center))
        if not self.visible:
            return

        veil = self.pg.Surface((w, h), self.pg.SRCALPHA)
        veil.fill((4, 10, 16, 230))
        surface.blit(veil, (0, 0))
        panel = self.pg.Rect(max(18, w // 2 - 330), 56, min(660, w - 36), h - 78)
        self.pg.draw.rect(surface, (15, 27, 38), panel, border_radius=16)
        self.pg.draw.rect(surface, (55, 83, 96), panel, 1, border_radius=16)
        title = self.big.render("MOTOR INTENT CALIBRATION", True, (239, 247, 248))
        surface.blit(title, (panel.x + 24, panel.y + 20))
        subtitle = self.small.render(
            f"user: {self.user or '(missing)'}   task-set model   F2/Esc: close",
            True, (145, 170, 179))
        surface.blit(subtitle, (panel.x + 25, panel.y + 54))

        if self.state == self.SETUP:
            self._draw_setup(surface, panel)
        elif self.state in (self.REST, self.CUE):
            self._draw_recording(surface, panel, now=time.monotonic() * 1000.0)
        elif self.state in (self.FINISH, self.TRAINING):
            self._draw_center(surface, panel, self.message or "Training locally...",
                              "Do not close the application")
        elif self.state == self.RESULT:
            self._draw_result(surface, panel)
        else:
            self._draw_center(surface, panel, "CALIBRATION ERROR", self.error,
                              color=(255, 128, 111))

    # --------------------------------------------------------------- setup UI
    def _fonts(self):
        if self.font is None:
            self.big = self.pg.font.SysFont("dejavusans,arial", 25, bold=True)
            self.font = self.pg.font.SysFont("dejavusans,arial", 17, bold=True)
            self.small = self.pg.font.SysFont("dejavusans,arial", 13)

    def _row_step(self, panel):
        # Fits up to ten actions on a 600 px window while retaining controls.
        return min(47, max(30, int((panel.height - 205) / max(1, len(self.targets)))))

    def _row_rect(self, panel, index):
        step = self._row_step(panel)
        return self.pg.Rect(panel.x + 25, panel.y + 86 + index * step,
                            panel.width - 50, max(25, step - 6))

    def _row_at(self, pos):
        surface = self.pg.display.get_surface()
        if surface is None:
            return None
        w, h = surface.get_size()
        panel = self.pg.Rect(max(18, w // 2 - 330), 56, min(660, w - 36), h - 78)
        for i in range(len(self.targets)):
            if self._row_rect(panel, i).collidepoint(pos):
                return i
        return None

    def _draw_setup(self, surface, panel):
        mapping = self.mapping.action_to_intent
        for i, target in enumerate(self.targets):
            row = self._row_rect(panel, i)
            selected = i == self.selected_row
            self.pg.draw.rect(surface, (26, 45, 58) if selected else (12, 24, 34),
                              row, border_radius=8)
            if selected:
                self.pg.draw.rect(surface, (71, 215, 208), row, 2, border_radius=8)
            action = self.font.render(str(target.name), True, (230, 240, 243))
            value = self.font.render(intent_display_name(mapping[target.name]), True,
                                     (244, 201, 93))
            surface.blit(action, (row.x + 12, row.y + 8))
            surface.blit(value, value.get_rect(midright=(row.right - 12, row.centery)))

        below = panel.y + 92 + len(self.targets) * self._row_step(panel)
        self.source_rect = self.pg.Rect(panel.x + 25, below, 160, 34)
        self.pg.draw.rect(surface, (20, 37, 48), self.source_rect, border_radius=8)
        source = self.small.render(
            "SOURCE: " + ("CYTON" if self.source == "cyton" else "DRY RUN"),
            True, (244, 201, 93))
        surface.blit(source, source.get_rect(center=self.source_rect.center))
        details = self.small.render(
            f"{self.action_trials} action + {self.idle_trials} idle trials · "
            f"{self.cue_ms/1000:g}s cue · automatic local training",
            True, (151, 176, 185))
        surface.blit(details, (panel.x + 200, below + 9))

        assessment = self.mapping.assessment
        compact = len(self.targets) > 6
        warning_y = below + 48
        if not compact:
            level = self.small.render(
                f"Difficulty: {assessment['difficulty']} · {assessment['task_set']}",
                True, (125, 219, 155))
            surface.blit(level, (panel.x + 25, warning_y))
            for index, warning in enumerate(assessment["warnings"][:3]):
                line = warning if len(warning) <= 84 else warning[:81] + "..."
                text = self.small.render("WARNING: " + line, True, (239, 165, 139))
                surface.blit(text, (panel.x + 25, warning_y + 24 + index * 20))

        self.start_rect = self.pg.Rect(panel.centerx - 110, panel.bottom - 62, 220, 42)
        self.pg.draw.rect(surface, (34, 111, 106), self.start_rect, border_radius=10)
        start = self.font.render("START CALIBRATION", True, (244, 251, 251))
        surface.blit(start, start.get_rect(center=self.start_rect.center))
        if not compact:
            help_text = self.small.render(
                "↑/↓ action · ←/→ change intent · D source · Enter start",
                True, (128, 151, 160))
            surface.blit(help_text, help_text.get_rect(center=(panel.centerx,
                                                               self.start_rect.y - 14)))

    def _draw_recording(self, surface, panel, now):
        left = max(0.0, (self._phase_end - now) / 1000.0)
        if self.state == self.REST:
            title, instruction = "REST", "Relax and prepare for the next cue."
        else:
            title = intent_display_name(self._intent)
            action = None if self._target is None else self._target.name
            instruction = intent_instruction(self._intent, action)
        t = self.big.render(title, True, (244, 201, 93))
        surface.blit(t, t.get_rect(center=(panel.centerx, panel.y + 165)))
        inst = self.small.render(instruction[:100], True, (207, 222, 226))
        surface.blit(inst, inst.get_rect(center=(panel.centerx, panel.y + 205)))
        counter = self.font.render(
            f"trial {self._trial}/{len(self._order)}    {left:4.1f}s",
            True, (235, 244, 246))
        surface.blit(counter, counter.get_rect(center=(panel.centerx, panel.y + 245)))
        width = panel.width - 80
        total = self.cue_ms if self.state == self.CUE else self.rest_ms
        frac = 1 - min(1, max(0, (self._phase_end - now) / max(1, total)))
        bg = self.pg.Rect(panel.x + 40, panel.y + 280, width, 12)
        self.pg.draw.rect(surface, (38, 57, 68), bg, border_radius=6)
        self.pg.draw.rect(surface, (71, 215, 208),
                          self.pg.Rect(bg.x, bg.y, int(width * frac), 12),
                          border_radius=6)
        stop = self.small.render("Esc aborts and discards this block", True,
                                 (157, 176, 182))
        surface.blit(stop, stop.get_rect(center=(panel.centerx, panel.bottom - 28)))

    def _draw_center(self, surface, panel, title, subtitle, color=(244, 201, 93)):
        t = self.big.render(title, True, color)
        surface.blit(t, t.get_rect(center=(panel.centerx, panel.centery - 20)))
        s = self.small.render(str(subtitle)[:105], True, (180, 200, 206))
        surface.blit(s, s.get_rect(center=(panel.centerx, panel.centery + 25)))

    def _draw_result(self, surface, panel):
        result = self.result or {}
        if result.get("dry_run"):
            self._draw_center(surface, panel, "DRY RUN COMPLETE",
                              "No EEG/model was saved. Switch SOURCE to CYTON for real calibration.")
            return
        accuracy = result.get("accuracy")
        acc = "unavailable" if accuracy is None else f"{accuracy:.1%}"
        title = self.big.render("MODEL TRAINED", True, (125, 219, 155))
        surface.blit(title, title.get_rect(center=(panel.centerx, panel.y + 145)))
        lines = [
            f"pipeline: {result.get('pipeline', '?')}",
            f"session-held-out balanced accuracy: {acc}",
            f"sessions: {result.get('n_sessions', 0)} / 3 recommended",
            f"reliable score: {'YES' if result.get('reliable') else 'NO'}",
        ]
        for i, line in enumerate(lines):
            text = self.font.render(line, True, (214, 228, 231))
            surface.blit(text, text.get_rect(center=(panel.centerx,
                                                     panel.y + 195 + i * 34)))
        if not result.get("reliable"):
            warning = self.small.render(
                "Record more separate blocks. The model runs, but its score is not trustworthy.",
                True, (239, 165, 139))
            surface.blit(warning, warning.get_rect(center=(panel.centerx,
                                                           panel.y + 345)))
        done = self.small.render("Enter: return to setup · F2/Esc: return to game",
                                 True, (143, 165, 173))
        surface.blit(done, done.get_rect(center=(panel.centerx, panel.bottom - 28)))

    # ----------------------------------------------------------- configuration
    def _toggle_source(self):
        self.source = "cyton" if self.source == "dry-run" else "dry-run"

    def _cycle_intent(self, delta):
        action = self.targets[self.selected_row].name
        current = self.mapping.action_to_intent[action]
        index = self.allowed_intents.index(current) if current in self.allowed_intents else 0
        for step in range(1, len(self.allowed_intents) + 1):
            candidate = self.allowed_intents[(index + delta * step) % len(self.allowed_intents)]
            if candidate not in self.mapping.intent_to_action or candidate == current:
                updated = dict(self.mapping.action_to_intent)
                updated[action] = candidate
                self.mapping = MotorIntentMap(updated)
                break

    # -------------------------------------------------------------- acquisition
    def start(self):
        if self.state != self.SETUP:
            return
        if not self.user:
            self.error = "Start the app with --user <name> before calibration."
            self.state = self.ERROR; return
        if len(self.mapping.intents) == 1 and self.idle_trials < 8:
            self.error = "One-intent calibration requires at least 8 idle trials."
            self.state = self.ERROR; return
        if self.source == "cyton" and not self.serial_port:
            self.error = "CYTON source needs --port COM4 (or the correct serial port)."
            self.state = self.ERROR; return
        if self.on_before_record:
            try:
                self.on_before_record()
            except Exception as exc:
                self.error = f"could not stop live BCI before calibration: {exc}"
                self.state = self.ERROR; return
        try:
            from stream_bridge import SyntheticBackend, BrainFlowBackend
            backend = (SyntheticBackend(fs=250, n_channels=8)
                       if self.source == "dry-run" else
                       BrainFlowBackend(board_id=0, serial_port=self.serial_port,
                                        n_channels=8))
            self._capture = _Capture(backend).start()
        except Exception as exc:
            self.error = f"could not start {self.source}: {exc}"
            self.state = self.ERROR; return

        actions = [self._by_action[name]
                   for name in self.mapping.action_to_intent]
        order = []
        while len(order) < self.action_trials:
            batch = list(actions); random.shuffle(batch); order.extend(batch)
        self._order = order[:self.action_trials] + [None] * self.idle_trials
        random.shuffle(self._order)
        self._log, self._trial = [], 0
        self.result, self.error = None, ""
        self.state = self.REST
        self._phase_end = time.monotonic() * 1000.0 + self.rest_ms
        self.message = "Recording"

    def _start_cue(self, now):
        self._target = self._order[self._trial]
        self._trial += 1
        for target in self.targets:
            target.stop_flash()
        if self._target is None:
            self._intent = IDLE
        else:
            self._intent = self.mapping.action_to_intent[self._target.name]
            self._target.flash(loop=True, waves=2, pulse_hz=.5)
        self._cue_sample = self._capture.n_samples
        self.state = self.CUE
        self._phase_end = now + self.cue_ms

    def _end_cue(self, now):
        if self._target is not None:
            self._target.stop_flash()
        self._log.append({
            "intent": self._intent,
            "label": make_intent_label(self._intent),
            "cue_sample": int(self._cue_sample),
        })
        self.state = self.REST
        self._phase_end = now + self.rest_ms

    def abort(self, reason="Calibration aborted"):
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:
                pass
        self._capture = None
        for target in self.targets:
            target.stop_flash()
        self.error = reason
        self.state = self.ERROR

    def _begin_finish_worker(self):
        if self.state == self.TRAINING:
            return
        self.state = self.TRAINING
        self.message = ("Checking dry run..." if self.source == "dry-run"
                        else "Saving and training locally...")
        self._worker = threading.Thread(target=self._finish_worker, daemon=True,
                                        name="IntentCalibrationTrainer")
        self._worker.start()

    def _finish_worker(self):
        try:
            fs = int(self._capture.fs)
            stream = self._capture.stop()
            self._capture = None
            if self.source == "dry-run":
                self.result = {"dry_run": True, "n_trials": len(self._log)}
                return
            n = int(3.0 * fs)
            X, labels = [], []
            for row in self._log:
                start = int(row["cue_sample"])
                if start >= 0 and start + n <= stream.shape[1]:
                    X.append(stream[:, start:start + n])
                    labels.append(row["label"])
            if not X:
                raise RuntimeError("no complete 3-second epochs were captured")
            values = canonical_intents(self.mapping.intents)
            class_names = list(values) + ([IDLE] if self.idle_trials else [])
            to_class = {name: i for i, name in enumerate(class_names)}
            y = np.array([to_class[label.split("::", 1)[1]] for label in labels])
            from EEG.calibration_record import save
            out = intent_profile_dir(self.user, values, root=self.root)
            save(np.asarray(X), y, np.asarray(labels), self.user, fs,
                 self.montage, (0.0, 0.0), out)
            from ML.train_intent_mi import train_intent_user
            self.result = train_intent_user(
                self.user, values, root=self.root, verbose=False)
            if self.result is None:
                raise RuntimeError("recording saved, but too few trials remained to train")
        except Exception as exc:
            self.error = str(exc)
