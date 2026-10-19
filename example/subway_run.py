"""
Author: Anson
Created on: 19/10/2026
Purpose: asset-free Subway-style pygame demo for three broad MI intentions.

Default reusable intent mapping:
    left_hand  -> move_left
    right_hand -> move_right
    both_feet  -> jump

Keyboard fallback:
    LEFT / A, RIGHT / D, UP / SPACE

Run without BCI:
    python example/subway_run.py

Run after calibrating the task set:
    python example/subway_run.py --bci intent-mi --user anson --port COM4
"""

import argparse
import math
import random
import sys
from pathlib import Path

import pygame

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pygame.init()
win = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
pygame.display.set_caption("Motor Intent Runner")
clock = pygame.time.Clock()

BCI_INTENT_MAP = {
    "move_left": "left_hand",
    "move_right": "right_hand",
    "jump": "both_feet",
}


class ActionTarget:
    """Small action tile that also satisfies data_record.py's cue contract."""

    def __init__(self, index, name, label, key_hint):
        self.id, self.name = index, name
        self.label, self.key_hint = label, key_hint
        self.rect = pygame.Rect(0, 0, 190, 58)
        self.is_flashing = False
        self._flash_start = 0

    def flash(self, **_kwargs):
        self.is_flashing = True
        self._flash_start = pygame.time.get_ticks()
        return self

    def stop_flash(self):
        self.is_flashing = False
        return self

    def click(self):
        return None

    def draw(self, surface, font, small):
        pulse = 0
        if self.is_flashing:
            pulse = int(30 + 25 * (0.5 + 0.5 * math.sin(
                (pygame.time.get_ticks() - self._flash_start) / 180.0)))
        face = (28 + pulse, 45 + pulse, 58 + pulse)
        border = (244, 201, 93) if self.is_flashing else (74, 100, 114)
        pygame.draw.rect(surface, face, self.rect, border_radius=12)
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=12)
        text = font.render(self.label, True, (238, 246, 248))
        surface.blit(text, (self.rect.x + 13, self.rect.y + 8))
        hint = small.render(self.key_hint, True, (150, 172, 180))
        surface.blit(hint, (self.rect.x + 13, self.rect.y + 34))


move_left_target = ActionTarget(0, "move_left", "MOVE LEFT", "LEFT HAND · ← / A")
move_right_target = ActionTarget(1, "move_right", "MOVE RIGHT", "RIGHT HAND · → / D")
jump_target = ActionTarget(2, "jump", "JUMP", "BOTH FEET · ↑ / SPACE")
BCI_RECORD_TARGETS = [move_left_target, move_right_target, jump_target]
targets = BCI_RECORD_TARGETS

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--bci", choices=["sine", "replay", "cyton", "intent-mi"],
                 default=None)
_ap.add_argument("--user", default=None)
_ap.add_argument("--port", default=None)
_ap.add_argument("--recording", default=None)
_ap.add_argument("--scheme", choices=["tree", "scan"], default="tree")
_ap.add_argument("--intent-map", dest="intent_map", default=None)
_ap.add_argument("--calibration-source", choices=["dry-run", "cyton"],
                 default="dry-run")
_ap.add_argument("--calibration-trials", type=int, default=36)
_ap.add_argument("--calibration-idle", type=int, default=12)
_ap.add_argument("--calibration-cue", type=float, default=4.0)
_ap.add_argument("--calibration-rest", type=float, default=2.0)
_args, _unknown = _ap.parse_known_args()
if _args.intent_map:
    from ML.motor_intent import parse_intent_map
    BCI_INTENT_MAP = parse_intent_map(_args.intent_map)

lane = 1
jump_started = None
last_action = "READY"
last_action_at = 0
score = 0.0
obstacles = []
spawn_at = 0


def move_left():
    global lane, last_action, last_action_at
    lane = max(0, lane - 1)
    last_action, last_action_at = "MOVE LEFT", pygame.time.get_ticks()


def move_right():
    global lane, last_action, last_action_at
    lane = min(2, lane + 1)
    last_action, last_action_at = "MOVE RIGHT", pygame.time.get_ticks()


def jump():
    global jump_started, last_action, last_action_at
    if jump_started is None:
        jump_started = pygame.time.get_ticks()
    last_action, last_action_at = "JUMP", pygame.time.get_ticks()


ACTIONS = {"move_left": move_left, "move_right": move_right, "jump": jump}


def choose(target):
    ACTIONS[target.name]()
    for item in targets:
        item.stop_flash()
    target.flash()


from bci_sdk.live import LiveBCI
from bci_sdk.intent_calibration import IntentCalibrationPanel
from ML.motor_intent import BODY_INTENTS

bci = None

def start_bci(source=None):
    global bci
    selected_source = source or _args.bci
    if not selected_source:
        return None
    try:
        bci = LiveBCI(
            targets, source=selected_source, user=_args.user,
            serial_port=_args.port, recording=_args.recording,
            scheme=_args.scheme, intent_map=BCI_INTENT_MAP,
            on_select=choose, verbose=True).start()
    except FileNotFoundError as exc:
        print(f"[BCI not started] {exc}")
        bci = None
    return bci


def stop_bci_for_calibration():
    global bci
    if bci is not None:
        bci.stop(); bci = None


def model_ready(_result):
    global BCI_INTENT_MAP
    BCI_INTENT_MAP = calibration.intent_map
    if calibration.source == "cyton":
        start_bci("intent-mi")


if _args.bci:
    start_bci()

calibration = IntentCalibrationPanel(
    targets, BCI_INTENT_MAP, user=_args.user, serial_port=_args.port,
    source=_args.calibration_source,
    action_trials=_args.calibration_trials,
    idle_trials=_args.calibration_idle,
    cue_s=_args.calibration_cue, rest_s=_args.calibration_rest,
    allowed_intents=BODY_INTENTS,
    enabled="__bci_recorder_frame__" not in globals(),
    on_before_record=stop_bci_for_calibration,
    on_model_ready=model_ready)


def layout(width, height):
    gap = 16
    total = 3 * 190 + 2 * gap
    x = (width - total) // 2
    for index, target in enumerate(targets):
        target.rect = pygame.Rect(x + index * (190 + gap), 16, 190, 58)


def lane_centres(width):
    road_w = min(500, int(width * 0.70))
    left = (width - road_w) // 2
    return [left + road_w * (i + .5) / 3 for i in range(3)], left, road_w


layout(*win.get_size())
font = pygame.font.SysFont("dejavusans,arial", 19, bold=True)
small = pygame.font.SysFont("dejavusans,arial", 12)
big = pygame.font.SysFont("dejavusans,arial", 29, bold=True)
running = True

try:
    while running:
        now = pygame.time.get_ticks()
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if calibration.handle_event(event):
                continue
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                    running = False
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    move_left()
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    move_right()
                elif event.key in (pygame.K_UP, pygame.K_SPACE, pygame.K_w):
                    jump()
            elif event.type == pygame.VIDEORESIZE:
                win = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                layout(*event.size)

        calibration.update()
        if calibration.visible:
            dt = 0.0
        if bci is not None and not calibration.visible:
            bci.poll()

        width, height = win.get_size()
        centres, road_left, road_w = lane_centres(width)
        if now >= spawn_at:
            blocked = random.randrange(3)
            obstacles.append([blocked, -40.0])
            spawn_at = now + random.randint(1200, 2100)
        speed = 170 + min(130, score / 8)
        for obstacle in obstacles:
            obstacle[1] += speed * dt
        obstacles[:] = [o for o in obstacles if o[1] < height + 60]
        score += dt * 10

        jump_height = 0.0
        if jump_started is not None:
            age = (now - jump_started) / 1000.0
            if age >= 1.0:
                jump_started = None
            else:
                jump_height = math.sin(age * math.pi) * 105

        win.fill((8, 15, 23))
        pygame.draw.rect(win, (20, 31, 40),
                         pygame.Rect(road_left, 92, road_w, height - 92))
        for i in (1, 2):
            x = road_left + road_w * i / 3
            for y in range(100, height, 52):
                pygame.draw.line(win, (57, 77, 87), (x, y), (x, y + 25), 3)

        for obstacle_lane, y in obstacles:
            x = centres[obstacle_lane]
            rect = pygame.Rect(0, 0, 64, 42)
            rect.center = (int(x), int(y))
            pygame.draw.rect(win, (207, 80, 74), rect, border_radius=9)
            pygame.draw.rect(win, (255, 144, 112), rect, 2, border_radius=9)

        player = pygame.Rect(0, 0, 54, 76)
        player.center = (int(centres[lane]), int(height - 70 - jump_height))
        pygame.draw.rect(win, (71, 215, 208), player, border_radius=12)
        pygame.draw.circle(win, (236, 246, 247),
                           (player.centerx, player.y + 17), 10)

        for target in targets:
            target.draw(win, font, small)
        score_s = small.render(f"SCORE {int(score):05d}", True, (180, 198, 205))
        win.blit(score_s, (18, height - 28))
        status = big.render(last_action, True, (244, 201, 93))
        if now - last_action_at < 900 or last_action == "READY":
            win.blit(status, status.get_rect(center=(width // 2, 108)))
        caption = small.render(
            "MI profile: LEFT HAND = left · RIGHT HAND = right · BOTH FEET = jump",
            True, (150, 172, 180))
        win.blit(caption, caption.get_rect(center=(width // 2, height - 22)))
        calibration.draw(win)

        recorder_frame = globals().get("__bci_recorder_frame__")
        if callable(recorder_frame):
            recorder_frame()
        pygame.display.flip()
finally:
    if bci is not None:
        bci.stop()
    pygame.quit()
