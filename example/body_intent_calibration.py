"""
Author: Anson
Created on: 8/11/2026
Purpose: universal 13-intention body-map UI driven by EEG/data_record.py.

Preferred command:
    python EEG/data_record.py --intent-mi --subject anson \
        --board-id 0 --serial-port COM4

The recorder injects cues. Running this file directly only previews the body map.
"""

import sys
from pathlib import Path
import pygame

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.motor_intent import UNIVERSAL_INTENTS
from bci_sdk.body_map import draw_body_intent_map

pygame.init()
win = pygame.display.set_mode((920, 680), pygame.RESIZABLE)
pygame.display.set_caption("Universal Motor-Intent Recording")
clock = pygame.time.Clock()
font = pygame.font.SysFont("dejavusans,arial", 17)
big = pygame.font.SysFont("dejavusans,arial", 25, bold=True)


class BodyCueTarget:
    def __init__(self, index, intent):
        self.id, self.name = index, intent
        self.rect = pygame.Rect(index * 12, 0, 10, 10)
        self.is_flashing = False

    def flash(self, **_kwargs):
        self.is_flashing = True
        return self

    def stop_flash(self):
        self.is_flashing = False
        return self

    def click(self):
        return None

    def draw(self, _surface):
        # The complete body map is drawn once by the application.
        return None


BCI_RECORD_TARGETS = [BodyCueTarget(i, intent)
                      for i, intent in enumerate(UNIVERSAL_INTENTS)]
BCI_INTENT_MAP = {intent: intent for intent in UNIVERSAL_INTENTS}


def highlighted_intent():
    for target in BCI_RECORD_TARGETS:
        if target.is_flashing:
            return target.name
    return "idle"


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE, pygame.K_BACKSPACE):
            running = False
        elif event.type == pygame.VIDEORESIZE:
            win = pygame.display.set_mode(event.size, pygame.RESIZABLE)

    w, h = win.get_size()
    win.fill((8, 16, 24))
    title = big.render("UNIVERSAL MOTOR-INTENT BODY MAP", True, (235, 245, 247))
    win.blit(title, title.get_rect(center=(w // 2, 32)))
    note = font.render(
        "10 fingers + left hand + right hand + both feet · keep the real body still",
        True, (147, 172, 181))
    win.blit(note, note.get_rect(center=(w // 2, 62)))
    map_rect = pygame.Rect(40, 78, w - 80, h - 108)
    pygame.draw.rect(win, (13, 27, 38), map_rect, border_radius=18)
    pygame.draw.rect(win, (50, 79, 91), map_rect, 1, border_radius=18)
    draw_body_intent_map(win, map_rect, highlighted_intent())

    recorder_frame = globals().get("__bci_recorder_frame__")
    if callable(recorder_frame):
        recorder_frame()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
