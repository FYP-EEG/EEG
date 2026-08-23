"""
Author: Anson Li / Brian (extended for BCI Speller)
Created on: 16/8/2026
Purpose: Rectangular, text-labelled tile widget for the BCI virtual keyboard grid.

Why a sibling class instead of editing Button.py:
    pygame_lib.Button is a *circular, icon-only* sprite -- it requires an image file
    (`pygame.image.load(icon)`) and its hit-test `check_within()` is a radius check.
    A 40-tile speller needs rectangular text tiles. Changing Button.py in place would
    break example/rockpaperscissors.py and example/realistic_ui.py, so TileButton is a
    drop-in sibling that keeps the same public surface:
        .update_layout(size, position) / .draw(surface) / .check_within(pos) / .click()

Flicker support is included here (`freq`, `phase`) so that Gap 5 (frame-accurate,
V-synced flickering) can be wired in later without touching the speller UI: the tile
already knows how to render an "on" and an "off" state.
"""

import pygame


class TileButton(pygame.sprite.Sprite):
    """A rectangular key on the speller grid."""

    #: visual states
    IDLE = "idle"
    SCAN = "scan"        # currently swept by the row/column scanner
    ARMED = "armed"      # inside the confirmation dwell
    SELECTED = "selected"

    def __init__(
        self,
        id,
        label,
        value,
        rect,
        func=None,
        kind="char",
        font=None,
        freq=None,
        background_color=(38, 42, 54),
        text_color=(232, 236, 244),
        scan_color=(58, 110, 165),
        armed_color=(196, 132, 32),
        selected_color=(46, 148, 96),
        border_radius=8,
    ):
        """
        :param id:      unique integer id
        :param label:   text drawn on the tile, e.g. "A" or "YES"
        :param value:   what is emitted to the text buffer, e.g. "A" or "Yes, please. "
        :param rect:    pygame.Rect (or 4-tuple) giving position + size in pixels
        :param kind:    'char' | 'phrase' | 'action'  (drives colour accent + buffer routing)
        :param freq:    optional SSVEP flicker frequency in Hz (used later by Gap 5)
        """
        super().__init__()
        self.id = id
        self.label = label
        self.value = value
        self.kind = kind
        self.func = func
        self.freq = freq

        self.background_color = background_color
        self.text_color = text_color
        self.scan_color = scan_color
        self.armed_color = armed_color
        self.selected_color = selected_color
        self.border_radius = border_radius

        self.state = TileButton.IDLE
        self.flicker_on = True          # toggled by the stimulus engine (Gap 5)
        self._font = font

        self.update_layout(rect)

    # ------------------------------------------------------------------ layout
    def update_layout(self, rect, position=None):
        """Reposition/resize. Accepts a Rect, or (size, position) Button-style."""
        if position is not None:                       # Button.py-compatible signature
            size = rect
            rect = pygame.Rect(0, 0, size * 2, size * 2)
            rect.center = position
        self.rect = pygame.Rect(rect)
        self.surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        if self._font is None or self._font.get_height() > self.rect.height * 0.8:
            fs = max(10, int(min(self.rect.height * 0.42, self.rect.width * 0.55)))
            self._font = pygame.font.SysFont("dejavusans,arial", fs, bold=True)

    def set_font(self, font):
        self._font = font

    # ------------------------------------------------------------------- colour
    def _fill_color(self):
        if self.state == TileButton.SELECTED:
            return self.selected_color
        if self.state == TileButton.ARMED:
            return self.armed_color
        if self.state == TileButton.SCAN:
            return self.scan_color
        if self.kind == "phrase":
            return (48, 46, 66)
        if self.kind == "action":
            return (58, 40, 44)
        return self.background_color

    # ------------------------------------------------------------------- render
    def draw(self, surface, progress=0.0):
        """
        :param progress: 0..1 dwell/debounce progress, drawn as a fill bar at the
                         bottom of the tile so the user gets feedback before commit.
        """
        self.surf.fill((0, 0, 0, 0))
        w, h = self.rect.size
        body = pygame.Rect(0, 0, w, h)

        color = self._fill_color()
        if not self.flicker_on:
            color = tuple(max(0, c - 55) for c in color)

        pygame.draw.rect(self.surf, color, body, border_radius=self.border_radius)
        pygame.draw.rect(self.surf, (86, 92, 112), body, width=2,
                         border_radius=self.border_radius)

        if progress > 0.0:
            bar = pygame.Rect(3, h - 8, int((w - 6) * min(1.0, progress)), 5)
            pygame.draw.rect(self.surf, (250, 214, 110), bar, border_radius=3)

        label = self.label
        txt = self._font.render(label, True, self.text_color)
        if txt.get_width() > w - 8:                      # shrink long phrase labels
            small = pygame.font.SysFont("dejavusans,arial",
                                        max(9, int(self._font.get_height() * 0.55)),
                                        bold=True)
            txt = small.render(label, True, self.text_color)
        self.surf.blit(txt, txt.get_rect(center=(w // 2, h // 2)))

        surface.blit(self.surf, self.rect)

    # -------------------------------------------------------------- interaction
    def check_within(self, pos):
        """Rectangular hit test (mouse fallback / developer testing)."""
        return self.rect.collidepoint(pos)

    def click(self):
        if self.func:
            return self.func(self)
        return self.value

    def __repr__(self):
        return f"<TileButton {self.id} {self.label!r} kind={self.kind}>"
