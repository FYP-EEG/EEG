"""
Author: Anson
Created on: 8/11/2026
Purpose: reusable two-choice Game / Calibration landing page for pygame demos.
"""


class GameCalibrationMenu:
    """Mouse-clickable landing page controlled by Left/Right and Enter."""

    def __init__(self, title="BCI DEMO"):
        import pygame
        self.pg = pygame
        self.title = title
        self.visible = True
        self.index = 0
        self.game_rect = pygame.Rect(0, 0, 250, 92)
        self.cal_rect = pygame.Rect(0, 0, 250, 92)
        self.big = self.font = self.small = None

    def _fonts(self):
        if self.font is None:
            self.big = self.pg.font.SysFont("dejavusans,arial", 34, bold=True)
            self.font = self.pg.font.SysFont("dejavusans,arial", 23, bold=True)
            self.small = self.pg.font.SysFont("dejavusans,arial", 14)

    def handle_event(self, event):
        """Return ``game``, ``calibration``, ``consumed`` or None."""
        if not self.visible:
            return None
        pg = self.pg
        if event.type == pg.QUIT:
            return None
        if event.type == pg.KEYDOWN:
            if event.key in (pg.K_ESCAPE, pg.K_BACKSPACE):
                return None
            if event.key in (pg.K_LEFT, pg.K_RIGHT, pg.K_TAB):
                self.index = 1 - self.index
                return "consumed"
            if event.key in (pg.K_RETURN, pg.K_SPACE):
                return "game" if self.index == 0 else "calibration"
            return "consumed"
        if event.type == pg.MOUSEMOTION:
            if self.game_rect.collidepoint(event.pos): self.index = 0
            elif self.cal_rect.collidepoint(event.pos): self.index = 1
            return "consumed"
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.game_rect.collidepoint(event.pos): return "game"
            if self.cal_rect.collidepoint(event.pos): return "calibration"
            return "consumed"
        return "consumed"

    def draw(self, surface):
        if not self.visible:
            return
        self._fonts()
        pg = self.pg
        w, h = surface.get_size()
        veil = pg.Surface((w, h), pg.SRCALPHA); veil.fill((4, 10, 16, 242))
        surface.blit(veil, (0, 0))
        title = self.big.render(self.title, True, (238, 247, 249))
        surface.blit(title, title.get_rect(center=(w // 2, int(h * .25))))
        subtitle = self.small.render(
            "Choose with mouse or Left/Right + Enter", True, (145, 170, 179))
        surface.blit(subtitle, subtitle.get_rect(center=(w // 2, int(h * .32))))
        gap = 34
        width = min(270, int(w * .34)); height = 94
        left = w // 2 - width - gap // 2
        top = int(h * .43)
        self.game_rect = pg.Rect(left, top, width, height)
        self.cal_rect = pg.Rect(w // 2 + gap // 2, top, width, height)
        for index, (rect, label, sub) in enumerate([
                (self.game_rect, "GAME", "Play with keyboard, mouse or BCI"),
                (self.cal_rect, "CALIBRATION", "Learn mapping, record and train")]):
            selected = index == self.index
            pg.draw.rect(surface, (29, 66, 75) if selected else (18, 34, 45),
                         rect, border_radius=14)
            pg.draw.rect(surface, (71, 215, 208) if selected else (63, 88, 99),
                         rect, 3 if selected else 1, border_radius=14)
            text = self.font.render(label, True,
                                    (244, 201, 93) if selected else (222, 234, 237))
            surface.blit(text, text.get_rect(center=(rect.centerx, rect.y + 34)))
            small = self.small.render(sub, True, (157, 179, 187))
            surface.blit(small, small.get_rect(center=(rect.centerx, rect.y + 68)))
        key = self.small.render("Esc / Backspace closes the application",
                                True, (117, 139, 148))
        surface.blit(key, key.get_rect(center=(w // 2, int(h * .72))))
