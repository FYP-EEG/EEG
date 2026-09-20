"""
Author: Anson Li
Created on: 30/6/2026
Purpose: Library for pygame buttons
"""
import pygame
import math

class Button(pygame.sprite.Sprite):
    def __init__(self, id, name, size, position, icon, func=None, background_color=(128, 128, 128), icon_color=(255, 255, 255), is_3d=True):
        super().__init__()

        self.id = id
        self.name = name
        self.size = size
        self.is_3d = is_3d
        self.icon_color = icon_color
        self.background_color = background_color
        self.offset = 0 if not is_3d else size//8
        self.position = position
        self.img = pygame.image.load(icon).convert_alpha()
        self.func = func

        # --- flash() state -------------------------------------------------
        # Initialised here so draw() is safe even if flash() is never called.
        self._flashing = False
        self._flash_start = 0
        self._flash_duration = 0
        self._flash_waves = 3
        self._flash_color = (255, 200, 60)
        self._flash_spread = 0.9
        self._flash_loop = False
        self._flash_pulse_hz = 0.5
        self._flash_surf = None          # cached, reallocated only on resize

        self.update_layout(self.size, self.position)

    def update_layout(self, size, position):
        self.position = position
        """
            size = width*0.05
            so when increase width only, height will overflow
            solution center_h should be adjust with radius:
            center_h = (height-radius)*0.95
        """

        # surface basically a block to draw on
        self.surf = pygame.Surface((size*2, (size+self.offset)*2), pygame.SRCALPHA)
        # draw white circle within block
        if self.is_3d:
            self.offset = size//8 #grey circle for visual effect
            pygame.draw.circle(self.surf, color=self.background_color, center=(size, size+self.offset), radius=size)
        pygame.draw.circle(self.surf, color=self.icon_color, center=(size, size), radius=size)

        self.size = size

        # make sure image at center of circle
        # scale image to fit within block
        scaled_icon = pygame.transform.scale(self.img, (size*1.5, size*1.5))
        # center it onto main circle
        icon_rect = scaled_icon.get_rect(center=(size, size))
        # draw icon onto button surface directly
        self.surf.blit(scaled_icon, icon_rect)
        # define where button sits on main screen
        self.rect = self.surf.get_rect(center=position)

    def flash(self, duration=2.0, waves=3, color=(255, 200, 60), spread=0.9,
              loop=False, pulse_hz=0.5):
        """
        Added by Anson
        Date: 20/9/2026
        Purpose: golden ring that pulses outward in smooth waves.

        Starts a non-blocking animation. Call it once; the rings are rendered by
        draw(), so an existing game loop animates them without any other change.
        Calling flash() again restarts the animation from the beginning.

        :param duration:  seconds the animation runs. Ignored when loop=True.
        :param waves:     how many rings are in flight at once. More = denser.
        :param color:     ring RGB. Default is gold.
        :param spread:    how far rings travel, as a fraction of the radius.
                          0.9 means a ring ends 90% wider than the button.
        :param loop:      True = run until stop_flash() is called. Useful for
                          "this button is armed / awaiting selection".
        :param pulse_hz:  ring emissions per second. LOW = slow, calm ripple.
                          0.5 means each ring takes 2 s to travel outward.
                          Above ~1.5 it starts to look like a strobe.

        Note: rings expand OUTSIDE the button's own rect, so they are drawn onto
        the target surface rather than baked into self.surf. That keeps the cached
        button image untouched and avoids re-running update_layout() every frame.
        """
        self._flashing = True
        self._flash_start = pygame.time.get_ticks()
        self._flash_duration = max(0.001, float(duration))
        self._flash_waves = max(1, int(waves))
        self._flash_color = tuple(color[:3])
        self._flash_spread = max(0.05, float(spread))
        self._flash_loop = bool(loop)
        self._flash_pulse_hz = max(0.1, float(pulse_hz))
        return self

    def stop_flash(self):
        """Stop the animation immediately (needed to end a loop=True flash)."""
        self._flashing = False
        return self

    @property
    def is_flashing(self):
        return self._flashing

    def _draw_flash(self, surface):
        """Render the expanding rings. Called by draw(); not part of the API."""
        elapsed = (pygame.time.get_ticks() - self._flash_start) / 1000.0

        if not self._flash_loop and elapsed >= self._flash_duration:
            self._flashing = False
            return

        # Global fade-out over the final 30% of a non-looping flash, so the
        # animation ends gently instead of vanishing mid-ripple.
        envelope = 1.0
        if not self._flash_loop:
            progress = elapsed / self._flash_duration
            if progress > 0.7:
                envelope = max(0.0, 1.0 - (progress - 0.7) / 0.3)

        max_extra = self.size * self._flash_spread
        # Pad by 2 px so the outermost stroke is not clipped by the surface edge.
        pad = int(max_extra) + 2
        side = (self.size + pad) * 2

        # Reallocate only when the button size changed — allocating a Surface
        # every frame would cost more than the drawing itself.
        if self._flash_surf is None or self._flash_surf.get_width() != side:
            self._flash_surf = pygame.Surface((side, side), pygame.SRCALPHA)
        ring_surf = self._flash_surf
        ring_surf.fill((0, 0, 0, 0))

        centre = (side // 2, side // 2)
        r, g, b = self._flash_color

        for i in range(self._flash_waves):
            # Stagger the rings evenly through one period so they trail each other.
            phase = (elapsed * self._flash_pulse_hz - i / self._flash_waves) % 1.0

            # Sine ease-out: leaves the rim at a moderate speed and glides to a
            # stop, the way a ripple loses energy. Quadratic easing was too
            # abrupt at birth, which read as a "pop" rather than a wave.
            eased = math.sin(phase * math.pi / 2)
            radius = self.size + eased * max_extra

            # Fade with distance travelled. The ring holds its brightness a
            # little longer than linear, then falls away - closer to how a real
            # ripple stays visible mid-travel and dissolves at the edge.
            fade = (1.0 - phase) ** 1.6
            # Gentle fade-IN over the first 12% so rings emerge from the rim
            # instead of appearing abruptly at full strength.
            birth = min(1.0, phase / 0.12)
            alpha = int(120 * fade * birth * envelope)
            if alpha <= 2:
                continue
            width = max(1, int(round(3 * (1.0 - phase * 0.55))))

            pygame.draw.circle(ring_surf, (r, g, b, alpha), centre,
                               int(radius), width)

        # A soft inner halo hugging the rim. Breathes at the pulse rate (not
        # double it) and at low amplitude, so it reads as a glow rather than a
        # blink.
        halo = int(45 * (0.55 + 0.45 * math.sin(elapsed * self._flash_pulse_hz
                                                * math.pi)) * envelope)
        if halo > 2:
            pygame.draw.circle(ring_surf, (r, g, b, halo), centre,
                               self.size, 2)

        # Align the ring surface with the button's visible circle. self.rect is
        # taller than it is wide when is_3d, so centre on the circle, not the rect.
        cx = self.rect.left + self.size
        cy = self.rect.top + self.size
        surface.blit(ring_surf, ring_surf.get_rect(center=(cx, cy)))

    def draw(self, surface):
        # draw white circle onto surface
        surface.blit(self.surf, self.rect)
        # rings go on top so they are not hidden behind the button face
        if self._flashing:
            self._draw_flash(surface)

    def get_bound(self):
        #return tuple of surface boundary
        """
        eg Rect(270,487,80,90)
        return (left,right,top,bottom)
        """
        return (self.rect[0],self.rect[0]+self.rect[2],self.rect[1],self.rect[1]+self.rect[3])
    
    def check_within(self, pos):
        """
        Edited by Brian
        date: 13/7/2026
        Purpose: 
        """
        #cicrle area check
        center_x = self.rect.left + self.size
        center_y = self.rect.top + self.size

        #get distance between cursor and center of button by calculating hypotenuse
        distance = math.hypot(pos[0] - center_x, pos[1] - center_y)

        return distance <= self.size

    def click(self):
        if(self.func):
            self.func()
        else:
            self.sample_func(surface=pygame.display.get_surface())

    def sample_func(self,surface):
        width, height = pygame.display.get_window_size()
        surface.fill((50,50,50))
        overlay_img = self.img
        img_rect = overlay_img.get_rect(center=(width // 2, height // 2))
        surface.blit(overlay_img, img_rect)