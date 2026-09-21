"""
Author: Anson Li
Created on: 8/8/2026
Purpose: a ui with more buttons and a distraction of background scenary
Edited on: 20/9/2026 - real game action buttons, flat translucent style

A mock game HUD: inventory / message / shoot / reload / map.

Background: drop a gameplay screenshot at assets/background.png and it is used
automatically. Otherwise the screen is black. A busy background matters for a
real BCI test - a blank screen flatters any classifier - so a real capture is
worth adding before recording.
"""

import pygame
import os
#import constants for easier access to key events
from pygame.locals import *

import sys
from pathlib import Path
# 1. Get the directory of 'pygame-test.py' (D:\EEG\example)
script_dir = Path(__file__).resolve().parent
# 2. Go up one level to 'D:\EEG' and down into 'pygame_lib'
lib_path = script_dir.parent / "pygame_lib"
# 3. Add it to sys.path
sys.path.insert(0, str(lib_path))
# 4. Import the module
from Button import Button

"""
sprite: 2d obj displayed on screen
surface: canva for drawing
rect: rectangle obj for positioning and collision
"""
current_dir = os.path.dirname(os.path.abspath(__file__))

##game
pygame.init()
win = pygame.display.set_mode((1720,880), pygame.RESIZABLE) #game window 800x600
width, height = win.get_size()

size = int(width*0.03)
margin = int(width*0.0125)
spacing = (size*2) + margin

center_x = width/2
center_y = (height-size)*0.95

path = "assets/" #os.path.join(current_dir, 'assets')

"""
Added by Anson
Date: 20/9/2026
Purpose: flat translucent button style.

Button.update_layout() draws ONE circle using icon_color when is_3d=False, so
icon_color is really the face colour of the button. A 4-tuple works because the
surface is created with SRCALPHA, so the alpha channel is honoured.
"""
FACE = (170, 170, 175, 70)     # slight grey, mostly see-through
GLYPH = (245, 245, 250)        # icon tint - see tint_icon() below


def tint_icon(btn, color=GLYPH):
    """Recolour a button's icon without touching the PNG on disk.

    The downloaded icons are BLACK glyphs on a transparent background. Black on
    a dark translucent circle is almost invisible, so each glyph is lifted to a
    light colour.

    BLEND_RGB_MAX takes max() per RGB channel and leaves alpha alone, so the
    glyph turns light while the transparent surround stays fully transparent.
    A plain fill() would flood the whole square.

    update_layout() is then re-run so the tinted icon is blitted into the
    cached button surface.
    """
    btn.img.fill(color, special_flags=pygame.BLEND_RGB_MAX)
    btn.update_layout(btn.size, btn.position)
    return btn


"""
Added by Anson
Date: 20/9/2026
Purpose: show which action fired. Callbacks only set state - all drawing happens
in the main loop, so nothing can be painted over later in the same frame.
"""
selected = {"button": None, "at": 0}

def make_action(name):
    def action():
        print(f"action: {name}")
    return action


# ---------------------------------------------------------------- LAYOUT
"""
Added by Anson
Date: 20/9/2026
Purpose: ONE place to change button placement.

Each entry is (id, name, icon file, x, y) where x and y are FRACTIONS of the
window, so the layout survives a resize. Edit these to move buttons around.
"""
LAYOUT = [
    (0, "inventory", "inventory.png", 0.08, 0.4),
    (1, "message",   "message.png",   0.08, 0.88),
    (2, "shoot",     "shoot.png",     0.75, 0.7),
    (3, "reload",    "reload.png",    0.9, 0.4),
    (4, "map",       "map.png",       0.08, 0.15),
]

def build_buttons():
    made = []
    for bid, name, icon, fx, fy in LAYOUT:
        b = Button(bid, name, size, (width*fx, height*fy),
                   os.path.join(path, icon),
                   func=make_action(name),
                   icon_color=FACE,      # the translucent grey face
                   is_3d=False)          # flat, no drop shadow
        tint_icon(b)
        made.append(b)
    return made

inventory, message, shoot, reload, map = build_buttons()
buttons = [inventory, message, shoot, reload, map]

def relayout():
    """Reposition + resize every button after the window changes."""
    for b, (bid, name, icon, fx, fy) in zip(buttons, LAYOUT):
        b.update_layout(size, (width*fx, height*fy))

cursor = ""
cursorUpdate = False

#game loop
run = True
"""
for but in buttons:
    print(but.rect)
    eg Rect(270,487,80,90)
    left,top,width,height
    so 
    left bound = 270
    right bound = 350
    top bound = 487
    bottom bound = 487+90
"""

"""
Added by Anson
Date: 20/9/2026
Purpose: frame clock. Without it the loop free-runs at thousands of FPS and any
animation (including Button.flash) is over before it can be seen.
"""
clock = pygame.time.Clock()


"""
Added by Anson
Date: 20/9/2026
Purpose: background image slot.

Drop a gameplay screenshot (e.g. a PUBG frame) at assets/background.png and it
is used automatically, scaled to the window. If the file is absent the screen
stays black - no scenery is drawn.

Kept as a cached scaled copy so the rescale only happens on resize, not on
every frame.
"""
BACKGROUND_FILE = os.path.join(path, "background.png")
_bg_raw = None
_bg_scaled = None
_bg_size = None

if os.path.exists(BACKGROUND_FILE):
    try:
        _bg_raw = pygame.image.load(BACKGROUND_FILE).convert()
        print(f"background: {BACKGROUND_FILE}")
    except Exception as exc:
        print(f"background failed to load ({exc}) - using black")
else:
    print("no assets/background.png - using black background")


def draw_background(surface, w, h):
    """Blit the gameplay screenshot, or fill black when there isn't one."""
    global _bg_scaled, _bg_size
    if _bg_raw is None:
        surface.fill((0, 0, 0))
        return
    if _bg_size != (w, h):
        _bg_scaled = pygame.transform.smoothscale(_bg_raw, (w, h))
        _bg_size = (w, h)
    surface.blit(_bg_scaled, (0, 0))


while run:
    """
    Edited by Anson
    Date: 20/9/2026
    Purpose: redraw the background at the START of every frame.

    This used to be a single win.fill() above the loop, so every frame painted
    on top of the last. Translucent shapes then stack up and saturate instead of
    staying see-through.
    """
    draw_background(win, width, height)

    for e in pygame.event.get():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_BACKSPACE):
            run = False
        elif e.type == VIDEORESIZE:
            #get window size
            width, height = e.size#ie 800*600,height*0.9=540
            #adjust button size
            size = int(width*0.05)#40 for 800 width
            margin = int(width*0.0125)
            spacing = (size*2) + margin
            center_x = width/2
            center_y = (height-size)*0.95

            win = pygame.display.set_mode((width,height), pygame.RESIZABLE)
            relayout()

        elif e.type == pygame.MOUSEBUTTONDOWN:
            """
            Edited by Anson
            Date: 20/9/2026
            Purpose: use the event's own data instead of polling the mouse.

            pygame.mouse.get_pressed() reports the state RIGHT NOW, not at the
            moment the event fired. On a quick click the button is often already
            released by the time we poll, so it returned (False, False, False)
            and the click was dropped silently. e.button/e.pos travel with the
            event, so they are always correct.
            """
            #e.button: 1 left, 2 middle, 3 right
            if e.button == 1:
                cursor = e.pos
                cursorUpdate = True

    if cursorUpdate:
        for button in buttons:
            if button.check_within(cursor):
                button.click()                      # runs the func= callback
                button.flash(duration=2.0, waves=2) # gold ripple
                for other in buttons:
                    if other is not button:
                        other.stop_flash()
                selected["button"] = button
                selected["at"] = pygame.time.get_ticks()
                break
        cursorUpdate = False

    """
    Added by Anson
    Date: 20/9/2026
    Purpose: show the chosen action's icon at screen centre for 1.2 s.
    """
    if selected["button"] is not None:
        age = pygame.time.get_ticks() - selected["at"]
        if age < 1200:
            side = int(min(width, height) * 0.22)
            big = pygame.transform.smoothscale(selected["button"].img, (side, side))
            if age > 900:                       # fade the last 300 ms
                big = big.copy()
                big.set_alpha(int(255 * (1200 - age) / 300))
            win.blit(big, big.get_rect(center=(width//2, height//2 - 30)))
        else:
            selected["button"] = None

    for button in buttons:
        button.draw(win)

    #update full display surface to screen
    pygame.display.flip()
    #cap at 60 FPS so animations run at a visible speed
    clock.tick(60)

pygame.quit()
