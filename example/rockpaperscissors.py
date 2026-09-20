"""
Author: Anson Li
Created on: 26/6/2026
Purpose: test script for pygame
Edited on: 20/9/2026 - added flash() demo + func= callbacks
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
win = pygame.display.set_mode((800,600), pygame.RESIZABLE) #game window 800x600
width, height = win.get_size()

size = int(width*0.05)
margin = int(width*0.0125)
spacing = (size*2) + margin

center_x = width/2
center_y = (height-size)*0.95

path = "assets/" #os.path.join(current_dir, 'assets')

"""
Added by Anson
Date: 20/9/2026
Purpose: func= callbacks so you can SEE that clicking works.

Each callback records which button was hit. The main loop then draws that
button's icon in the middle of the screen.

Why not draw directly in the callback?
The old default (Button.sample_func) did exactly that - it called
surface.fill() and blitted an icon straight onto the display. Anything drawn
later in the same frame covered it, and anything drawn in the NEXT frame
covered it again. Worse, it wiped the flash rings the moment they appeared.

Callbacks should only change STATE. All drawing belongs in the one place that
runs every frame: the main loop.
"""
selected = {"button": None}          # which button was last clicked

def pick_rock():
    selected["button"] = rock
    print("clicked: rock")

def pick_paper():
    selected["button"] = paper
    print("clicked: paper")

def pick_scissors():
    selected["button"] = scissors
    print("clicked: scissors")

rock = Button(0, "rock", size, (center_x - spacing, center_y),
              os.path.join(path, "hand-fist-solid.png"), func=pick_rock)
paper = Button(1, "paper", size, (center_x, center_y),
               os.path.join(path, "hand-solid.png"), func=pick_paper)
scissors = Button(2, "scissors", size, (center_x + spacing, center_y),
                  os.path.join(path, "hand-peace-solid.png"), func=pick_scissors)

buttons = [rock, paper, scissors]
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
Purpose: frame clock. Without it the loop free-runs at thousands of FPS and a
2-second animation is over before you can see it.
"""
clock = pygame.time.Clock()

while run:
    """
    Edited by Anson
    Date: 20/9/2026
    Purpose: clear the screen at the START of every frame.

    This used to sit ABOVE the while loop, so the screen was cleared once and
    every frame painted on top of the last. Opaque circles hid that, but
    flash() draws translucent rings - stacking ~25 alpha layers saturates them
    into one solid gold blob instead of a ripple.
    """
    win.fill((50,50,50))

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
            rock.update_layout(size, (center_x - spacing, center_y))
            paper.update_layout(size, (center_x, center_y))
            scissors.update_layout(size, (center_x + spacing, center_y))

        elif e.type == pygame.MOUSEBUTTONDOWN:
            """
            Edited by Anson
            Date: 20/9/2026
            Purpose: use the event's own data instead of polling the mouse.

            This used to be:
                mouse_pressed = pygame.mouse.get_pressed()
                if mouse_pressed[0]: ...

            get_pressed() reports the mouse state RIGHT NOW, not the state at
            the moment the event fired. On a quick click the button is often
            already released by the time we poll, so it returned
            (False, False, False) and the click was dropped silently.
            e.button and e.pos travel with the event, so they are always right.
            """
            #e.button: 1 left, 2 middle, 3 right
            if e.button == 1:
                cursor = e.pos
                cursorUpdate = True

    """
    Edited by Anson
    Date: 20/9/2026
    Purpose: click() runs the func= callback, flash() starts the gold ripple.
    """
    if cursorUpdate and rock.check_within(cursor):
        rock.click()                              # runs pick_rock()
        rock.flash(duration=4.0, waves=2)         # gold ripple
        paper.stop_flash()
        scissors.stop_flash()
        cursorUpdate = False
    elif cursorUpdate and paper.check_within(cursor):
        paper.click()
        paper.flash(duration=4.0, waves=2)
        rock.stop_flash()
        scissors.stop_flash()
        cursorUpdate = False
    elif cursorUpdate and scissors.check_within(cursor):
        scissors.click()
        scissors.flash(duration=4.0, waves=2)
        rock.stop_flash()
        paper.stop_flash()
        cursorUpdate = False
    elif cursorUpdate:
        #clicked empty space - clear everything
        selected["button"] = None
        for button in buttons:
            button.stop_flash()
        cursorUpdate = False

    """
    Added by Anson
    Date: 20/9/2026
    Purpose: show the clicked button's icon in the middle of the screen.
    Drawn here, every frame, so nothing else can overwrite it.
    """
    if selected["button"] is not None:
        big = pygame.transform.scale(selected["button"].img, (160, 160))
        win.blit(big, big.get_rect(center=(width // 2, height // 2 - 40)))

    for button in buttons:
        button.draw(win)

    #update full display surface to screen
    pygame.display.flip()
    #cap at 60 FPS so the animation runs at a visible speed
    clock.tick(60)

pygame.quit()
