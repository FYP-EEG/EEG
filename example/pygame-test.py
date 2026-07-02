"""
Author: Anson Li
Created on: 26/6/2026
Purpose: test script
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
class Sq(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        #size of square
        self.surf = pygame.Surface((25,25))
        #color of square (R,G,B)
        self.surf.fill((0,200,255))

##game
pygame.init()
win = pygame.display.set_mode((800,600), pygame.RESIZABLE) #game window 800x600
width, height = win.get_size()

size = int(width*0.05)
margin = int(width*0.0125)
spacing = (size*2) + margin

center_x = width/2
center_y = (height-size)*0.95

path = "assets/"#os.path.join(current_dir, 'assets')
rock = Button("rock", size, (center_x - spacing, center_y), os.path.join(path, "hand-fist-solid.png"))
paper = Button("paper", size, (center_x, center_y), os.path.join(path, "hand-solid.png"))
scissors = Button("scissors", size, (center_x + spacing, center_y), os.path.join(path, "hand-peace-solid.png"))

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
win.fill((50,50,50))
while run:
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
            mouse_pressed = pygame.mouse.get_pressed()
            #0 left 1 middle 2 right
            if mouse_pressed[0]:
                cursor = pygame.mouse.get_pos()
                cursorUpdate = True

    if cursorUpdate and rock.check_within(cursor):
        rock.click()
        cursorUpdate = False
    elif cursorUpdate and paper.check_within(cursor):
        paper.click()
        cursorUpdate = False
    elif cursorUpdate and scissors.check_within(cursor):
        scissors.click()
        cursorUpdate = False
    
    for button in buttons:
        button.draw(win)
    #update full display surface to screen
    pygame.display.flip()