import pygame
import os
#import constants for easier access to key events
from pygame.locals import *
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

path = os.path.join(current_dir, 'assets')
rock = Button(size, (center_x - spacing, height*0.9), os.path.join(path, "hand-fist-solid.png"))
paper = Button(size, (center_x, height*0.9), os.path.join(path, "hand-solid.png"))
scissors = Button(size, (center_x + spacing, height*0.9), os.path.join(path, "hand-peace-solid.png"))

buttons = [rock, paper, scissors]

#game loop
run = True
while run:
    for e in pygame.event.get():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_BACKSPACE):
            run = False
        elif e.type == VIDEORESIZE:
            width, height = e.size
            size = int(width*0.05)
            margin = int(width*0.0125)
            spacing = (size*2) + margin
            center_x = width/2

            win = pygame.display.set_mode((width,height), pygame.RESIZABLE)
            rock.update_layout(size, (center_x - spacing, height*0.9))
            paper.update_layout(size, (center_x, height*0.9))
            scissors.update_layout(size, (center_x + spacing, height*0.9))

        win.fill((50,50,50))
        
        for button in buttons:
            button.draw(win)

        #update full display surface to screen
        pygame.display.flip()