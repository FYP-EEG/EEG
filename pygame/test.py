import pygame
import os
#import constants for easier access to key events
from pygame.locals import *

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


class Button(pygame.sprite.Sprite):
    def __init__(self, size, position,icon):
        super().__init__()

        self.size = size

        path = os.path.join(current_dir, 'assets', icon)
        #size
        self.surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(self.surf, (255,255,255), center=(size, size), radius=size)

        self.img = pygame.image \
                            .load(path) \
                            .convert_alpha()
        self.img = pygame.transform.scale(self.img, (size*1.5,size*1.5))
        
        #define where button is placed on main screen
        self.rect = self.surf.get_rect(center=position)
        #calculate where to place icon
        self.img_rect = self.img.get_rect(center=self.rect.center)

    def update_layout(self, size, position):
        #surface basically a block to draw on
        self.surf = pygame.Surface((size*2,size*2), pygame.SRCALPHA)
        #draw white circle within block
        pygame.draw.circle(self.surf, (255,255,255), center=(size, size), radius=size)
        
        #make sure image at center of circle
        ##scale image to fit within block
        self.img = pygame.transform.scale(self.img, (size*1.5,size*1.5))
        ##get center of rectangle
        self.rect = self.surf.get_rect(center=position)
        ##get center of image
        self.img_rect = self.img.get_rect(center=self.rect.center)

    def draw(self, surface):
        #draw white circle onto surface
        surface.blit(self.surf, self.rect)
        #draw icon onto surface
        surface.blit(self.img, self.img_rect)


pygame.init()
win = pygame.display.set_mode((800,600), pygame.RESIZABLE) #game window 800x600
width, height = win.get_size()

size = int(width*0.05)
margin = int(width*0.0125)
spacing = (size*2) + margin

center_x = width/2


rock = Button(size, (center_x - spacing, height*0.9), "hand-fist-solid.png")
paper = Button(size, (center_x, height*0.9), "hand-solid.png")
scissors = Button(size, (center_x + spacing, height*0.9), "hand-peace-solid.png")

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