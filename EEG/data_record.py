"""
Author: Anson Li
Created on: 26/6/2026
Purpose: script for easier data recording
"""

import pygame
#import constants for easier access to key events
from pygame.locals import *

##game
pygame.init()
pygame.font.init()
clock = pygame.time.Clock()

win = pygame.display.set_mode((800,600))
width, height = win.get_size()

size = int(width*0.1)
center = (width/2, height/2)
my_font = pygame.font.SysFont('Comic Sans MS', size)

show = ""
count = 0
show_text = False
allow_input = True

#game loop
run = True
win.fill((0,0,0))

while run:
    current_time = pygame.time.get_ticks()
    for e in pygame.event.get():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            run = False
        elif e.type == KEYDOWN:
            if e.key != K_RETURN and e.unicode.isalnum() and allow_input:
                show += e.unicode
            elif e.key == K_RETURN and show and allow_input:
                allow_input = False
                count += 1
                print(f"Starting {count} cycle")
                show_text = True
                display_time = current_time + 2000
                
    if show_text and current_time >= display_time:
        print(show)
        print(f"Finished {count} cycle")
        show = ""
        show_text = False
        allow_input = True

    if show_text:
        text_surface = my_font.render(show, False, (255, 255, 255))
        textRect = text_surface.get_rect(center=center)
        win.blit(text_surface, textRect)
    else:
        win.fill((0, 0, 0))
        
    #update full display surface to screen
    pygame.display.flip()
    clock.tick(60)
