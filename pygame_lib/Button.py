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
        scaled_icon = pygame.transform.scale(self.img, (int(size * 1.5), int(size * 1.5)))
        # center it onto main circle
        icon_rect = scaled_icon.get_rect(center=(size, size))
        # draw icon onto button surface directly
        self.surf.blit(scaled_icon, icon_rect)
        # define where button sits on main screen
        self.rect = self.surf.get_rect(center=position)

    def draw(self, surface):
        # draw white circle onto surface
        surface.blit(self.surf, self.rect)

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