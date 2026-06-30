import pygame
class Button(pygame.sprite.Sprite):
    def __init__(self, size, position,icon, background_color=(128,128,128), icon_color=(255,255,255)):
        super().__init__()

        #button size
        self.size = size
        self.icon_color = icon_color
        self.background_color = background_color
        self.offset = self.size//8
        self.position = position

        self.img = pygame.image.load(icon).convert_alpha()

        self.update_layout(self.size, self.position)
        """
        #surface size
        self.surf = pygame.Surface((size*2.5, size*2+self.offset), pygame.SRCALPHA)

        pygame.draw.circle(self.surf, color=background_color, center=(size,size+self.offset), radius=size)
        pygame.draw.circle(self.surf, color=self.icon_color, center=(size, size), radius=size)
        
        self.img = pygame.image \
                            .load(icon) \
                            .convert_alpha()
        self.img = pygame.transform.scale(self.img, (size*1.5,size*1.5))
        
        #define where button is placed on main screen
        self.rect = self.surf.get_rect(center=position)
        #calculate where to place icon
        self.img_rect = self.img.get_rect(center=self.rect.center)"""

    def update_layout(self, size, position):
        self.size = size
        self.position = position
        self.offset = self.size//8

        #surface basically a block to draw on
        self.surf = pygame.Surface((size*2,size*2+self.offset), pygame.SRCALPHA)
        #draw white circle within block
        pygame.draw.circle(self.surf, color=self.background_color, center=(size,size+self.offset), radius=size)
        pygame.draw.circle(self.surf, color=self.icon_color, center=(size, size), radius=size)

        #make sure image at center of circle
        ##scale image to fit within block
        scaled_icon = pygame.transform.scale(self.img, (size*1.5,size*1.5))
        ##center it onto main circle
        icon_rect = scaled_icon.get_rect(center=(self.size, self.size))
        ##draw icon onto button surface directly
        self.surf.blit(scaled_icon, icon_rect)
        #define where button sits on main screen
        self.rect = self.surf.get_rect(center=position)

    def draw(self, surface):
        #draw white circle onto surface
        surface.blit(self.surf, self.rect)