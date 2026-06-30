import pygame
class Button(pygame.sprite.Sprite):
    def __init__(self, size, position,icon):
        super().__init__()

        #button size
        self.size = size

        #surface size
        self.surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(self.surf, (255,255,255), center=(size, size), radius=size)

        self.img = pygame.image \
                            .load(icon) \
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