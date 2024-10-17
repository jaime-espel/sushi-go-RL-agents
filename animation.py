import pygame
from Point import *
from settings import WIDTH

class ScaleAnimation:
    def __init__(self, scale_min: float, scale_max: float, scale_increment: float, 
                 pos_ref: str, pos: Point, screen: pygame.Surface):
        # Scale parameters
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_increment = scale_increment
        # Current scale
        self.scale_current = scale_min
        # Position
        self.pos_ref = pos_ref # Reference point (screen points): 'topright_ref', else: normal references
        self.pos = pos
        # State
        self.is_scaling_up = True
        self.is_active = False
        # Surface to render in
        self.screen = screen

    def reset(self):
        self.scale_current = self.scale_min
        self.is_scaling_up = True
        self.is_active = True

    def animate(self, img: pygame.Surface):
        if self.is_active:
            if self.is_scaling_up:
                # scale
                P = pygame.transform.rotozoom(img, 0, self.scale_current)
                # determine center pos
                if self.pos_ref == 'topright_ref': P_rect = P.get_rect(center=(WIDTH - self.pos.x , self.pos.y))
                else: P_rect = P.get_rect(center=self.pos.get_point())
                # render
                self.screen.blit(P, P_rect)
                # increment scale
                self.scale_current += self.scale_increment
                # check scale size
                if self.scale_current > self.scale_max:
                    self.is_scaling_up = False
            else:
                # scale
                P = pygame.transform.rotozoom(img, 0, self.scale_current)
                # determine center pos
                if self.pos_ref == 'topright_ref': P_rect = P.get_rect(center=(WIDTH - self.pos.x , self.pos.y))
                else: P_rect = P.get_rect(center=self.pos.get_point())
                # render
                self.screen.blit(P, P_rect)
                # decrement scale
                self.scale_current -= self.scale_increment
                # check scale size
                if self.scale_current < self.scale_min:
                    self.is_active = False
 
            