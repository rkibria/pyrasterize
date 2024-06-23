#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common code for FPS-like controls
"""

import math
import pygame

from . import vecmat

class FpsControls:
    def __init__(self, RASTER_SCR_SIZE, camera) -> None:
        self.RASTER_SCR_SIZE = RASTER_SCR_SIZE
        self.RASTER_SCR_WIDTH,self.RASTER_SCR_HEIGHT = RASTER_SCR_SIZE

        self.DELTA_ROT = vecmat.deg_to_rad(3)

        self.camera = camera
        # xyz delta relative to camera direction
        self.move_dir = [0, 0, 0]
        # horizontal/vertical rotation direction
        self.rot_dir = [0, 0]

        self.first_mouse_move = True

        self.cross_size = 20
        cross_width = 2
        rgb_cross = (255, 255, 255, 100)
        self.cross_surface = pygame.Surface((2 * self.cross_size, 2 * self.cross_size))
        pygame.draw.rect(self.cross_surface, rgb_cross, (self.cross_size - cross_width,
                                                         0,
                                                         cross_width * 2,
                                                         self.cross_size * 2))
        pygame.draw.rect(self.cross_surface, rgb_cross, (0, self.cross_size - cross_width,
                                                         self.cross_size * 2, cross_width * 2))
        pygame.draw.rect(self.cross_surface, (0, 0, 0), (self.cross_size - 2 * cross_width,
                                                         self.cross_size - 2 * cross_width,
                                                         cross_width * 4,
                                                         cross_width * 4))

        self.textblock_fps = None

    def on_mouse_movement(self, x, y):
        """Handle mouse movement"""
        rot = self.camera["rot"]
        rot[0] -= vecmat.deg_to_rad(y * 0.2)
        rot[1] -= vecmat.deg_to_rad(x * 0.2)
        # limit up/down rotation around x-axis to straight up/down at most
        rot[0] = min(math.pi/2, max(-math.pi/2, rot[0]))

    def on_key_down(self, key):
        """"""
        if key == pygame.K_w:
            self.move_dir[2] = -1
        elif key == pygame.K_s:
            self.move_dir[2] = 1
        elif key == pygame.K_a:
            self.move_dir[0] = -1
        elif key == pygame.K_d:
            self.move_dir[0] = 1
        elif key == pygame.K_LEFT:
            self.rot_dir[0] = 1
        elif key == pygame.K_RIGHT:
            self.rot_dir[0] = -1
        elif key == pygame.K_UP:
            self.rot_dir[1] = 1
        elif key == pygame.K_DOWN:
            self.rot_dir[1] = -1

    def on_key_up(self, key):
        """"""
        if key == pygame.K_w or key == pygame.K_s:
            self.move_dir[2] = 0
        elif key == pygame.K_a or key == pygame.K_d:
            self.move_dir[0] = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:
            self.rot_dir[0] = 0
        elif key == pygame.K_UP or key == pygame.K_DOWN:
            self.rot_dir[1] = 0

    def on_event(self, event):
      if event.type == pygame.KEYDOWN:
          self.on_key_down(event.key)
      elif event.type == pygame.KEYUP:
          self.on_key_up(event.key)
      elif event.type == pygame.MOUSEMOTION:
          mouse_position = pygame.mouse.get_rel()
          if self.first_mouse_move:
              self.first_mouse_move = False
          else:
              self.on_mouse_movement(mouse_position[0], mouse_position[1])

    def do_movement(self):
        """"""
        rot = self.camera["rot"]
        rot[0] += self.rot_dir[1] * self.DELTA_ROT
        rot[1] += self.rot_dir[0] * self.DELTA_ROT

        if self.move_dir == [0, 0, 0]:
            return
        # forward movement:
        # add vector pointing in the direction of the camera to pos.
        #
        # The camera direction for movement is in the x/z plane (y=0).
        # The relevant rotation axis is Y
        cam_rot_y = self.camera["rot"][1]
        move_scale = 0.1
        cam_v_forward = [move_scale * math.sin(cam_rot_y), 0, move_scale * math.cos(cam_rot_y)]
        cam_pos = self.camera["pos"]
        speed = self.move_dir[2]
        cam_pos[0] += cam_v_forward[0] * speed
        cam_pos[2] += cam_v_forward[2] * speed
        # strafing:
        # add vector perpendicular to camera direction to pos.
        cam_v_right = [-cam_v_forward[2], 0, cam_v_forward[0]] # 90 deg rotate: (-y, x)
        speed = self.move_dir[0]
        cam_pos[0] -= cam_v_right[0] * speed
        cam_pos[2] -= cam_v_right[2] * speed

    def draw(self, surface):
        surface.blit(self.cross_surface,
                     (self.RASTER_SCR_WIDTH // 2 - self.cross_size,
                      self.RASTER_SCR_HEIGHT // 2 - self.cross_size),
                      special_flags=pygame.BLEND_RGBA_ADD)
        if self.textblock_fps:
            surface.blit(self.textblock_fps, (30, 30))

    def update_hud(self, font, clock, text_col=(200, 200, 230)):
        pos = [round(p, 2) for p in self.camera['pos']]
        rot = [round(vecmat.rad_to_deg(p), 2) for p in self.camera['rot']]
        self.textblock_fps = font.render(f"pos: {pos} - rot: {rot} - {round(clock.get_fps(), 1)} fps",
                                         True, text_col)
