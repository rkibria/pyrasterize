#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common code for FPS-like controls
"""

import math
import time

import pygame as pg

from . import vecmat
from . import uiwndmgr
from . import uiwidget

class FpsControls:
    MODE_GAME = 0
    MODE_MENU = 1

    LABEL_COLOR = (255, 255, 255)
    UNDER_LABEL_COLOR = (32, 32, 32)

    def __init__(self, RASTER_SCR_SIZE, camera, render_settings, clock : pg.time.Clock) -> None:
        self.time = time.perf_counter()

        self.on_mouse_button_down_cb = None
 
        self.RASTER_SCR_SIZE = RASTER_SCR_SIZE
        self.RASTER_SCR_WIDTH,self.RASTER_SCR_HEIGHT = RASTER_SCR_SIZE

        self.mode = self.MODE_GAME

        self.DELTA_ROT = vecmat.deg_to_rad(3)
        self.DELTA_POS = 0.2

        self.last_cam_pos = []

        self.camera = camera
        # xyz delta relative to camera direction / xyz camera rotation
        self.move_dir = [0, 0, 0,
                         0, 0, 0]

        self.render_settings = render_settings
        self.clock = clock

        # key: (index, value)
        self.key_moves = {
            # WASD
            pg.K_w: (2, -1),
            pg.K_s: (2, 1),
            pg.K_a: (0, -1),
            pg.K_d: (0, 1),
            # Camera rotation
            pg.K_LEFT: (3, 1),
            pg.K_RIGHT: (3, -1),
            pg.K_UP: (4, 1),
            pg.K_DOWN: (4, -1),
        }

        self.first_mouse_move = True

        self.cross_size = 20
        cross_width = 2
        rgb_cross = (255, 255, 255, 100)
        self.cross_surface = pg.Surface((2 * self.cross_size, 2 * self.cross_size))
        pg.draw.rect(self.cross_surface, rgb_cross, (self.cross_size - cross_width,
                                                         0,
                                                         cross_width * 2,
                                                         self.cross_size * 2))
        pg.draw.rect(self.cross_surface, rgb_cross, (0, self.cross_size - cross_width,
                                                         self.cross_size * 2, cross_width * 2))
        pg.draw.rect(self.cross_surface, (0, 0, 0), (self.cross_size - 2 * cross_width,
                                                         self.cross_size - 2 * cross_width,
                                                         cross_width * 4,
                                                         cross_width * 4))

        self.wmgr_mode_menu = uiwndmgr.WindowManager(self.RASTER_SCR_SIZE)
        self.wmgr_mode_game = uiwndmgr.WindowManager(self.RASTER_SCR_SIZE)
        self.wmgr_mode_game.cursor_enable = False

        self.fps_label = uiwidget.Label("fps", "---", 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR, pos=(0, 0))

        # Game mode HUD
        hud_layout = uiwidget.Widget("hud", (20, 20))
        hud_layout.add_child(self.fps_label)
        hud_layout.add_child(uiwidget.Label("info", "F1: menu", 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR, pos=(0, 14)))
        self.wmgr_mode_game.add_widget(hud_layout)

        # F1 menu setup
        settings_layout = uiwidget.Widget("settings", (20, 20))
        self.wmgr_mode_menu.add_widget(settings_layout)

        settings_layout.add_child(self.fps_label)

        def fog_dist_to_progress(dist):
            return abs(dist) / 30
        def fog_progress_to_dist(prog):
            return -30.0 * prog
        def fog_dist_text():
            return f"Fog distance: {round(render_settings['fog_distance'], 1)}"
        def color_comp_to_progress(comp):
            return comp / 255.0
        def color_comp_progress_to_int(progress):
            return int(progress * 255)
        def fog_rgb_text():
            color = [int(v) for v in render_settings["fog_color"]]
            return f"{color[0]}, {color[1]}, {color[2]}"

        def add_fog_widgets(pos : pg.Vector2):
            fog_dist_layout = uiwidget.Widget("fog_dist_layout", pos)
            settings_layout.add_child(fog_dist_layout)
            fog_dist_label = uiwidget.Label("fog_distance", fog_dist_text(), 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR, pos=(0, 0))
            fog_dist_layout.add_child(fog_dist_label)
            fog_distance_slider = uiwidget.HorizontalSlider("fog_distance_slider", self.wmgr_mode_menu, "barYellow", "blue", (120, 3), (150, 6))
            fog_distance_slider.progress = fog_dist_to_progress(self.render_settings["fog_distance"])
            def on_fog_slider_changed(progress):
                self.render_settings["fog_distance"] = fog_progress_to_dist(progress)
                fog_dist_label.set_text(fog_dist_text(), 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR)
            fog_distance_slider.on_change_cb = on_fog_slider_changed
            fog_dist_layout.add_child(fog_distance_slider)

            fog_color_layout = uiwidget.Widget("fog_color_layout", pos + pg.Vector2(0, 20))
            settings_layout.add_child(fog_color_layout)
            fog_color_label = uiwidget.Label("fog_color", "Fog color", 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR, pos=(0, 0))
            fog_color_layout.add_child(fog_color_label)
            fog_rgb_label = uiwidget.Label("fog_rgb", fog_rgb_text(), 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR, pos=(0, 14))
            fog_color_layout.add_child(fog_rgb_label)
            def update_fog_rgb_label():
                fog_rgb_label.set_text(fog_rgb_text(), 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR)

            fog_red_slider = uiwidget.HorizontalSlider("fog_red_slider", self.wmgr_mode_menu, "barRed", "beige", (120, 3), (150, 6))
            def on_fog_red_changed(progress):
                self.render_settings["fog_color"][0] = color_comp_progress_to_int(progress)
                update_fog_rgb_label()
            fog_red_slider.on_change_cb = on_fog_red_changed
            fog_color_layout.add_child(fog_red_slider)
            fog_green_slider = uiwidget.HorizontalSlider("fog_green_slider", self.wmgr_mode_menu, "barGreen", "beige", (120, 3+8), (150, 6))
            def on_fog_green_changed(progress):
                self.render_settings["fog_color"][1] = color_comp_progress_to_int(progress)
                update_fog_rgb_label()
            fog_green_slider.on_change_cb = on_fog_green_changed
            fog_color_layout.add_child(fog_green_slider)
            fog_blue_slider = uiwidget.HorizontalSlider("fog_blue_slider", self.wmgr_mode_menu, "barBlue", "beige", (120, 3+2*8), (150, 6))
            def on_fog_blue_changed(progress):
                self.render_settings["fog_color"][2] = color_comp_progress_to_int(progress)
                update_fog_rgb_label()
            fog_blue_slider.on_change_cb = on_fog_blue_changed
            fog_color_layout.add_child(fog_blue_slider)

        def nearclipdist_to_progress(dist):
            return abs(dist) / 10
        def nearclipdist_progress_to_dist(prog):
            return min(-0.1, -10.0 * prog)
        def nearclipdist_text():
            return f"Near clip: {round(render_settings['near_clip'], 1)}"

        def farclipdist_to_progress(dist):
            return abs(dist) / 100
        def farclipdist_progress_to_dist(prog):
            return min(-0.1, -100.0 * prog)
        def farclipdist_text():
            return f"Far clip: {round(render_settings['far_clip'], 1)}"

        def add_clipdist_widgets(pos : pg.Vector2):
            clipdist_layout = uiwidget.Widget("clipdist_layout", pos)
            settings_layout.add_child(clipdist_layout)
            nearclipdist_label = uiwidget.Label("nearclipdist", nearclipdist_text(), 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR, pos=(0, 0))
            clipdist_layout.add_child(nearclipdist_label)
            nearclipdist_slider = uiwidget.HorizontalSlider("nearclipdist_slider", self.wmgr_mode_menu, "barYellow", "blue", (120, 3), (150, 6))
            nearclipdist_slider.progress = nearclipdist_to_progress(self.render_settings["near_clip"])
            def on_nearclipdist_slider_changed(progress):
                self.render_settings["near_clip"] = nearclipdist_progress_to_dist(progress)
                nearclipdist_label.set_text(nearclipdist_text(), 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR)
            nearclipdist_slider.on_change_cb = on_nearclipdist_slider_changed
            clipdist_layout.add_child(nearclipdist_slider)

            farclipdist_label = uiwidget.Label("farclipdist", farclipdist_text(), 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR, pos=(0, 20))
            clipdist_layout.add_child(farclipdist_label)
            farclipdist_slider = uiwidget.HorizontalSlider("farclipdist_slider", self.wmgr_mode_menu, "barYellow", "blue", (120, 23), (150, 6))
            farclipdist_slider.progress = farclipdist_to_progress(self.render_settings["far_clip"])
            def on_farclipdist_slider_changed(progress):
                self.render_settings["far_clip"] = farclipdist_progress_to_dist(progress)
                farclipdist_label.set_text(farclipdist_text(), 16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR)
            farclipdist_slider.on_change_cb = on_farclipdist_slider_changed
            clipdist_layout.add_child(farclipdist_slider)

        # Add subwidgets
        add_clipdist_widgets((0, 25))
        add_fog_widgets((0, 70))

        self.update_fps_label()

    def update_fps_label(self):
        pos = [round(p, 1) for p in self.camera['pos']]
        rot = [round(vecmat.rad_to_deg(p), 1) for p in self.camera['rot']]
        self.fps_label.set_text(f"pos {pos} rot {rot} - {round(self.clock.get_fps(), 1)} fps",
                                16, font_color=self.LABEL_COLOR, under_color=self.UNDER_LABEL_COLOR)

    def on_mouse_movement(self, x, y):
        """Handle mouse movement"""
        if self.mode == self.MODE_GAME:
            rot = self.camera["rot"]
            rot[0] -= vecmat.deg_to_rad(y * 0.2)
            rot[1] -= vecmat.deg_to_rad(x * 0.2)
            # limit up/down rotation around x-axis to straight up/down at most
            rot[0] = min(math.pi/2, max(-math.pi/2, rot[0]))
        else:
            pass

    def on_key_down(self, key):
        """"""
        if key == pg.K_F1:
            self.mode = self.MODE_MENU if self.mode == self.MODE_GAME else self.MODE_GAME
        else:
            if self.mode == self.MODE_GAME:
                if key in self.key_moves:
                    index, value = self.key_moves[key]
                    self.move_dir[index] = value
            else:
                pass

    def on_key_up(self, key):
        """"""
        if self.mode == self.MODE_GAME:
            if key in self.key_moves:
                index, _ = self.key_moves[key]
                self.move_dir[index] = 0
        else:
            pass

    def on_event(self, event):
        if self.mode == self.MODE_GAME:
            if event.type == pg.KEYDOWN:
                self.on_key_down(event.key)
            elif event.type == pg.KEYUP:
                self.on_key_up(event.key)
            elif event.type == pg.MOUSEMOTION:
                mouse_position = pg.mouse.get_rel()
                if self.first_mouse_move:
                    self.first_mouse_move = False
                else:
                    self.on_mouse_movement(mouse_position[0], mouse_position[1])
            elif event.type == pg.MOUSEBUTTONDOWN:
                if self.on_mouse_button_down_cb is not None:
                    self.on_mouse_button_down_cb(event)
        else:
            if event.type == pg.KEYDOWN:
                self.on_key_down(event.key)
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.wmgr_mode_menu.on_mouse_button_down(event)
            elif event.type == pg.MOUSEMOTION:
                mouse_position = pg.mouse.get_pos()
                self.wmgr_mode_menu.set_cursor_pos(mouse_position)

    def do_movement(self):
        """"""
        if self.mode != self.MODE_GAME:
            return

        if not any(self.move_dir):
            return

        cam_rot = self.camera["rot"]
        cam_pos = self.camera["pos"]

        # Forward movement:
        # add vector pointing in the direction of the camera to pos.
        #
        # The camera direction for movement is in the x/z plane (y=0).
        # The relevant rotation axis is Y
        cam_rot_y = cam_rot[1]
        cam_v_forward = [math.sin(cam_rot_y), 0, math.cos(cam_rot_y)]
        speed = self.move_dir[2]
        total_movement = [0.0, 0.0, 0.0]
        total_movement[0] += cam_v_forward[0] * speed
        total_movement[2] += cam_v_forward[2] * speed
        # strafing:
        # add vector perpendicular to camera direction to pos.
        cam_v_right = [-cam_v_forward[2], 0, cam_v_forward[0]] # 90 deg rotate: (-y, x)
        speed = self.move_dir[0]
        total_movement[0] -= cam_v_right[0] * speed
        total_movement[2] -= cam_v_right[2] * speed
        # normalize the movement vector so moving diagonally isn't faster than straight moves
        total_movement = vecmat.norm_vec3(total_movement)
        new_pos = [cam_pos[0] + total_movement[0] * self.DELTA_POS,
                   cam_pos[2] + total_movement[2] * self.DELTA_POS]
        self.last_cam_pos = [v for v in cam_pos]
        cam_pos[0] = new_pos[0]
        cam_pos[2] = new_pos[1]

        # Camera rotation
        cam_rot[0] += self.move_dir[4] * self.DELTA_ROT
        cam_rot[1] += self.move_dir[3] * self.DELTA_ROT
        cam_rot[0] = min(math.pi/2, max(-math.pi/2, cam_rot[0]))

    def draw(self, surface):
        elapsed_time = time.perf_counter() - self.time
        if elapsed_time > 0.5:
            self.update_fps_label()
            self.time = time.perf_counter()
        if self.mode == self.MODE_GAME:
            surface.blit(self.cross_surface,
                        (self.RASTER_SCR_WIDTH // 2 - self.cross_size,
                        self.RASTER_SCR_HEIGHT // 2 - self.cross_size),
                        special_flags=pg.BLEND_RGBA_ADD)
            self.wmgr_mode_game.draw(surface)
        else:
            self.wmgr_mode_menu.draw(surface)
