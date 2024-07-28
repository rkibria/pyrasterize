#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Moving and collision checking of
sprites (enemies, projectiles, items)
displayed as billboards and with
spherical bounding boxes in the world.
"""

from . import vecmat
from . import rasterizer

class Sprites:
    MOVE_MODE_NONE = 0
    MOVE_MODE_LINEAR = 1 # Data: [direction vec4, per move cb]

    def __init__(self, scenegraph_root) -> None:
        # Content: [sg instance, bbox radius, [move mode, move data]]
        self.sprites = dict()
        self.scenegraph_root = scenegraph_root

    def add(self,
            name : str,
            billboard : dict,
            bbox_radius : float) -> None:
        self.scenegraph_root["children"][name] = rasterizer.get_model_instance(billboard)
        self.sprites[name] = [self.scenegraph_root["children"][name],
                              bbox_radius,
                              [self.MOVE_MODE_NONE, None]]

    def get_instance(self, name) -> dict:
        return self.sprites[name][0]

    def get_move_setting(self, name) -> list:
        return self.sprites[name][2]

    def update(self):
        for _,sprite in self.sprites.items():
            sprite_inst = sprite[0]
            if sprite_inst["enabled"]:
                move_mode,move_data = sprite[2]
                if move_mode == self.MOVE_MODE_LINEAR:
                    move_dir,move_cb = move_data
                    billboard_translate = sprite_inst["model"]["translate"]
                    new_translate = billboard_translate.copy()
                    new_translate[0] += move_dir[0]
                    new_translate[1] += move_dir[1]
                    new_translate[2] += move_dir[2]
                    move_cb(sprite_inst, new_translate)

    def move_linear_from_cam(self, name, camera, move_cb) -> None:
        """Move a sprite starting from camera position towards viewing direction"""
        inst = self.get_instance(name)
        inst["enabled"] = True

        dir = vecmat.vec4_mat4_mul((0, 0, -1, 0), vecmat.get_rot_x_m4(camera["rot"][0]))
        dir = vecmat.vec4_mat4_mul(dir, vecmat.get_rot_y_m4(camera["rot"][1]))

        trans = inst["model"]["translate"]
        for i in range(3):
            trans[i] = camera["pos"][i]

        move_setting = self.get_move_setting(name)
        move_setting[0] = self.MOVE_MODE_LINEAR
        move_setting[1] = [dir, move_cb]
