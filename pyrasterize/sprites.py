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
    def __init__(self, scenegraph_root) -> None:
        self.sprites = dict()
        self.scenegraph_root = scenegraph_root

    def add(self,
            name : str,
            billboard : dict,
            bbox_radius : float):
        self.scenegraph_root["children"][name] = rasterizer.get_model_instance(billboard)
        self.sprites[name] = [self.scenegraph_root["children"][name], bbox_radius]

    def get_instance(self, name):
        return self.sprites[name][0]
    