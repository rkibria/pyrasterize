#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for camera frustum"""

import math
import vecmat

def test_project_point():
    """???"""
    world_points = [
        [0, 0, -1, 1],
        [1, 0, -1, 1],
        [2, 0, -2, 1],
        [-1, 0, -1, 1],
        [-2, 0, -2, 1],
        ]
    fov = 90
    view_plane = 1 / math.tan(vecmat.deg_to_rad(fov / 2))

    for v_hom in world_points:
        proj_point = (view_plane * v_hom[0] / -v_hom[2], view_plane * v_hom[1] / -v_hom[2])
        print(v_hom, proj_point)

