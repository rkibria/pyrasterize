#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for camera frustum"""

import math
import vecmat

def perspective_projection(v_hom, view_plane):
    """Projection"""
    return (view_plane * v_hom[0] / -v_hom[2], view_plane * v_hom[1] / -v_hom[2])

def get_view_plane_from_fov(fov):
    """Return view plane distance"""
    return 1 / math.tan(vecmat.deg_to_rad(fov / 2))

def test_project_point():
    """???"""
    world_points = [
        [0, 0, -1, 1],
        [1, 0, -1, 1],
        [2, 0, -2, 1],
        [-1, 0, -1, 1],
        [-2, 0, -2, 1],
        ]

    view_plane = get_view_plane_from_fov(90)
    for v_hom in world_points:
        proj_point = perspective_projection(v_hom, view_plane)
        print(v_hom, proj_point)
