#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for camera frustum"""

import math
import vecmat

def perspective_projection(v_hom):
    """Projection v4->v2"""
    return (v_hom[0] / -v_hom[2], v_hom[1] / -v_hom[2])

def get_view_plane_from_fov(fov):
    """Return view plane distance"""
    return 1 / math.tan(vecmat.deg_to_rad(fov / 2))

def get_persp_m4(d, ar):
    """Return perspective transformation matrix"""
    return [d,    0.0,   0.0,  0.0,
            0.0,  d*ar,  0.0,  0.0,
            0.0,  0.0,   1.0,  0.0,
            0.0,  0.0,   1.0,  0.0]

def get_camera_m(cam):
    """Return matrix based on camera dict(rot: [x,y,z], pos: [x,y,z], fov: degs)"""
    cam_pos = cam["pos"]
    cam_rot = cam["rot"]
    camera_m = vecmat.get_transl_m4(-cam_pos[0], -cam_pos[1], -cam_pos[2])
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_z_m4(-cam_rot[2]), camera_m)
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_y_m4(-cam_rot[1]), camera_m)
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_x_m4(-cam_rot[0]), camera_m)
    return camera_m

def test_project_point():
    """???"""

    camera = { "pos": [0,0,0], "rot": [0,0,0], "fov": 90 }

    world_points = [
        [0, 0, -1, 1],
        [1, 0, -1, 1],
        [1, 0, -2, 1],
        [2, 0, -2, 1],
        [-1, 0, -1, 1],
        [-2, 0, -2, 1],
        ]

    view_plane = get_view_plane_from_fov(60)
    ar = 1
    persp_m = get_persp_m4(view_plane, ar)

    for v_world in world_points:
        proj_v = vecmat.vec4_mat4_mul(v_world, persp_m)
        persp_v = perspective_projection(proj_v)
        print(f"{v_world} -> {proj_v} -> {persp_v}")

