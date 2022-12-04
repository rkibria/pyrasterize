#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mesh generation, manipulation and information
"""

import math

MESH_DEFAULT_COLOR = (200, 200, 200)

def get_test_triangle_mesh():
    """triangle to 1,1,0"""
    return {
        "verts" : [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
        ],
        "tris" : [(0, 1, 2)],
        "colors" : [MESH_DEFAULT_COLOR]
    }

def get_cube_mesh(color=MESH_DEFAULT_COLOR):
    """Return a unit cube mesh model dictionary:
    'verts': vertices (float vec3s for point positions in local coordinates)
    'tris': triangles (int vec3s indexing the 3 vertices in 'verts' of triangle)
    'colors': triangle colors (float vec3s of triangle RGB color)
    """
    return {
        "verts" : [
            ( 0.5,  0.5, 0.5),  # front top right     0
            ( 0.5, -0.5, 0.5),  # front bottom right  1
            (-0.5, -0.5, 0.5),  # front bottom left   2
            (-0.5,  0.5, 0.5),  # front top left      3
            ( 0.5,  0.5, -0.5), # back top right      4
            ( 0.5, -0.5, -0.5), # back bottom right   5
            (-0.5, -0.5, -0.5), # back bottom left    6
            (-0.5,  0.5, -0.5)  # back top left       7
            ],
        "tris" : [ # CCW winding order
            (0, 3, 1), # front face
            (2, 1, 3), #
            (3, 7, 2), # left face
            (6, 2, 7), #
            (4, 0, 5), # right face
            (1, 5, 0), #
            (4, 7, 0), # top face
            (3, 0, 7), #
            (1, 2, 5), # bottom face
            (6, 5, 2), #
            (7, 4, 6), # back face
            (5, 6, 4)  #
            ],
        "colors": [[color[0], color[1], color[2]]] * 12
        }

def get_rect_mesh(r_size, r_divs, colors=(MESH_DEFAULT_COLOR, MESH_DEFAULT_COLOR)):
    """Return 2D rectangle mesh of given size and subdivision
    with checkerboard coloring"""
    mesh = { "verts": [], "tris": [], "colors": []}
    d_x,d_y = r_divs

    start_x = -r_size[0] / 2.0
    step_x = r_size[0] / d_x
    start_y = -r_size[1] / 2.0
    step_y = r_size[1] / d_y
    for i_y in range(d_y + 1):
        for i_x in range(d_x + 1):
            mesh["verts"].append((start_x + step_x * i_x, start_y + step_y * i_y, 0))

    for i_y in range(d_y):
        for i_x in range(d_x):
            u_l = i_x + i_y * (d_x + 1)
            mesh["tris"].append((u_l, u_l + 1, u_l + 1 + (d_x + 1)))
            mesh["tris"].append((u_l, u_l + 1 + (d_x + 1), u_l + (d_x + 1)))
            color = colors[0] if (i_x + i_y) % 2 == 0 else colors[1]
            mesh["colors"].append(color)
            mesh["colors"].append(color)
    return mesh

def get_sphere_mesh(radius, r_divs, l_divs, color=MESH_DEFAULT_COLOR):
    r_divs = max(3, r_divs)
    l_divs = max(2, l_divs)
    mesh = { "verts": [], "tris": [], "colors": []}
    bottom_y = -radius
    top_y = radius
    bottom_center_v = 0
    top_center_v = 1
    mesh["verts"].append((0, bottom_y, 0))
    mesh["verts"].append((0, top_y, 0))
    r_phi_step = 2 * math.pi / r_divs
    l_phi_step = math.pi / l_divs

    for l_i in range(l_divs - 1):
        l_phi = l_phi_step * (l_i + 1)
        for r_i in range(r_divs):
            radius_i = radius * math.sin(l_phi)

            r_phi = r_phi_step * r_i
            x_i = radius_i * math.cos(r_phi)
            y_i = -radius + (2 * radius / l_divs) * (l_i + 1)
            z_i = -radius_i * math.sin(r_phi)
            mesh["verts"].append((x_i, y_i, z_i))
    for i in range(r_divs):
        bottom_v = 2 + i
        next_bottom_v = bottom_v + 1
        if i == r_divs - 1:
            next_bottom_v = 2
        mesh["tris"].append((next_bottom_v, bottom_v, bottom_center_v))
        mesh["colors"].append(color)
    for i in range(r_divs):
        top_v = 2 + (l_divs - 2) * r_divs + i
        next_top_v = top_v + 1
        if i == r_divs - 1:
            next_top_v = 2 + (l_divs - 2) * r_divs
        mesh["tris"].append((top_v, next_top_v, top_center_v))
        mesh["colors"].append(color)

    return mesh

def get_cylinder_mesh(length, radius, r_divs, color=MESH_DEFAULT_COLOR,
    close_top=True, close_bottom=True):
    """
    Return a cylinder of requested length, radius and division count
    Caution: cylinder insides are not rendered if top/bottom missing
    Center of cylinder is at origin of model space, orientation is lengthwise
    along the y axis.
    """
    r_divs = max(3, r_divs)
    mesh = { "verts": [], "tris": [], "colors": []}
    bottom_y = -length/2
    top_y = length/2
    bottom_center_v = 0
    top_center_v = 1
    mesh["verts"].append((0, bottom_y, 0))
    mesh["verts"].append((0, top_y, 0))
    phi_step = 2 * math.pi / r_divs
    for i in range(r_divs): # wall verts
        phi = phi_step * i
        x_i = radius * math.cos(phi)
        z_i = -radius * math.sin(phi)
        mesh["verts"].append((x_i, bottom_y, z_i))
        mesh["verts"].append((x_i, top_y, z_i))
    for i in range(r_divs):
        bottom_v = 2 + i * 2
        top_v = bottom_v + 1
        next_bottom_v = top_v + 1
        next_top_v = next_bottom_v + 1
        if i == r_divs - 1:
            next_bottom_v = 2
            next_top_v = next_bottom_v + 1
        # cylinder wall
        mesh["tris"].append((bottom_v, next_top_v, top_v))
        mesh["tris"].append((bottom_v, next_bottom_v, next_top_v))
        mesh["colors"].append(color)
        mesh["colors"].append(color)
        if close_top:
            mesh["tris"].append((top_v, next_top_v, top_center_v))
            mesh["colors"].append(color)
        if close_bottom:
            mesh["tris"].append((next_bottom_v, bottom_v, bottom_center_v))
            mesh["colors"].append(color)
    return mesh

def get_model_centering_offset(model):
    """Get vec3 to center position (with translate matrix) of the model"""
    avg = [0, 0, 0]
    for v_3 in model["verts"]:
        for i in range(3):
            avg[i] += v_3[i]
    for i in range(3):
        avg[i] /= len(model["verts"])
        avg[i] *= -1
    return avg
