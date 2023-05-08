#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mesh generation, manipulation and information
"""

from . import rasterizer
from . import vecmat

import math

MESH_DEFAULT_COLOR = (200, 200, 200)

def scale_vertices(model, s_x, s_y, s_z):
    """
    Mulitply the coordinates of every vertex
    by the given constants
    """
    for v in model["verts"]:
        v[0] *= s_x
        v[1] *= s_y
        v[2] *= s_z

def get_test_texture_mesh(mip_textures):
    """centered square with texture"""
    return {
        "verts" : [
            [-0.5, -0.5, 0],
            [0.5, -0.5, 0],
            [0.5, 0.5, 0],
            [-0.5, 0.5, 0],
        ],
        "uv" : [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        "tris" : [
            (0, 1, 2),
            (0, 2, 3),
        ],
        "texture": mip_textures,
    }

def get_test_texture_cube_instance(mip_textures, gouraud=False, gouraud_iterations=1):
    inst = rasterizer.get_model_instance(None)
    front_mesh = get_test_texture_mesh(mip_textures)
    inst["children"]["front"] = rasterizer.get_model_instance(front_mesh, vecmat.get_transl_m4(0, 0, 0.5))
    inst["children"]["front"]["gouraud"] = gouraud
    inst["children"]["front"]["gouraud_max_iterations"] = gouraud_iterations

    inst["children"]["back"] = rasterizer.get_model_instance(front_mesh,
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, 0, -0.5), vecmat.get_rot_x_m4(vecmat.deg_to_rad(180))))
    inst["children"]["back"]["gouraud"] = gouraud
    inst["children"]["back"]["gouraud_max_iterations"] = gouraud_iterations

    inst["children"]["left"] = rasterizer.get_model_instance(front_mesh,
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(-0.5, 0, 0), vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90))))
    inst["children"]["left"]["gouraud"] = gouraud
    inst["children"]["left"]["gouraud_max_iterations"] = gouraud_iterations

    inst["children"]["right"] = rasterizer.get_model_instance(front_mesh,
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0.5, 0, 0), vecmat.get_rot_y_m4(vecmat.deg_to_rad(90))))
    inst["children"]["right"]["gouraud"] = gouraud
    inst["children"]["right"]["gouraud_max_iterations"] = gouraud_iterations

    inst["children"]["top"] = rasterizer.get_model_instance(front_mesh,
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, 0.5, 0), vecmat.get_rot_x_m4(vecmat.deg_to_rad(-90))))
    inst["children"]["top"]["gouraud"] = gouraud
    inst["children"]["top"]["gouraud_max_iterations"] = gouraud_iterations

    inst["children"]["bottom"] = rasterizer.get_model_instance(front_mesh,
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, -0.5, 0), vecmat.get_rot_x_m4(vecmat.deg_to_rad(90))))
    inst["children"]["bottom"]["gouraud"] = gouraud
    inst["children"]["bottom"]["gouraud_max_iterations"] = gouraud_iterations

    return inst

def get_test_triangle_mesh():
    """triangle to 1,1,0"""
    return {
        "verts" : [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        "uv" : [
            [0, 0],
            [1, 0],
            [1, 1],
        ],
        "texture": [ # 2x2 texture
            [(255, 0, 0), (0, 255, 0)],
            [(0, 0, 255), (255, 0, 255)],
        ],
        "tris" : [(0, 1, 2)],
        "colors" : [MESH_DEFAULT_COLOR]
    }

def get_cube_mesh(color=MESH_DEFAULT_COLOR):
    """Return a unit cube mesh model dictionary:
    'verts': vertices (float vec3s for point positions in local coordinates)
    'tris': triangles (int vec3s indexing the 3 vertices in 'verts' of triangle)
    if color is not None: 'colors': triangle colors (float vec3s of triangle RGB color)
    """
    model = {
        "verts" : [
            [ 0.5,  0.5, 0.5],  # front top right     0
            [ 0.5, -0.5, 0.5],  # front bottom right  1
            [-0.5, -0.5, 0.5],  # front bottom left   2
            [-0.5,  0.5, 0.5],  # front top left      3
            [ 0.5,  0.5, -0.5], # back top right      4
            [ 0.5, -0.5, -0.5], # back bottom right   5
            [-0.5, -0.5, -0.5], # back bottom left    6
            [-0.5,  0.5, -0.5]  # back top left       7
            ],
        "uv" : [
            [1, 1],
            [1, 0],
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 0],
            [0, 1],
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
        }
    if color is not None:
        model["colors"] = [[color[0], color[1], color[2]]] * 12
    return model

def get_block_instance(sx, sy, sz, front_divs, side_divs, top_divs, colors=(MESH_DEFAULT_COLOR, MESH_DEFAULT_COLOR)):
    """
    Return a block made of separate 2d rectangles
    """
    inst = rasterizer.get_model_instance(None)
    front_mesh = get_rect_mesh((sx, sy), front_divs, colors)
    inst["children"]["front"] = rasterizer.get_model_instance(front_mesh, vecmat.get_transl_m4(0, 0, sz/2))
    inst["children"]["back"] = rasterizer.get_model_instance(front_mesh,
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, 0, -sz/2), vecmat.get_rot_x_m4(vecmat.deg_to_rad(180))))
    side_mesh = get_rect_mesh((sz, sy), side_divs, colors)
    inst["children"]["left"] = rasterizer.get_model_instance(side_mesh,
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(-sx/2, 0, 0), vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90))))
    inst["children"]["right"] = rasterizer.get_model_instance(side_mesh,
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(sx/2, 0, 0), vecmat.get_rot_y_m4(vecmat.deg_to_rad(90))))
    # TODO top
    return inst

def get_rect_mesh(r_size, r_divs, colors=(MESH_DEFAULT_COLOR, MESH_DEFAULT_COLOR)):
    """
    Return 2D rectangle mesh of given size and subdivision with checkerboard coloring
    The rectangle is created in the x/y plane facing toward z, i.e.
    an observer near origin looking to -z would see a "wall"
    """
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
    """
    Return a sphere with r_divs divisons along the radius
    and l_divs divisions along the length from pole to pole
    """
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
        for r_i in range(r_divs):
            # divide surface arc from bottom to top into l_divs
            l_phi = l_phi_step * (l_i + 1)
            y_i = -radius * math.cos(l_phi)
            radius_i = (radius ** 2 - y_i ** 2) ** 0.5

            r_phi = r_phi_step * r_i
            x_i = radius_i * math.cos(r_phi)
            z_i = -radius_i * math.sin(r_phi)
            mesh["verts"].append((x_i, y_i, z_i))

    for l_i in range(l_divs - 2):
        for r_i in range(r_divs):
            bottom_v = 2 + l_i * r_divs + r_i
            next_bottom_v = bottom_v + 1
            top_v = bottom_v + r_divs
            next_top_v = top_v + 1
            if r_i == r_divs - 1:
                next_bottom_v = 2 + l_i * r_divs
                next_top_v = next_bottom_v + r_divs
            mesh["tris"].append((bottom_v, next_top_v, top_v))
            mesh["tris"].append((bottom_v, next_bottom_v, next_top_v))
            mesh["colors"].append(color)
            mesh["colors"].append(color)

    for i in range(r_divs): # bottom cap
        bottom_v = 2 + i
        next_bottom_v = bottom_v + 1
        if i == r_divs - 1:
            next_bottom_v = 2
        mesh["tris"].append((next_bottom_v, bottom_v, bottom_center_v))
        mesh["colors"].append(color)

    for i in range(r_divs): # top cap
        top_v = 2 + (l_divs - 2) * r_divs + i
        next_top_v = top_v + 1
        if i == r_divs - 1:
            next_top_v = 2 + (l_divs - 2) * r_divs
        mesh["tris"].append((top_v, next_top_v, top_center_v))
        mesh["colors"].append(color)

    return mesh

def get_cylinder_mesh(length, radius, r_divs, color=MESH_DEFAULT_COLOR,
    close_top=True, close_bottom=True,
    top_offset=0.0, bottom_offset=0.0):
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
    mesh["verts"].append((0, bottom_y - bottom_offset, 0))
    mesh["verts"].append((0, top_y + top_offset, 0))
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
