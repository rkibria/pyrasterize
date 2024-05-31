#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model file loading functions
"""

import os

def parse_obj_file(obj_lines : list, mtl_lines : list) -> dict:
    """
    Get model from Wavefront .obj/.mtl

    :param obj_lines: Stripped lines  with .obj contents
    :param mtl_lines: Optional, Stripped lines with .mtl contents
    :returns: Mesh dict
    """
    mesh = {"verts": [], "tris": [], "colors": [], "uv": []}
    cur_color = (200, 200, 200)
    vts = []
    mtl_colors = {}
    uvs = {}

    if mtl_lines is not None:
        cur_mtl = None
        for mtl_line in mtl_lines:
            if mtl_line.startswith("newmtl "):
                cur_mtl = mtl_line.split()[1]
            elif mtl_line.startswith("Kd "):
                tokens = mtl_line.split()[1:]
                color = [int(float(tokens[i]) * 255) for i in range(3)]
                mtl_colors[cur_mtl] = color

    for line in obj_lines:
        if line.startswith("v "):
            tokens = line.split()
            mesh["verts"].append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
        if line.startswith("vt "):
            tokens = line.split()
            vts.append((float(tokens[1]), float(tokens[2])))
        elif line.startswith("usemtl "):
            mtl = line.split()[1]
            if mtl in mtl_colors:
                cur_color = mtl_colors[mtl]
        elif line.startswith("f "):
            indices = []
            tokens = line.split()[1:]
            for face_token in tokens:
                token_parts = face_token.split("/")
                vert_i = int(token_parts[0]) - 1
                indices.append(vert_i)
                if len(token_parts) > 1:
                    uv_token = token_parts[1]
                    if len(uv_token):
                        uv_i = int(uv_token) - 1
                        uvs[vert_i] = vts[uv_i]
            if len(indices) == 3:
                mesh["tris"].append((indices[0], indices[1], indices[2]))
                mesh["colors"].append(cur_color)
            elif len(indices) >= 4:
                for i in range(len(indices) - 2):
                    mesh["tris"].append((indices[0], indices[i+1], indices[i+2]))
                    mesh["colors"].append(cur_color)
            else:
                print("? indices " + str(indices))
    if uvs:
        for i in range(len(uvs)):
            if i in uvs:
                mesh["uv"].append(uvs[i])
            else:
                mesh["uv"].append((0.0, 0.0))
    return mesh

def get_model_from_obj_file(fname : str) -> dict:
    """
    Load Wavefront .obj file, loads referenced .mtl file from same path if present

    :param fname: String with .obj contents
    :returns: Mesh dict
    """
    with open(fname, encoding="utf-8") as file:
        obj_str = file.readlines()
    obj_lines = [x.strip() for x in obj_str]
    mtl_lines = None

    for line in obj_lines:
        if line.startswith("mtllib "):
            tokens = line.split()[1:]
            mtl_filename = tokens[0]
            mtl_filename = os.path.join(os.path.dirname(os.path.abspath(fname)), mtl_filename)
            try:
                with open(mtl_filename, encoding="utf-8") as file:
                    mtl_content = file.readlines()
                mtl_lines = [x.strip() for x in mtl_content]
            except:
                pass

    return parse_obj_file(obj_lines, mtl_lines)
