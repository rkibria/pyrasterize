#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model file loading functions
"""

import os, pathlib

def get_model_from_obj_file(fname):
    """Return model loaded from a Wavefront .obj file"""
    with open(fname, encoding="utf-8") as file:
        content = file.readlines()
    content = [x.strip() for x in content]

    mesh = {"verts": [], "tris": [], "colors": []}
    cur_color = (200, 200, 200)
    mtl_colors = {}
    for line in content:
        if line.startswith("v "):
            tokens = line.split()
            mesh["verts"].append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
        if line.startswith("mtllib "):
            tokens = line.split()[1:]
            mtl_filename = tokens[0]
            mtl_filename = os.path.join(os.path.dirname(os.path.abspath(fname)), mtl_filename)
            with open(mtl_filename, encoding="utf-8") as file:
                mtl_content = file.readlines()
            mtl_content = [x.strip() for x in mtl_content]
            cur_mtl = None
            for mtl_line in mtl_content:
                if mtl_line.startswith("newmtl "):
                    cur_mtl = mtl_line.split()[1]
                elif mtl_line.startswith("Kd "):
                    tokens = mtl_line.split()[1:]
                    color = [int(float(tokens[i]) * 255) for i in range(3)]
                    mtl_colors[cur_mtl] = color
        elif line.startswith("usemtl "):
            mtl = line.split()[1]
            if mtl in mtl_colors:
                cur_color = mtl_colors[mtl]
        elif line.startswith("f "):
            indices = []
            tokens = line.split()[1:]
            for face_token in tokens:
                indices.append(int(face_token.split("/")[0]) - 1)
            if len(indices) == 3:
                mesh["tris"].append((indices[0], indices[1], indices[2]))
                mesh["colors"].append(cur_color)
            elif len(indices) >= 4:
                for i in range(len(indices) - 2):
                    mesh["tris"].append((indices[0], indices[i+1], indices[i+2]))
                    mesh["colors"].append(cur_color)
            else:
                print("? indices " + str(indices))
    # print(f"--- loaded {fname}: {len(mesh['verts'])} vertices, {len(mesh['tris'])} triangles")
    return mesh
