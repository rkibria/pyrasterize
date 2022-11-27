#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D rasterizer engine
"""

import pygame

from . import vecmat

def get_model_instance(model, preproc_m4=None, xform_m4=None, children=None):
    """Return model instance
    These are the key values in a scene graph {name_1: instance_1, ...} dictionary
    Optional keys:
    * wireframe (boolean): draw this as wireframe instead of filled polygons
    * bound_sph_r (float): radius of the bounding sphere of this model, can check for e.g. selection
    * noCulling (boolean): don't cull back face triangles
    """
    if preproc_m4 is None:
        preproc_m4 = vecmat.get_unit_m4()
    if xform_m4 is None:
        xform_m4 = vecmat.get_unit_m4()
    if children is None:
        children = {}
    return {
        "enabled" : True,
        "model": model,
        "preproc_m4": preproc_m4,
        "xform_m4": xform_m4,
        "children": children
        }

def render(surface, screen_area, scene_graph, camera_m, persp_m, lighting):
    """Render the scene graph
    screen_area is (x,y,w,h) inside the surface
    """
    scr_origin_x = screen_area[0] + screen_area[2] / 2
    scr_origin_y = screen_area[1] + screen_area[3] / 2

    ambient = lighting["ambient"]
    diffuse = lighting["diffuse"]

    def get_proj_light_dir():
        light_dir_vec3 = lighting["lightDir"]
        light_dir_vec4 = (light_dir_vec3[0], light_dir_vec3[1], light_dir_vec3[2], 0)
        return vecmat.norm_vec3(vecmat.vec4_mat4_mul(light_dir_vec4, camera_m)[0:3])

    proj_light_dir = get_proj_light_dir()

    scene_triangles = []

    def project_to_screen(view_v):
        minus_z = -view_v[2]
        if minus_z == 0:
            return None
        else:
            screen_v = vecmat.vec4_mat4_mul(view_v, persp_m)
            return [screen_v[0]/minus_z, screen_v[1]/minus_z]

    def get_visible_instance_tris(tris, view_verts, no_culling, clip_planes=(-0.5,-100)):
        """Returns ([indices of visible triangles],
            [normals of all tris],
            [screen verts of visible tris])"""
        idcs = []
        normals = []
        screen_verts = list(map(project_to_screen, view_verts))
        i = -1
        near_clip = clip_planes[0]
        far_clip = clip_planes[1]
        for tri in tris:
            i += 1
            v_0 = view_verts[tri[0]]
            v_1 = view_verts[tri[1]]
            v_2 = view_verts[tri[2]]
            # clip near/far plane
            if ( (v_0[2] >= near_clip or v_1[2] >= near_clip or v_2[2] >= near_clip)
            or (v_0[2] <= far_clip  or v_1[2] <= far_clip  or v_2[2] <= far_clip)):
                normals.append(None)
                continue
            # clip left/right/top/bottom
            sv_0 = screen_verts[tri[0]]
            sv_1 = screen_verts[tri[1]]
            sv_2 = screen_verts[tri[2]]
            one_scr_v_visible = (
                   ((sv_0[0] >= -1 and sv_0[0] <= 1) and (sv_0[1] >= -1 and sv_0[1] <= 1))
                or ((sv_1[0] >= -1 and sv_1[0] <= 1) and (sv_1[1] >= -1 and sv_1[1] <= 1))
                or ((sv_2[0] >= -1 and sv_2[0] <= 1) and (sv_2[1] >= -1 and sv_2[1] <= 1)))
            if not one_scr_v_visible:
                normals.append(None)
                continue
            # normal = cross_product(v_1 - v_0, v_2 - v_0)
            sub10 = (v_1[0] - v_0[0], v_1[1] - v_0[1], v_1[2] - v_0[2])
            sub20 = (v_2[0] - v_0[0], v_2[1] - v_0[1], v_2[2] - v_0[2])
            normal = (sub10[1]*sub20[2] - sub10[2]*sub20[1],
                sub10[2]*sub20[0] - sub10[0]*sub20[2],
                    sub10[0]*sub20[1] - sub10[1]*sub20[0])
            normals.append(normal)
            # Back-face culling: visible if dot_product(v_0, normal) < 0
            if no_culling or (v_0[0] * normal[0] + v_0[1] * normal[1] + v_0[2] * normal[2]) < 0:
                idcs.append(i)
        return (idcs, normals, screen_verts)

    def get_screen_tris_for_instance(instance, model_m):
        """Get (lighted) triangles from this instance and insert them into scene_triangles"""
        model = instance["model"]
        if not model:
            return

        def project_to_view(model_v):
            """Return vec4"""
            return vecmat.vec4_mat4_mul((model_v[0], model_v[1], model_v[2], 1), model_m)
        view_verts = list(map(project_to_view, model["verts"]))

        draw_as_wireframe = ("wireframe" in instance) and instance["wireframe"]
        no_culling = ("noCulling" in instance) and instance["noCulling"]
        model_colors = model["colors"]
        model_tris = model["tris"]

        idcs,normals,screen_verts = get_visible_instance_tris(model_tris, view_verts, no_culling)
        screen_verts = [(int(scr_origin_x + v_2[0] * scr_origin_x),
            int(scr_origin_y - v_2[1] * scr_origin_y)) for v_2 in screen_verts]

        for idx in idcs:
            tri = model_tris[idx]
            points = [screen_verts[tri[i]] for i in range(3)]
            if not draw_as_wireframe:
                normal = vecmat.norm_vec3(normals[idx])
                color = model_colors[idx]
                dot_prd = max(0, proj_light_dir[0] * normal[0]
                    + proj_light_dir[1] * normal[1]
                    + proj_light_dir[2] * normal[2])
                intensity = min(1, max(0, ambient + diffuse * dot_prd))
                lighted_color = (intensity * color[0], intensity * color[1], intensity * color[2])
            else:
                lighted_color = model_colors[idx]
            scene_triangles.append((
                (view_verts[tri[0]][2] + view_verts[tri[1]][2] + view_verts[tri[2]][2]) / 3,
                points, lighted_color, draw_as_wireframe))

    def traverse_scene_graph(subgraph, parent_m):
        for _,instance in subgraph.items():
            if instance["enabled"]:
                proj_m = vecmat.mat4_mat4_mul(instance["xform_m4"], instance["preproc_m4"])
                proj_m = vecmat.mat4_mat4_mul(parent_m, proj_m)
                proj_m = vecmat.mat4_mat4_mul(camera_m, proj_m)
                get_screen_tris_for_instance(instance, proj_m)
                pass_m = vecmat.mat4_mat4_mul(parent_m, instance["xform_m4"])
                if instance["children"]:
                    traverse_scene_graph(instance["children"], pass_m)

    traverse_scene_graph(scene_graph, vecmat.get_unit_m4())
    # Wireframe lines are always drawn last
    scene_triangles.sort(key=lambda x: (1 if x[3] else 0, x[0]), reverse=False)
    for _,points,color,draw_as_wireframe in scene_triangles:
        if not draw_as_wireframe:
            pygame.draw.polygon(surface, color, points)
        else:
            pygame.draw.lines(surface, color, True, points)

def get_selection(screen_area, mouse_pos, scene_graph, camera_m):
    """Return closest instance"""
    # TODO apply scaling transforms to bounding sphere radius
    r_orig = [0, 0, 0]
    # TODO handle non standard screen region
    r_dir = vecmat.mouse_pos_to_ray(mouse_pos, [screen_area[2], screen_area[3]])
    min_t = -1
    selected = None

    def check_if_selected(instance, model_m):
        # TODO handle off center bounding spheres
        sph_orig = vecmat.vec4_mat4_mul((0, 0, 0, 1), model_m)[:3]
        sph_r = instance["bound_sph_r"]
        return vecmat.ray_sphere_intersect(r_orig, r_dir, sph_orig, sph_r)

    def traverse_scene_graph(subgraph, parent_m):
        for _,instance in subgraph.items():
            proj_m = vecmat.mat4_mat4_mul(instance["xform_m4"], instance["preproc_m4"])
            proj_m = vecmat.mat4_mat4_mul(parent_m, proj_m)
            proj_m = vecmat.mat4_mat4_mul(camera_m, proj_m)

            if "bound_sph_r" in instance:
                ray_t = check_if_selected(instance, proj_m)
                if ray_t is not None:
                    nonlocal min_t
                    nonlocal selected
                    if min_t < 0:
                        min_t = ray_t
                        selected = instance
                    else:
                        if ray_t < min_t:
                            min_t = ray_t
                            selected = instance
            else: # Only check outermost models for selection
                pass_m = vecmat.mat4_mat4_mul(parent_m, instance["xform_m4"])
                if instance["children"]:
                    traverse_scene_graph(instance["children"], pass_m)

    traverse_scene_graph(scene_graph, vecmat.get_unit_m4())
    return selected
