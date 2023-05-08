#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D rasterizer engine
"""

from collections import deque
import math

import pygame

DEBUG_FONT = None

from . import vecmat
from . import drawing

DRAW_MODE_WIREFRAME = 0
DRAW_MODE_FLAT = 1
DRAW_MODE_GOURAUD = 2
DRAW_MODE_BILLBOARD = 3
DRAW_MODE_PARTICLE = 4

BILLBOARD_PLAY_ALWAYS = 0
BILLBOARD_PLAY_ONCE = 1


def get_model_instance(model, preproc_m4=None, xform_m4=None, children=None):
    """Return model instance
    These are the key values in a scene graph {name_1: instance_1, ...} dictionary
    Optional keys:
    * wireframe (boolean): draw model as wireframe instead of filled triangles
    * bound_sph_r (float): sets radius of the bounding sphere of this model, can check for e.g. selection
    * noCulling (boolean): don't cull back face triangles when drawing this model
    * gouraud (boolean):   draw model with Gouraud shaded triangles
    """
    if preproc_m4 is None:
        preproc_m4 = vecmat.get_unit_m4()
    if xform_m4 is None:
        xform_m4 = vecmat.get_unit_m4()
    if children is None:
        children = {}

    # It's about 10% faster not to have to create temporary vec4s for matmults
    if model is not None and "billboard" not in model and "particles" not in model:
        model["verts"] = list([model_v[0], model_v[1], model_v[2], 1.0] for model_v in model["verts"])

    if model is not None and "billboard" not in model and "particles" not in model and "normals" not in model:
        normals = []
        verts = model["verts"]
        sum_normals = [[0, 0, 0] for _ in range(len(verts))]
        for tri in model["tris"]:
            i_0 = tri[0]
            i_1 = tri[1]
            i_2 = tri[2]
            v_0 = verts[i_0]
            v_1 = verts[i_1]
            v_2 = verts[i_2]
            v_a = vecmat.sub_vec3(v_1, v_0)
            v_b = vecmat.sub_vec3(v_2, v_0)
            normal = vecmat.norm_vec3(vecmat.cross_vec3(v_a, v_b))
            normals.append(normal)

            n_x = normal[0]
            n_y = normal[1]
            n_z = normal[2]
            sum_normals[i_0][0] += n_x
            sum_normals[i_0][1] += n_y
            sum_normals[i_0][2] += n_z
            sum_normals[i_1][0] += n_x
            sum_normals[i_1][1] += n_y
            sum_normals[i_1][2] += n_z
            sum_normals[i_2][0] += n_x
            sum_normals[i_2][1] += n_y
            sum_normals[i_2][2] += n_z

        model["normals"] = [[*v, 0.0] for v in normals]
        model["vert_normals"] = list(map(lambda v: [*vecmat.norm_vec3(v), 0.0], sum_normals))

    return {
        "enabled" : True,
        "model": model,
        "preproc_m4": preproc_m4,
        "xform_m4": xform_m4,
        "children": children
        }


def visit_instances(scene_graph, func, enabled_only=False):
    """
    Call func(name, instance) on all instances of the scene graph
    """
    for name,instance in scene_graph.items():
        if not enabled_only or instance["enabled"]:
            func(name, instance)
            if instance["children"]:
                visit_instances(instance["children"], func, enabled_only)


def get_proj_light_dir(lighting, camera_m):
    """
    Get the resulting light direction for the given camera
    """
    light_dir_vec3 = lighting["lightDir"]
    light_dir_vec4 = (light_dir_vec3[0], light_dir_vec3[1], light_dir_vec3[2], 0)
    return vecmat.norm_vec3(vecmat.vec4_mat4_mul(light_dir_vec4, camera_m)[0:3])


def project_to_clip_space(view_v, persp_m):
    """
    Project view space point to clip space.
    Points on the POSITIVE side of z are mirrored!
    Takes vec4
    Returns vec3 (last component is original z) or None
    """
    z = view_v[2]
    if z == 0:
        return None
    perp_div = abs(z)
    screen_v = vecmat.vec4_mat4_mul(view_v, persp_m)
    return [screen_v[0] / perp_div, screen_v[1] / perp_div, z]

def clip_space_tri_overlaps_view_frustum(v_0, v_1, v_2, near_clip, far_clip):
    """
    Args are vec3s in clip space
    """
    # All coordinate intervals must overlap
    min_x = min(v_0[0], min(v_1[0], v_2[0]))
    max_x = max(v_0[0], max(v_1[0], v_2[0]))
    # Intervals overlap if they have at least one common point,
    # i.e. if max(left bounds) <= min(right bounds)
    if not (max(min_x, -1) <= min(max_x, 1)):
        return False
    min_y = min(v_0[1], min(v_1[1], v_2[1]))
    max_y = max(v_0[1], max(v_1[1], v_2[1]))
    if not (max(min_y, -1) <= min(max_y, 1)):
        return False
    min_z = min(v_0[2], min(v_1[2], v_2[2]))
    max_z = max(v_0[2], max(v_1[2], v_2[2]))
    # furthest z is MORE NEGATIVE (smallest), so far clip < near clip
    if not (max(min_z, far_clip) <= min(max_z, near_clip)):
        return False
    return True

def _get_visible_instance_tris(persp_m, near_clip, far_clip, model, view_verts, view_normals, vert_normals, no_culling):
    """
    Compute the triangles we can see, i.e. are not back facing or outside view frustum
    - Also returns clip space projections of all vertices
    - Also may create new triangles due to clipping.
    Returns: ([indices of visible triangles], [clip space verts of visible tris])
    SIDE EFFECTS: model's tris and colors get new entries which should be removed immediately!
    """
    visible_tri_idcs = []
    clip_verts = list(map(lambda x: project_to_clip_space(x, persp_m), view_verts))

    # TODO clip for textured triangles
    textured = "texture" in model

    # May add new triangles as we loop
    tris = model["tris"]
    num_orig_tris = len(tris)
    for tri_idx in range(num_orig_tris):
        tri = tris[tri_idx]

        i_0 = tri[0]
        i_1 = tri[1]
        i_2 = tri[2]

        cv_0 = clip_verts[i_0]
        cv_1 = clip_verts[i_1]
        cv_2 = clip_verts[i_2]

        if cv_0 is None or cv_1 is None or cv_2 is None:
            # Can not process vertices that sit exactly on the xy plane with z=0
            continue

        if not clip_space_tri_overlaps_view_frustum(cv_0, cv_1, cv_2, near_clip, far_clip):
            continue

        v_0 = view_verts[i_0]
        v_1 = view_verts[i_1]
        v_2 = view_verts[i_2]

        v_0_behind = v_0[2] > near_clip
        v_1_behind = v_1[2] > near_clip
        v_2_behind = v_2[2] > near_clip
        num_behind = (1 if v_0_behind else 0) + (1 if v_1_behind else 0) + (1 if v_2_behind else 0)
        if num_behind == 3:
            # Clip triangles totally behind near clip plane
            continue

        # Back-face culling: visible if dot_product(v_0, normal) < 0
        if not no_culling:
            normal = view_normals[tri_idx]
            if not ((v_0[0] * normal[0] + v_0[1] * normal[1] + v_0[2] * normal[2]) < 0):
                continue

        if not textured: # TODO
            colors = model["colors"]
            # Check if triangle extends behind near clip plane
            if num_behind == 2:
                if not v_0_behind:
                    front_point = v_0
                    back_point_1 = v_1
                    back_point_2 = v_2
                    front_point_i = i_0
                    back_point_1_i = i_1
                    back_point_2_i = i_2
                elif not v_1_behind:
                    front_point = v_1
                    back_point_1 = v_0
                    back_point_2 = v_2
                    front_point_i = i_1
                    back_point_1_i = i_0
                    back_point_2_i = i_2
                else:
                    front_point = v_2
                    back_point_1 = v_0
                    back_point_2 = v_1
                    front_point_i = i_2
                    back_point_1_i = i_0
                    back_point_2_i = i_1
                front_point_z = front_point[2]
                # t = (near - v0.z) / (v1.z - v0.z)
                intersect_t_1 = (near_clip - front_point_z) / (back_point_1[2] - front_point_z)
                intersect_t_2 = (near_clip - front_point_z) / (back_point_2[2] - front_point_z)
                new_back_1 = [
                    front_point[0] + intersect_t_1 * (back_point_1[0] - front_point[0]),
                    front_point[1] + intersect_t_1 * (back_point_1[1] - front_point[1]),
                    front_point[2] + intersect_t_1 * (back_point_1[2] - front_point[2]),
                    1.0
                ]
                new_back_2 = [
                    front_point[0] + intersect_t_2 * (back_point_2[0] - front_point[0]),
                    front_point[1] + intersect_t_2 * (back_point_2[1] - front_point[1]),
                    front_point[2] + intersect_t_2 * (back_point_2[2] - front_point[2]),
                    1.0
                ]
                # Add the new vertices and their screen projections to the end of the list
                new_verts_idx = len(view_verts)
                view_verts.append(new_back_1)
                clip_verts.append(project_to_clip_space(new_back_1, persp_m))
                view_verts.append(new_back_2)
                clip_verts.append(project_to_clip_space(new_back_2, persp_m))
                # Copy the normals of the original triangle and vertices
                view_normals.append(view_normals[tri_idx])
                if vert_normals is not None:
                    vert_normals.append(vert_normals[back_point_1_i])
                    vert_normals.append(vert_normals[back_point_2_i])
                # Add the new triangle
                visible_tri_idcs.append(len(tris))
                tris.append((front_point_i, new_verts_idx, new_verts_idx + 1))
                # Copy the color of the original triangle
                colors.append(colors[tri_idx])
                continue
            elif num_behind == 1:
                if v_0_behind:
                    back_point = v_0
                    front_point_1 = v_1
                    front_point_2 = v_2
                    front_point_i_1 = i_1
                    front_point_i_2 = i_2
                elif v_1_behind:
                    back_point = v_1
                    front_point_1 = v_0
                    front_point_2 = v_2
                    front_point_i_1 = i_0
                    front_point_i_2 = i_2
                else:
                    back_point = v_2
                    front_point_1 = v_0
                    front_point_2 = v_1
                    front_point_i_1 = i_0
                    front_point_i_2 = i_1
                back_point_z = back_point[2]
                intersect_t_1 = (near_clip - back_point_z) / (front_point_1[2] - back_point_z)
                intersect_t_2 = (near_clip - back_point_z) / (front_point_2[2] - back_point_z)
                new_front_1 = [
                    back_point[0] + intersect_t_1 * (front_point_1[0] - back_point[0]),
                    back_point[1] + intersect_t_1 * (front_point_1[1] - back_point[1]),
                    back_point[2] + intersect_t_1 * (front_point_1[2] - back_point[2]),
                    1.0
                ]
                new_front_2 = [
                    back_point[0] + intersect_t_2 * (front_point_2[0] - back_point[0]),
                    back_point[1] + intersect_t_2 * (front_point_2[1] - back_point[1]),
                    back_point[2] + intersect_t_2 * (front_point_2[2] - back_point[2]),
                    1.0
                ]
                # Add the new vertices and their screen projections to the end of the list
                new_verts_idx = len(view_verts)
                view_verts.append(new_front_1)
                clip_verts.append(project_to_clip_space(new_front_1, persp_m))
                view_verts.append(new_front_2)
                clip_verts.append(project_to_clip_space(new_front_2, persp_m))
                # Copy the normals of the original triangle and vertices
                view_normals.append(view_normals[tri_idx])
                view_normals.append(view_normals[tri_idx])
                if vert_normals is not None:
                    vert_normals.append(vert_normals[front_point_i_1])
                    vert_normals.append(vert_normals[front_point_i_2])
                # Add the two new triangles
                visible_tri_idcs.append(len(tris))
                tris.append((front_point_i_1, front_point_i_2, new_verts_idx))
                visible_tri_idcs.append(len(tris))
                tris.append((front_point_i_2, new_verts_idx + 1, new_verts_idx))
                # Copy the colors of the original triangle
                colors.append(colors[tri_idx])
                colors.append(colors[tri_idx])
                continue

        # Append a non-clipped visible triangle
        visible_tri_idcs.append(tri_idx)

    return (visible_tri_idcs, clip_verts)


def _get_screen_tris_for_instance(scene_triangles, near_clip, far_clip, persp_m, scr_origin_x, scr_origin_y,
                                  lighting, proj_light_dir, instance, model_m, camera_m):
    """Get (lighted) triangles from this instance and insert them into scene_triangles"""
    model = instance["model"]
    if not model:
        return

    subdivide_default_max_iterations = 1

    fade_distance = instance["fade_distance"] if "fade_distance" in instance else 0

    if "billboard" in model:
        cam_pos = vecmat.vec4_mat4_mul(model["translate"], model_m)
        cur_z = cam_pos[2]
        if cur_z > near_clip or cur_z < far_clip:
            return
        clip_pos = project_to_clip_space(cam_pos, persp_m)
        if clip_pos is not None:
            scale = model["size_scale"]
            size = (model["size"][0] * scale, model["size"][1] * scale)
            model_imgs = model["img"]
            num_frames = len(model_imgs)
            int_cur_frame = int(model["cur_frame"])
            if int_cur_frame >= num_frames:
                return

            if fade_distance > 0:
                img = model_imgs[int_cur_frame].copy()
                fade_factor = 1
                if fade_distance > 0:
                    z = abs(cur_z)
                    fade_factor = 1 if z < 1 else max(0, (1 / fade_distance) * (fade_distance - z))
                dark = pygame.Surface(img.get_size()).convert_alpha()
                darken_value = fade_factor * 255
                dark.fill((darken_value, darken_value, darken_value, 255))
                img.blit(dark, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
            else:
                img = model_imgs[int_cur_frame]

            inv_z = 1.0 / abs(cur_z)
            proj_size = (img.get_width() * inv_z * size[0], img.get_height() * inv_z * size[1])
            scale_img = pygame.transform.scale(img, proj_size)
            scr_pos = (int(scr_origin_x + clip_pos[0] * scr_origin_x - scale_img.get_width() / 2),
                       int(scr_origin_y - clip_pos[1] * scr_origin_y - scale_img.get_height() / 2))
            if num_frames > 1:
                model["cur_frame"] += model["frame_advance"]
                if int(model["cur_frame"]) == num_frames:
                    if model["play_mode"] == BILLBOARD_PLAY_ALWAYS:
                        model["cur_frame"] = 0

            scene_triangles.append((
                cur_z,
                scr_pos,
                scale_img,
                DRAW_MODE_BILLBOARD))
        return

    if "particles" in model:
        img = model["img"]
        size = model["size"]
        enabled = model["enabled"]
        cam_positions = [vecmat.vec4_mat4_mul(v, model_m) for v in model["positions"]]
        for i in range(len(cam_positions)):
            if not enabled[i]:
                continue
            cam_pos = cam_positions[i]
            cur_z = cam_pos[2]
            if cur_z > near_clip:
                continue
            clip_pos = project_to_clip_space(cam_pos, persp_m)
            if clip_pos is not None:
                inv_z = 1.0 / abs(cur_z)
                proj_size = (img.get_width() * inv_z * size[0], img.get_height() * inv_z * size[1])
                scale_img = pygame.transform.scale(img, proj_size)
                scr_pos = (int(scr_origin_x + clip_pos[0] * scr_origin_x - scale_img.get_width() / 2),
                        int(scr_origin_y - clip_pos[1] * scr_origin_y - scale_img.get_height() / 2))
                scene_triangles.append((
                    cur_z,
                    scr_pos,
                    scale_img,
                    DRAW_MODE_PARTICLE))
        return

    view_verts = list(map(lambda model_v: vecmat.vec4_mat4_mul(model_v, model_m), model["verts"]))
    view_normals = list(map(lambda model_n: vecmat.norm_vec3_from_vec4(vecmat.vec4_mat4_mul(model_n, model_m)), model["normals"]))

    draw_as_wireframe = ("wireframe" in instance) and instance["wireframe"]
    no_culling = ("noCulling" in instance) and instance["noCulling"]
    model_tris = model["tris"]
    draw_gouraud_shaded = ("gouraud" in instance) and instance["gouraud"]
    textured = "texture" in model
    use_minimum_z_order = ("use_minimum_z_order" in instance) and instance["use_minimum_z_order"]
    pointlight_enabled = ("pointlight_enabled" in lighting) and lighting["pointlight_enabled"]
    if pointlight_enabled:
        pointlight_falloff = lighting["pointlight_falloff"]
        pointlight_cam_pos = vecmat.vec4_mat4_mul(lighting["pointlight"], camera_m)
    subdivide_max_iterations = instance["subdivide_max_iterations"] if "subdivide_max_iterations" in instance else subdivide_default_max_iterations

    if "instance_normal" in instance:
        instance_normal = instance["instance_normal"]
        proj_inst_normal = vecmat.vec4_mat4_mul((instance_normal[0], instance_normal[1], instance_normal[2], 0), model_m)
        # Skip instance if it faces away from camera (visible if dot_product(v_0, normal) < 0)
        v_instance = vecmat.vec4_mat4_mul((0, 0, 0, 1), model_m)
        if (v_instance[0] * proj_inst_normal[0] + v_instance[1] * proj_inst_normal[1] + v_instance[2] * proj_inst_normal[2]) >= 0:
            return

    vert_normals = None
    if draw_gouraud_shaded:
        vert_normals = list(map(lambda model_n: vecmat.norm_vec3_from_vec4(vecmat.vec4_mat4_mul(model_n, model_m)), model["vert_normals"]))

    # This function may add temporary triangles due to clipping
    # We reset the model's lists to their original size after processing
    num_orig_model_tris = len(model_tris)
    visible_tri_idcs,screen_verts = _get_visible_instance_tris(persp_m, near_clip, far_clip, model, view_verts, view_normals, vert_normals, no_culling)
    screen_verts = [(int(scr_origin_x + v_2[0] * scr_origin_x), int(scr_origin_y - v_2[1] * scr_origin_y)) if v_2 is not None else None for v_2 in screen_verts]

    draw_mode = DRAW_MODE_WIREFRAME if draw_as_wireframe else (DRAW_MODE_GOURAUD if draw_gouraud_shaded else DRAW_MODE_FLAT)

    ambient = lighting["ambient"]
    diffuse = lighting["diffuse"]

    if not textured:
        model_colors = model["colors"]

    # Compute colors for each required vertex for Gouraud shading
    vert_colors = [None] * len(view_verts)
    if draw_gouraud_shaded:
        for tri_idx in visible_tri_idcs:
            if not textured:
                tri_color = model_colors[tri_idx]
            tri = model_tris[tri_idx]
            for vert_idx in tri:
                if vert_colors[vert_idx] is None:
                    normal = vert_normals[vert_idx]
                    dot_prd = max(0, proj_light_dir[0] * normal[0]
                        + proj_light_dir[1] * normal[1]
                        + proj_light_dir[2] * normal[2])
                    intensity = min(1, max(0, ambient + diffuse * dot_prd))
                    if textured:
                        vert_colors[vert_idx] = intensity
                    else:
                        vert_colors[vert_idx] = (intensity * tri_color[0], intensity * tri_color[1], intensity * tri_color[2])

    for tri_idx in visible_tri_idcs:
        tri = model_tris[tri_idx]

        # Using the minimum tends to look glitchier in a lot of cases,
        # but also works better for placement of billboards and big triangles
        if use_minimum_z_order:
            z_order = min(view_verts[tri[0]][2], view_verts[tri[1]][2], view_verts[tri[2]][2])
        else:
            z_order = (view_verts[tri[0]][2] + view_verts[tri[1]][2] + view_verts[tri[2]][2]) / 3

        if draw_mode == DRAW_MODE_WIREFRAME:
            color_data = model_colors[tri_idx]
        elif draw_mode == DRAW_MODE_FLAT:
            normal = view_normals[tri_idx]
            dot_prd = max(0, proj_light_dir[0] * normal[0]
                + proj_light_dir[1] * normal[1]
                + proj_light_dir[2] * normal[2])
            intensity = min(1, max(0, ambient + diffuse * dot_prd))

            fade_factor = 1
            if fade_distance > 0:
                z = abs(z_order)
                fade_factor = 1 if z < 1 else max(0, (1 / fade_distance) * (fade_distance - z))
                intensity *= fade_factor

            if pointlight_enabled:
                centroid = vecmat.get_vec3_triangle_centroid(view_verts[tri[0]], view_verts[tri[1]], view_verts[tri[2]])
                dist_to_light = vecmat.mag_vec3(vecmat.sub_vec3(centroid, pointlight_cam_pos))
                intensity += 1 if dist_to_light < 1 else max(0, (1 / pointlight_falloff) * (pointlight_falloff - dist_to_light))

            intensity = min(1, intensity)
            textured = "texture" in model
            color_data = (textured,
                          intensity, 
                          model["texture"] if textured else model_colors[tri_idx],
                          [model["uv"][vert_idx] for vert_idx in tri] if textured else None,
                          subdivide_max_iterations)
        else: # draw_mode == DRAW_MODE_GOURAUD:
            if textured:
                uv = model["uv"]
                color_data = [textured, [uv[vert_idx] for vert_idx in tri],
                              model["texture"],
                              [vert_colors[vert_idx] for vert_idx in tri],
                              subdivide_max_iterations]
            else:
                color_data = [textured, [vert_colors[vert_idx] for vert_idx in tri], subdivide_max_iterations]

        scene_triangles.append((
            z_order,
            [screen_verts[tri[i]] for i in range(3)],
            color_data,
            draw_mode))

    # Remove temporary triangles
    if num_orig_model_tris != len(model_tris):
        del model_tris[num_orig_model_tris:]
        del model_colors[num_orig_model_tris:]

def render(surface, screen_area, scene_graph, camera_m, persp_m, lighting, near_clip=-0.5, far_clip=-100.0, mip_dist=50):
    """Render the scene graph
    screen_area is (x,y,w,h) inside the surface
    """
    # global DEBUG_FONT
    # if DEBUG_FONT is None:
    #     DEBUG_FONT = pygame.font.Font(None, 16)

    scr_origin_x = screen_area[0] + screen_area[2] / 2
    scr_origin_y = screen_area[1] + screen_area[3] / 2

    scr_min_x = screen_area[0]
    scr_max_x = screen_area[0] + screen_area[2] - 1
    scr_min_y = screen_area[1]
    scr_max_y = screen_area[1] + screen_area[3] - 1

    # Collect all visible triangles. Elements are tuples:
    # (average z depth, screen points of triangle, lighted color, draw mode)
    # Sorted by depth before drawing, draw mode overrides order so wireframes come last
    scene_triangles = []

    proj_light_dir = get_proj_light_dir(lighting, camera_m)

    def traverse_scene_graph(subgraph, parent_m):
        for _,instance in subgraph.items():
            if instance["enabled"]:
                proj_m = vecmat.mat4_mat4_mul(instance["xform_m4"], instance["preproc_m4"])
                proj_m = vecmat.mat4_mat4_mul(parent_m, proj_m)
                proj_m = vecmat.mat4_mat4_mul(camera_m, proj_m)
                _get_screen_tris_for_instance(scene_triangles, near_clip, far_clip, persp_m,
                                              scr_origin_x, scr_origin_y, lighting, proj_light_dir,
                                              instance, proj_m, camera_m)
                pass_m = vecmat.mat4_mat4_mul(parent_m, instance["xform_m4"])
                if instance["children"]:
                    traverse_scene_graph(instance["children"], pass_m)

    # Traverse the scene graph and build scene_triangles values
    traverse_scene_graph(scene_graph, vecmat.get_unit_m4())

    # Sort triangles in ascending z order but wireframe triangles should be drawn last
    scene_triangles.sort(key=lambda x: (1 if x[3] == DRAW_MODE_WIREFRAME else 0, x[0]), reverse=False)
    # print(f"tris: {len(scene_triangles)} -> {[v[1] for v in scene_triangles]}")

    for z_order,points,color_data,draw_mode in scene_triangles:
        if draw_mode == DRAW_MODE_GOURAUD:
            textured = color_data[0]

            v_a = (points[0][0], points[0][1])
            v_b = (points[1][0], points[1][1])
            v_c = (points[2][0], points[2][1])
            bary = vecmat.Barycentric2dTriangle(v_a, v_b, v_c)
            if bary.area_sq == 0:
                continue

            if not textured:
                subdivide_max_iterations = color_data[2]
                colors = color_data[1]
                avg_color = vecmat.get_average_color(colors[0], colors[1], colors[2])
                col_diff = sum([abs(a-i) + abs(a-j) + abs(a-k)
                            for a,i,j,k in zip(avg_color, colors[0], colors[1], colors[2])])
                if col_diff <= 20:
                    pygame.draw.polygon(surface, avg_color, ((v_a[0], v_a[1]), (v_b[0], v_b[1]), (v_c[0], v_c[1])))
                    continue
            else:
                intensities = color_data[3]
                subdivide_max_iterations = color_data[4]

            if not textured:
                if bary.area_sq <= 10:
                    pygame.draw.polygon(surface, avg_color, ((v_a[0], v_a[1]), (v_b[0], v_b[1]), (v_c[0], v_c[1])))
                    continue

            def get_interpolated_color(x, y):
                u,v,w = bary.get_uvw(x, y)
                r = max(0, min(255, int(colors[0][0] * u + colors[1][0] * v + colors[2][0] * w)))
                g = max(0, min(255, int(colors[0][1] * u + colors[1][1] * v + colors[2][1] * w)))
                b = max(0, min(255, int(colors[0][2] * u + colors[1][2] * v + colors[2][2] * w)))
                return (r, g, b)

            if textured:
                uv = color_data[1]
                mip_textures = color_data[2]
                tex_ip = vecmat.TextureInterpolation(uv, mip_textures, z_order, mip_dist)

            if subdivide_max_iterations > 0:
                area = math.sqrt(bary.area_sq)
                def cb_subdivide_gouraud(v_0, v_1, v_2, iteration):
                    if textured:
                        divisor = 2 ** iteration
                        uv_w, uv_h = tex_ip.uv_extent[0] / divisor, tex_ip.uv_extent[1] / divisor
                        pix_w, pix_h = uv_w * tex_ip.tex_w, uv_h * tex_ip.tex_h
                        if pix_w <= 1 or pix_h <= 1:
                            centroid = vecmat.get_vec2_triangle_centroid(v_0, v_1, v_2)
                            x,y = centroid[0], centroid[1]
                            u,v,w = bary.get_uvw(x, y)
                            color = tex_ip.get_color(u, v, w)
                            intensity = u * intensities[0] + v * intensities[1] + w * intensities[2]
                            color = (intensity * color[0], intensity * color[1], intensity * color[2])
                            pygame.draw.polygon(surface, color, ((v_0[0], v_0[1]), (v_1[0], v_1[1]), (v_2[0], v_2[1])))
                            return True
                    else:
                        cur_area = area / (4 ** iteration)
                        if cur_area < 10 or iteration == subdivide_max_iterations:
                            c_0 = get_interpolated_color(*v_0)
                            c_1 = get_interpolated_color(*v_1)
                            c_2 = get_interpolated_color(*v_2)
                            avg_color = vecmat.get_average_color(c_0, c_1, c_2)
                            pygame.draw.polygon(surface, avg_color, ((v_0[0], v_0[1]), (v_1[0], v_1[1]), (v_2[0], v_2[1])))
                            return True
                vecmat.subdivide_2d_triangle(v_a, v_b, v_c, cb_subdivide_gouraud)
            else:
                # Per pixel Gouraud shading
                px_array = pygame.PixelArray(surface) # TODO pygbag doesn't like this
                if textured:
                    for x,y in drawing.triangle(v_a[0], v_a[1], v_b[0], v_b[1], v_c[0], v_c[1]):
                        if x < scr_min_x or x > scr_max_x or y < scr_min_y or y > scr_max_y:
                            continue
                        u,v,w = bary.get_uvw(x, y)
                        color = tex_ip.get_color(u, v, w)
                        intensity = u * intensities[0] + v * intensities[1] + w * intensities[2]
                        color = (intensity * color[0], intensity * color[1], intensity * color[2])
                        px_array[x, y] = color
                else:
                    for x,y in drawing.triangle(v_a[0], v_a[1], v_b[0], v_b[1], v_c[0], v_c[1]):
                        if x < scr_min_x or x > scr_max_x or y < scr_min_y or y > scr_max_y:
                            continue
                        r,g,b = get_interpolated_color(x, y)
                        px_array[x, y] = (r << 16) | (g << 8) | b
                del px_array
        elif draw_mode == DRAW_MODE_FLAT:
            textured = color_data[0]
            intensity = color_data[1]
            subdivide_max_iterations = color_data[4]
            if textured:
                mip_textures = color_data[2]
                uv = color_data[3]
                v_a = (points[0][0], points[0][1])
                v_b = (points[1][0], points[1][1])
                v_c = (points[2][0], points[2][1])
                bary = vecmat.Barycentric2dTriangle(v_a, v_b, v_c)
                if bary.area_sq == 0:
                    continue
                tex_ip = vecmat.TextureInterpolation(uv, mip_textures, z_order, mip_dist)
                def cb_subdivide(v_0, v_1, v_2, iteration):
                    divisor = 2 ** iteration
                    uv_w, uv_h = tex_ip.uv_extent[0] / divisor, tex_ip.uv_extent[1] / divisor
                    pix_w, pix_h = uv_w * tex_ip.tex_w, uv_h * tex_ip.tex_h
                    if pix_w <= 1 or pix_h <= 1 or iteration == subdivide_max_iterations:
                        centroid = vecmat.get_vec2_triangle_centroid(v_0, v_1, v_2)
                        x,y = centroid[0], centroid[1]
                        u,v,w = bary.get_uvw(x, y)
                        color = tex_ip.get_color(u, v, w)
                        color = (intensity * color[0], intensity * color[1], intensity * color[2])
                        pygame.draw.polygon(surface, color, ((v_0[0], v_0[1]), (v_1[0], v_1[1]), (v_2[0], v_2[1])))
                        return True
                    return False
                vecmat.subdivide_2d_triangle(v_a, v_b, v_c, cb_subdivide)
            else:
                color = color_data[2]
                color = (intensity * color[0], intensity * color[1], intensity * color[2])
                pygame.draw.polygon(surface, color, points)
        elif draw_mode == DRAW_MODE_WIREFRAME:
            pygame.draw.lines(surface, color_data, True, points)
        elif draw_mode == DRAW_MODE_BILLBOARD:
            surface.blit(color_data, points)
        elif draw_mode == DRAW_MODE_PARTICLE:
            surface.blit(color_data, points)

def get_animated_billboard(dx, dy, dz, sx, sy, img_list):
    """Create a billboard object with several animation frames"""
    return {
        "billboard": True,
        "translate": [dx, dy, dz, 1.0],
        "size": [sx, sy],
        "size_scale": 1.0,
        "img": img_list,
        "cur_frame": 0.0,
        "play_mode": BILLBOARD_PLAY_ALWAYS,
        "frame_advance": 1.0,
    }

def get_billboard(dx, dy, dz, sx, sy, img):
    """Create a billboard object with only one animation frame"""
    return get_animated_billboard(dx, dy, dz, sx, sy, [img])

def get_particles(img, num_particles, sx, sy):
    """Create a particles object"""
    return {
        "particles": True,
        "positions": [[0.0, 0.0, 0.0, 1.0] for _ in range(num_particles)],
        "enabled": [True] * num_particles,
        "img": img,
        "size": [sx, sy],
        "user_data": []
    }
