#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# 3D rasterizer engine

Entry point: render()

Painter's Algorithm implementation:
- Traverse scene graph, extract prims from each instance
  results in => scene_primitives = [(z order, screen points, shading data, DRAW_MODE_*), ...]
  - _get_screen_primitives_for_instance() appends to scene_primitives
    - MODEL_TYPE_BILLBOARD/MODEL_TYPE_PARTICLES handled on their own
    - Meshes:
      - MODEL_TYPE_ANIMATED_MESH picks underlying mesh model depending on playback frame, then as usual
      - view_verts = [model vertices X current matrix]
      - view_normals = [model normals X current matrix]
- Sort scene_primitives by z
- Draw all in scene_primitives

"""

import math

import pygame
from pygame.gfxdraw import aapolygon, filled_polygon

DEBUG_FONT = None

from .vecmat import get_unit_m4, sub_vec3, norm_vec3, norm_vec3_from_vec4
from .vecmat import get_vec3_triangle_centroid, mag_vec3, mat4_mat4_mul, get_unit_m4
from .vecmat import get_2d_triangle_area, get_barycentric_vec2, get_average_color
from .vecmat import TextureInterpolation, get_vec2_triangle_centroid, subdivide_2d_triangle_x4
from .vecmat import get_2d_triangle_area, subdivide_2d_triangle, mul_vec3, add_vec3
from .vecmat import cross_vec3, vec4_mat4_mul_for_points, vec4_mat4_mul_for_dirs

from . import drawing
from . import meshes as pyr_meshes

DRAW_MODE_WIREFRAME = 0
DRAW_MODE_FLAT = 1
DRAW_MODE_GOURAUD = 2
DRAW_MODE_BILLBOARD = 3
DRAW_MODE_PARTICLE = 4
DRAW_MODE_DEBUG_DOT = 5
DRAW_MODE_TEXTURE_RECT = 6

BILLBOARD_PLAY_ALWAYS = 0
BILLBOARD_PLAY_ONCE = 1

MODEL_TYPE_MESH = 0
MODEL_TYPE_BILLBOARD = 1
MODEL_TYPE_PARTICLES = 2
MODEL_TYPE_ANIMATED_MESH = 3
MODEL_TYPE_TEXTURE_RECT = 4


def get_default_render_settings():
    """
    Create a default render settings/lighting setup with reasonable values
    """
    return {"near_clip": -0.5,
            "far_clip": -100.0,
            "mip_dist": 50,

            # Light comes from a right, top, and back direction (over the "right shoulder")
            "lightDir": (1, 1, 1),
            "ambient": 0.3,
            "diffuse": 0.7,

            "pointlight_enabled": False,
            "pointlight": [0, 0, 0, 1], # Position vec4
            "pointlight_falloff": 5,
            "pointlight_color": (255, 255, 255), # TODO

            "fog_distance": 0, # 0 means no fog
            "fog_color": [0, 0, 0, 0]
            }

def get_model_instance(model : dict, preproc_m4 : list = None, xform_m4 : list = None, children : dict = None, create_bbox=True) -> dict:
    """
    Create a model instance

    Instances reference an actual mesh (model) and store a transformation matrix
    and a preprocessing matrix (applied BEFORE the transformation matrix).
    All children (dictionary) inherit the transform of the parent (but not the preprocess).

    The model may be an animation dict: {"animation_1": [mesh_frame1, mesh_frame2, ...], "animation_2": ...}
    Instance key "animation" chooses which to display.

    Adding extra keys to the instance has different effects:
    *       wireframe (boolean): draw model as wireframe instead of filled triangles
    *       bound_sph_r (float): sets radius of the bounding sphere of this model, can check for e.g. selection
    *       noCulling (boolean): don't cull back face triangles when drawing this model
    *         gouraud (boolean): draw model with Gouraud shaded triangles
    * instance_normal (boolean): Skip instance if it faces away from camera (visible if dot_product(v_0, normal) < 0)
    * ignore_lighting (boolean): ignore lighting, draw with flat triangle color

    """
    if preproc_m4 is None:
        preproc_m4 = get_unit_m4()
    if xform_m4 is None:
        xform_m4 = get_unit_m4()
    if children is None:
        children = {}

    def preprocess_mesh(mesh):
        """
        Cache some values, compute normals and vertex normals for the mesh
        """
        # It's about 10% faster not to have to create temporary vec4s for matmults
        mesh["verts"] = list([model_v[0], model_v[1], model_v[2], 1.0] for model_v in mesh["verts"])
        if create_bbox:
            mesh["sphere_bbox"] = pyr_meshes.get_mesh_sphere_bbox(mesh)

        if "normals" not in mesh:
            normals = []
            verts = mesh["verts"]
            sum_normals = [[0, 0, 0] for _ in range(len(verts))]
            for tri in mesh["tris"]:
                i_0 = tri[0]
                i_1 = tri[1]
                i_2 = tri[2]
                v_0 = verts[i_0]
                v_1 = verts[i_1]
                v_2 = verts[i_2]
                v_a = sub_vec3(v_1, v_0)
                v_b = sub_vec3(v_2, v_0)
                normal = norm_vec3(cross_vec3(v_a, v_b))
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
            mesh["normals"] = [[*v, 0.0] for v in normals]
            mesh["vert_normals"] = list(map(lambda v: [*norm_vec3(v), 0.0], sum_normals))

    if model is not None:
        if model["model_type"] == MODEL_TYPE_MESH:
            preprocess_mesh(model)
        elif model["model_type"] == MODEL_TYPE_ANIMATED_MESH:
            for _,meshes in model["animations"].items():
                for mesh in meshes:
                    preprocess_mesh(mesh)

    return {
        "enabled" : True,
        "model": model,
        "preproc_m4": preproc_m4,
        "xform_m4": xform_m4,
        "children": children,

        # Settings for animation-type models
        "animation": None,
        "animation_frame": 0.0,
        "animation_speed": 1.0
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


def get_proj_light_dir(render_settings, camera_m) -> list:
    """
    Get the resulting light direction for the given camera
    """
    light_dir_vec3 = render_settings["lightDir"]
    return norm_vec3(vec4_mat4_mul_for_dirs(light_dir_vec3, camera_m)[0:3])


def project_to_clip_space(view_v, persp_m):
    """
    Project view space point to clip space.
    Points on the POSITIVE side of z are mirrored!
    Takes vec4
    Returns vec3 (last component is original z) or None
    """
    z = view_v[2]
    perp_div = abs(z)
    # Skip values too close to zero
    if perp_div < 0.0001:
        return None
    screen_v = vec4_mat4_mul_for_points(view_v, persp_m)
    return (screen_v[0] / perp_div, screen_v[1] / perp_div, z)

def clip_space_tri_overlaps_view_frustum(v_0, v_1, v_2, near_clip, far_clip):
    """
    Args are vec3s in clip space
    """
    # All coordinate intervals must overlap
    min_x = min(v_0[0], v_1[0], v_2[0])
    max_x = max(v_0[0], v_1[0], v_2[0])
    # Intervals overlap if they have at least one common point,
    # i.e. if max(left bounds) <= min(right bounds)
    if not (max(min_x, -1) <= min(max_x, 1)):
        return False
    min_y = min(v_0[1], v_1[1], v_2[1])
    max_y = max(v_0[1], v_1[1], v_2[1])
    if not (max(min_y, -1) <= min(max_y, 1)):
        return False
    min_z = min(v_0[2], v_1[2], v_2[2])
    max_z = max(v_0[2], v_1[2], v_2[2])
    # furthest z is MORE NEGATIVE (smallest), so far clip < near clip
    if not (max(min_z, far_clip) <= min(max_z, near_clip)):
        return False
    return True

def clip_space_quad_overlaps_view_frustum(v_0, v_1, v_2, v_3, near_clip, far_clip):
    """
    Args are vec3s in clip space
    """
    # All coordinate intervals must overlap
    min_x = min(v_0[0], v_1[0], v_2[0], v_3[0])
    max_x = max(v_0[0], v_1[0], v_2[0], v_3[0])
    # Intervals overlap if they have at least one common point,
    # i.e. if max(left bounds) <= min(right bounds)
    if not (max(min_x, -1) <= min(max_x, 1)):
        return False
    min_y = min(v_0[1], v_1[1], v_2[1], v_3[1])
    max_y = max(v_0[1], v_1[1], v_2[1], v_3[1])
    if not (max(min_y, -1) <= min(max_y, 1)):
        return False
    min_z = min(v_0[2], v_1[2], v_2[2], v_3[2])
    max_z = max(v_0[2], v_1[2], v_2[2], v_3[2])
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
                new_back_1 = (front_point[0] + intersect_t_1 * (back_point_1[0] - front_point[0]),
                              front_point[1] + intersect_t_1 * (back_point_1[1] - front_point[1]),
                              front_point[2] + intersect_t_1 * (back_point_1[2] - front_point[2]),
                              1.0)
                new_back_2 = (front_point[0] + intersect_t_2 * (back_point_2[0] - front_point[0]),
                              front_point[1] + intersect_t_2 * (back_point_2[1] - front_point[1]),
                              front_point[2] + intersect_t_2 * (back_point_2[2] - front_point[2]),
                              1.0)
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
                new_front_1 = (back_point[0] + intersect_t_1 * (front_point_1[0] - back_point[0]),
                               back_point[1] + intersect_t_1 * (front_point_1[1] - back_point[1]),
                               back_point[2] + intersect_t_1 * (front_point_1[2] - back_point[2]),
                               1.0)
                new_front_2 = (back_point[0] + intersect_t_2 * (front_point_2[0] - back_point[0]),
                               back_point[1] + intersect_t_2 * (front_point_2[1] - back_point[1]),
                               back_point[2] + intersect_t_2 * (front_point_2[2] - back_point[2]),
                               1.0)
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

def _is_bounding_sphere_in_frustum(bbox_center, bbox_radius, model_m, persp_m, near_clip, far_clip):
    """Returns True if at least one point of the sphere bbox is inside frustum"""
    offsets = ((0, 0, 0, 0),
               (bbox_radius, 0, 0, 0),
               (-bbox_radius, 0, 0, 0),
               (0, bbox_radius, 0, 0),
               (0, -bbox_radius, 0, 0),
               (0, 0, bbox_radius, 0),
               (0, 0, -bbox_radius, 0))
    for offset in offsets:
        bbox_p = [bbox_center[i] + offset[i] for i in range(4)]
        bbox_v = vec4_mat4_mul_for_points(bbox_p, model_m)
        bbox_clip = project_to_clip_space(bbox_v, persp_m)
        if bbox_clip is not None:
            cx,cy,cz = bbox_clip
            if (cz <= near_clip and cz >= far_clip) and (cx >= -1 and cx <= 1) and (cy >= -1 and cy <= 1):
                return True
    return False

def _get_screen_primitives_for_instance(scene_primitives, near_clip, far_clip, persp_m, scr_origin_x, scr_origin_y,
                                        render_settings, proj_light_dir, instance, model_m, camera_m):
    """Get primitives and shading data from this instance"""
    model = instance["model"]
    if not model:
        return

    subdivide_default_max_iterations = 1

    ambient = render_settings["ambient"]
    diffuse = render_settings["diffuse"]
    pointlight_enabled = ("pointlight_enabled" in render_settings) and render_settings["pointlight_enabled"]
    if pointlight_enabled:
        pointlight_falloff = render_settings["pointlight_falloff"]
        pointlight_cam_pos = vec4_mat4_mul_for_points(render_settings["pointlight"], camera_m)
    ignore_lighting = ("ignore_lighting" in instance) and instance["ignore_lighting"]

    if model["model_type"] == MODEL_TYPE_BILLBOARD:
        center_pos = vec4_mat4_mul_for_points(model["translate"], model_m)
        cur_z = center_pos[2]
        if cur_z > near_clip or cur_z < far_clip:
            return
        clip_pos = project_to_clip_space(center_pos, persp_m)
        if clip_pos is not None:
            scale = model["size_scale"]
            size = (model["size"][0] * scale, model["size"][1] * scale)
            model_imgs = model["img"]
            num_frames = len(model_imgs)
            int_cur_frame = int(model["cur_frame"])
            if int_cur_frame >= num_frames:
                return

            pointlight_intensity = 0
            if pointlight_enabled:
                dist_to_light = mag_vec3(sub_vec3(center_pos, pointlight_cam_pos))
                pointlight_intensity = 1 if dist_to_light < 1 else max(0, (1 / pointlight_falloff) * (pointlight_falloff - dist_to_light))

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

            scene_primitives.append((
                cur_z,
                scr_pos,
                (scale_img, pointlight_intensity),
                DRAW_MODE_BILLBOARD))
        return
    elif model["model_type"] == MODEL_TYPE_PARTICLES:
        img = model["img"]
        size = model["size"]
        enabled = model["enabled"]
        cam_positions = [vec4_mat4_mul_for_points(v, model_m) for v in model["positions"]]
        for i, center_pos in enumerate(cam_positions):
            if not enabled[i]:
                continue

            cur_z = center_pos[2]
            if cur_z > near_clip:
                continue
            clip_pos = project_to_clip_space(center_pos, persp_m)
            if clip_pos is not None:
                inv_z = 1.0 / abs(cur_z)
                proj_size = (img.get_width() * inv_z * size[0], img.get_height() * inv_z * size[1])
                scale_img = pygame.transform.scale(img, proj_size)
                scr_pos = (int(scr_origin_x + clip_pos[0] * scr_origin_x - scale_img.get_width() / 2),
                           int(scr_origin_y - clip_pos[1] * scr_origin_y - scale_img.get_height() / 2))
                scene_primitives.append((
                    cur_z,
                    scr_pos,
                    scale_img,
                    DRAW_MODE_PARTICLE))
        return
    elif model["model_type"] == MODEL_TYPE_TEXTURE_RECT:
        normal = norm_vec3_from_vec4(vec4_mat4_mul_for_dirs(model["normal"], model_m))
        v_instance = vec4_mat4_mul_for_points((0, 0, 0, 1), model_m)
        if (v_instance[0] * normal[0] + v_instance[1] * normal[1] + v_instance[2] * normal[2]) >= 0:
            return

        dot_prd = max(0, proj_light_dir[0] * normal[0]
            + proj_light_dir[1] * normal[1]
            + proj_light_dir[2] * normal[2])
        intensity = min(1, max(0, ambient + diffuse * dot_prd))

        pointlight_intensity = 0
        if pointlight_enabled:
            dist_to_light = mag_vec3(sub_vec3(v_instance, pointlight_cam_pos))
            pointlight_intensity = 1 if dist_to_light < 1 else max(0, (1 / pointlight_falloff) * (pointlight_falloff - dist_to_light))

        quad_verts = [vec4_mat4_mul_for_points(v, model_m) for v in model["quad"]]
        cur_z = min([v[2] for v in quad_verts])
        if cur_z > 0 or cur_z < far_clip:
            return

        mip_textures = model["mip_textures"]
        mip_verts = model["mip_verts"]

        mip_level = min(len(mip_textures) - 1, math.floor(-cur_z / model["mip_dist"]))
        img = mip_textures[mip_level]
        tex_w = len(img[0])
        tex_h = len(img)
        verts = mip_verts[mip_level]

        cam_verts = [vec4_mat4_mul_for_points(v, model_m) for v in verts]
        clip_verts = [project_to_clip_space(v, persp_m) for v in cam_verts]

        scr_posns = [(int(scr_origin_x + v[0] * scr_origin_x),
                      int(scr_origin_y - v[1] * scr_origin_y)) if v is not None else None
                      for v in clip_verts]

        scene_primitives.append((
            cur_z,
            (clip_verts, scr_posns),
            (img, tex_w, tex_h, intensity, pointlight_intensity),
            DRAW_MODE_TEXTURE_RECT))
        return

    if model["model_type"] == MODEL_TYPE_ANIMATED_MESH:
        animation_name = instance["animation"]
        instance["animation_frame"] += instance["animation_speed"]
        meshes = model["animations"][animation_name]
        cur_frame = int(round(instance["animation_frame"]))
        if cur_frame >= len(meshes):
            instance["animation_frame"] = 0.0
            cur_frame = 0
        model = meshes[cur_frame]

    if "instance_normal" in instance:
        instance_normal = instance["instance_normal"]
        proj_inst_normal = vec4_mat4_mul_for_dirs((instance_normal[0], instance_normal[1], instance_normal[2], 0), model_m)
        v_instance = vec4_mat4_mul_for_points((0, 0, 0, 1), model_m)
        if (v_instance[0] * proj_inst_normal[0] + v_instance[1] * proj_inst_normal[1] + v_instance[2] * proj_inst_normal[2]) >= 0:
            return

    # Bounding spheres test
    if "sphere_bbox" in model:
        bbox_center,bbox_radius = model["sphere_bbox"]
        if not _is_bounding_sphere_in_frustum(bbox_center, bbox_radius, model_m, persp_m, near_clip, far_clip):
            return

    view_verts = list(map(lambda model_v: vec4_mat4_mul_for_points(model_v, model_m), model["verts"]))
    view_normals = list(map(lambda model_n: norm_vec3_from_vec4(vec4_mat4_mul_for_dirs(model_n, model_m)), model["normals"]))

    draw_as_wireframe = ("wireframe" in instance) and instance["wireframe"]
    no_culling = ("noCulling" in instance) and instance["noCulling"]
    model_tris = model["tris"]
    draw_gouraud_shaded = ("gouraud" in instance) and instance["gouraud"]
    textured = "texture" in model
    use_minimum_z_order = ("use_minimum_z_order" in instance) and instance["use_minimum_z_order"]
    subdivide_max_iterations = instance["subdivide_max_iterations"] if "subdivide_max_iterations" in instance else subdivide_default_max_iterations

    vert_normals = None
    if draw_gouraud_shaded:
        vert_normals = list(map(lambda model_n: norm_vec3_from_vec4(vec4_mat4_mul_for_dirs(model_n, model_m)), model["vert_normals"]))

    # This function may add temporary triangles due to clipping
    # We reset the model's lists to their original size after processing
    num_orig_model_tris = len(model_tris)
    visible_tri_idcs,orig_screen_verts = _get_visible_instance_tris(persp_m, near_clip, far_clip, model, view_verts, view_normals, vert_normals, no_culling)
    screen_verts = [(int(scr_origin_x + v_2[0] * scr_origin_x), int(scr_origin_y - v_2[1] * scr_origin_y)) if v_2 is not None else None for v_2 in orig_screen_verts]

    draw_mode = DRAW_MODE_WIREFRAME if draw_as_wireframe else (DRAW_MODE_GOURAUD if draw_gouraud_shaded else DRAW_MODE_FLAT)

    if not textured:
        model_colors = model["colors"]

    def get_pointlight_intensity(tri):
        if pointlight_enabled:
            centroid = get_vec3_triangle_centroid(view_verts[tri[0]], view_verts[tri[1]], view_verts[tri[2]])
            dist_to_light = mag_vec3(sub_vec3(centroid, pointlight_cam_pos))
            return 1 if dist_to_light < 1 else max(0, (1 / pointlight_falloff) * (pointlight_falloff - dist_to_light))
        return 0

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
            if not ignore_lighting:
                normal = view_normals[tri_idx]
                dot_prd = max(0, proj_light_dir[0] * normal[0]
                    + proj_light_dir[1] * normal[1]
                    + proj_light_dir[2] * normal[2])
                intensity = min(1, max(0, ambient + diffuse * dot_prd))
            else:
                intensity = 1

            intensity = min(1, intensity)
            textured = "texture" in model
            color_data = (textured,
                          (intensity, get_pointlight_intensity(tri)),
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

        scene_primitives.append((
            z_order,
            [screen_verts[tri[i]] for i in range(3)],
            color_data,
            draw_mode))

    # Remove temporary triangles
    if num_orig_model_tris != len(model_tris):
        del model_tris[num_orig_model_tris:]
        del model_colors[num_orig_model_tris:]

def _no_fog(z : float, color : tuple, intensity : float, pointlight_intensity : float):
    intensity = min(1, intensity + pointlight_intensity)
    color = (intensity * color[0], intensity * color[1], intensity * color[2])
    return color

def render(surface : pygame.surface.Surface, screen_area,
           scene_graph,
           camera_m, persp_m,
           render_settings):
    """Render the scene graph
    screen_area is (x,y,w,h) inside the surface
    """
    # global DEBUG_FONT
    # if DEBUG_FONT is None:
    #     DEBUG_FONT = pygame.font.Font(None, 16)

    near_clip = render_settings["near_clip"]
    far_clip = render_settings["far_clip"]
    mip_dist = render_settings["mip_dist"]

    scr_origin_x = screen_area[0] + screen_area[2] / 2
    scr_origin_y = screen_area[1] + screen_area[3] / 2

    scr_min_x = screen_area[0]
    scr_max_x = screen_area[0] + screen_area[2] - 1
    scr_min_y = screen_area[1]
    scr_max_y = screen_area[1] + screen_area[3] - 1

    # Collect all visible triangles. Elements are tuples:
    # (average z depth, screen points of triangle, lighted color, draw mode)
    # Sorted by depth before drawing, draw mode overrides order so wireframes come last
    scene_primitives = []

    proj_light_dir = get_proj_light_dir(render_settings, camera_m)

    if "fog_distance" in render_settings:
        fog_distance = render_settings["fog_distance"]
        fog_denom = fog_distance - near_clip
        fog_color = render_settings["fog_color"]
    else:
        fog_distance = 0
    def _get_color_with_fog(z : float, color : tuple, intensity : float, pointlight_intensity : float):
        intensity = min(1, intensity)
        if z <= fog_distance:
            if pointlight_intensity > 0:
                color = (pointlight_intensity * color[0], pointlight_intensity * color[1], pointlight_intensity * color[2])
                f = min(0.8, (z - near_clip) / fog_denom) # TODO Is this a good constant to use?
                neg_f = 1 - f
                return (f * fog_color[0] + neg_f * color[0],
                        f * fog_color[1] + neg_f * color[1],
                        f * fog_color[2] + neg_f * color[2])
            else:
                return fog_color
        intensity = min(1, intensity + pointlight_intensity)
        color = (intensity * color[0], intensity * color[1], intensity * color[2])
        f = min(1, (z - near_clip) / fog_denom) # if 1 then equal to fog color
        neg_f = 1 - f
        return (f * fog_color[0] + neg_f * color[0],
                f * fog_color[1] + neg_f * color[1],
                f * fog_color[2] + neg_f * color[2])

    get_color_with_fog = _no_fog if fog_distance == 0 else _get_color_with_fog

    def traverse_scene_graph(subgraph, parent_m):
        for _,instance in subgraph.items():
            if instance["enabled"]:
                proj_m = mat4_mat4_mul(instance["xform_m4"], instance["preproc_m4"])
                proj_m = mat4_mat4_mul(parent_m, proj_m)
                proj_m = mat4_mat4_mul(camera_m, proj_m)
                _get_screen_primitives_for_instance(scene_primitives, near_clip, far_clip, persp_m,
                                                    scr_origin_x, scr_origin_y, render_settings, proj_light_dir,
                                                    instance, proj_m, camera_m)
                pass_m = mat4_mat4_mul(parent_m, instance["xform_m4"])
                if instance["children"]:
                    traverse_scene_graph(instance["children"], pass_m)

    # Traverse the scene graph and build scene_triangles values
    traverse_scene_graph(scene_graph, get_unit_m4())

    # Sort triangles in ascending z order but wireframe triangles should be drawn last
    # scene_primitives.sort(key=lambda x: (1 if x[3] == DRAW_MODE_WIREFRAME else 0, x[0]), reverse=False)
    scene_primitives.sort(key=lambda x: x[0], reverse=False) # Much faster without the extra comparison
    # print(f"tris: {len(scene_triangles)} -> {[v[1] for v in scene_triangles]}")

    for z_order,points,shading_data,draw_mode in scene_primitives:
        if draw_mode == DRAW_MODE_GOURAUD:
            textured = shading_data[0]

            v_a = (points[0][0], points[0][1])
            v_b = (points[1][0], points[1][1])
            v_c = (points[2][0], points[2][1])
            tri_area = get_2d_triangle_area(v_a, v_b, v_c)
            if tri_area <= 0:
                continue

            def get_interpolated_color(colors : tuple, p : tuple):
                col_0,col_1,col_2 = colors
                u,v,w = get_barycentric_vec2(v_a, v_b, v_c, p)
                r = max(0, min(255, int(col_0[0] * u + col_1[0] * v + col_2[0] * w)))
                g = max(0, min(255, int(col_0[1] * u + col_1[1] * v + col_2[1] * w)))
                b = max(0, min(255, int(col_0[2] * u + col_1[2] * v + col_2[2] * w)))
                return (r, g, b)

            if not textured:
                subdivide_max_iterations = shading_data[2]
                colors = shading_data[1]
                avg_color = get_average_color(colors[0], colors[1], colors[2])
                col_diff = sum(abs(a-i) + abs(a-j) + abs(a-k)
                               for a,i,j,k in zip(avg_color, colors[0], colors[1], colors[2]))
                if col_diff <= 20:
                    posns = ((v_a[0], v_a[1]), (v_b[0], v_b[1]), (v_c[0], v_c[1]))
                    aapolygon(surface, posns, avg_color)
                    filled_polygon(surface, posns, avg_color)
                    continue
            else:
                intensities = shading_data[3]
                subdivide_max_iterations = shading_data[4]

            if not textured:
                if tri_area <= 10:
                    posns = ((v_a[0], v_a[1]), (v_b[0], v_b[1]), (v_c[0], v_c[1]))
                    aapolygon(surface, posns, avg_color)
                    filled_polygon(surface, posns, avg_color)
                    continue

            if textured:
                uv = shading_data[1]
                mip_textures = shading_data[2]
                tex_ip = TextureInterpolation(uv, mip_textures, z_order, mip_dist)

            if subdivide_max_iterations > 0:
                def cb_subdivide_gouraud(v_0, v_1, v_2, iteration):
                    area = tri_area / (4 ** iteration) # triangles split in 4 per iteration
                    if textured:
                        if area <= 5 or iteration == subdivide_max_iterations:
                            centroid = get_vec2_triangle_centroid(v_0, v_1, v_2)
                            x,y = centroid[0], centroid[1]
                            u,v,w = get_barycentric_vec2(v_a, v_b, v_c, (x, y))
                            color = tex_ip.get_color(u, v, w)
                            intensity = u * intensities[0] + v * intensities[1] + w * intensities[2]
                            color = (intensity * color[0], intensity * color[1], intensity * color[2])
                            posns = ((v_0[0], v_0[1]), (v_1[0], v_1[1]), (v_2[0], v_2[1]))
                            aapolygon(surface, posns, color)
                            filled_polygon(surface, posns, color)
                            return True
                    else:
                        if area <= 5 or iteration == subdivide_max_iterations:
                            c_0 = get_interpolated_color(colors, v_0)
                            c_1 = get_interpolated_color(colors, v_1)
                            c_2 = get_interpolated_color(colors, v_2)
                            avg_color = get_average_color(c_0, c_1, c_2)
                            posns = ((v_0[0], v_0[1]), (v_1[0], v_1[1]), (v_2[0], v_2[1]))
                            aapolygon(surface, posns, avg_color)
                            filled_polygon(surface, posns, avg_color)
                            return True
                subdivide_2d_triangle_x4(v_a, v_b, v_c, cb_subdivide_gouraud)
            else:
                # Per pixel Gouraud shading
                px_array = pygame.PixelArray(surface) # TODO pygbag doesn't like this
                if textured:
                    for x,y in drawing.triangle(v_a[0], v_a[1], v_b[0], v_b[1], v_c[0], v_c[1]):
                        if x < scr_min_x or x > scr_max_x or y < scr_min_y or y > scr_max_y:
                            continue
                        u,v,w = get_barycentric_vec2(v_a, v_b, v_c, (x, y))
                        color = tex_ip.get_color(u, v, w)
                        intensity = u * intensities[0] + v * intensities[1] + w * intensities[2]
                        color = (intensity * color[0], intensity * color[1], intensity * color[2])
                        px_array[x, y] = color
                else:
                    for x,y in drawing.triangle(v_a[0], v_a[1], v_b[0], v_b[1], v_c[0], v_c[1]):
                        if x < scr_min_x or x > scr_max_x or y < scr_min_y or y > scr_max_y:
                            continue
                        r,g,b = get_interpolated_color(colors, (x, y))
                        px_array[x, y] = (r << 16) | (g << 8) | b
                del px_array
        elif draw_mode == DRAW_MODE_FLAT:
            textured = shading_data[0]
            intensity,pointlight_intensity = shading_data[1]
            subdivide_max_iterations = shading_data[4]
            if textured:
                mip_textures = shading_data[2]
                uv = shading_data[3]
                v_a = (points[0][0], points[0][1])
                v_b = (points[1][0], points[1][1])
                v_c = (points[2][0], points[2][1])
                tri_area = get_2d_triangle_area(v_a, v_b, v_c)
                if tri_area > 0:
                    tex_ip = TextureInterpolation(uv, mip_textures, z_order, mip_dist)
                    def cb_subdivide(v_0, v_1, v_2, iteration):
                        area = tri_area / (2 ** iteration) # triangles split in 2 per iteration
                        if area <= 10 or iteration == subdivide_max_iterations:
                            centroid = get_vec2_triangle_centroid(v_0, v_1, v_2)
                            x,y = centroid[0], centroid[1]
                            u,v,w = get_barycentric_vec2(v_a, v_b, v_c, (x, y))
                            color = tex_ip.get_color(u, v, w)
                            color = (intensity * color[0], intensity * color[1], intensity * color[2])
                            posns = ((v_0[0], v_0[1]), (v_1[0], v_1[1]), (v_2[0], v_2[1]))
                            aapolygon(surface, posns, color)
                            filled_polygon(surface, posns, color)
                            return True
                        return False
                    subdivide_2d_triangle(v_a, v_b, v_c, cb_subdivide)
            else:
                color = shading_data[2]
                color = get_color_with_fog(z_order, color, intensity, pointlight_intensity)
                aapolygon(surface, points, color)
                filled_polygon(surface, points, color)
        elif draw_mode == DRAW_MODE_WIREFRAME:
            aapolygon(surface, points, shading_data)
        elif draw_mode == DRAW_MODE_BILLBOARD:
            scale_img, pointlight_intensity = shading_data
            if fog_distance == 0:
                surface.blit(scale_img, points)
            else:
                fog_img = pygame.Surface(scale_img.get_size()).convert_alpha()
                f = min(1, (z_order - near_clip) / fog_denom)
                f = max(0, f - pointlight_intensity)
                neg_f = 1 - f
                color = (255, 255, 255)
                fog_img.fill((f * fog_color[0] + neg_f * color[0],
                              f * fog_color[1] + neg_f * color[1],
                              f * fog_color[2] + neg_f * color[2],
                              neg_f * 255))
                scale_img.blit(fog_img, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
                scale_img.set_alpha(255 * neg_f) # TODO Can use max(K,neg_f) to show outline from far
                surface.blit(scale_img, points)

        elif draw_mode == DRAW_MODE_PARTICLE:
            surface.blit(shading_data, points)
        elif draw_mode == DRAW_MODE_DEBUG_DOT:
            pygame.draw.rect(surface, shading_data, pygame.Rect(points[0]-1, points[1]-1, 2, 2))
        elif draw_mode == DRAW_MODE_TEXTURE_RECT:
            img,cols,rows,intensity,pointlight_intensity = shading_data
            clip_verts, scr_posns = points
            for row in range(rows):
                for col in range(cols):
                    color = img[row][col]
                    i_0 = row * (cols + 1) + col
                    i_1 = i_0 + 1
                    i_2 = i_0 + 1 + cols + 1
                    i_3 = i_0 + cols + 1
                    posns = (scr_posns[i_0],
                             scr_posns[i_1],
                             scr_posns[i_2],
                             scr_posns[i_3])
                    if not any(map(lambda x: x is None, posns)):
                        cv_0 = clip_verts[i_0]
                        cv_1 = clip_verts[i_1]
                        cv_2 = clip_verts[i_2]
                        cv_3 = clip_verts[i_3]
                        if clip_space_quad_overlaps_view_frustum(cv_0,
                                                                 cv_1,
                                                                 cv_2,
                                                                 cv_3,
                                                                 near_clip, far_clip):
                            z = min(cv_0[2], cv_1[2], cv_2[2], cv_3[2])
                            color = get_color_with_fog(z, color, intensity, pointlight_intensity)
                            aapolygon(surface, posns, color)
                            filled_polygon(surface, posns, color)

def get_animated_billboard(dx, dy, dz, sx, sy, img_list):
    """Create a billboard object with several animation frames"""
    return {
        "model_type": MODEL_TYPE_BILLBOARD,
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
        "model_type": MODEL_TYPE_PARTICLES,
        "positions": [[0.0, 0.0, 0.0, 1.0] for _ in range(num_particles)],
        "enabled": [True] * num_particles,
        "img": img,
        "size": [sx, sy],
        "user_data": []
    }

def get_texture_rect(mip_textures : list,
                     quad_points_v3 : list,
                     mip_dist : float):
    """Create a rectangular texture"""
    mip_verts = []

    v_0,v_1,v_2,v_3 = quad_points_v3

    v_03 = sub_vec3(v_3, v_0)
    v_12 = sub_vec3(v_2, v_1)

    v_a = sub_vec3(v_1, v_0)
    v_b = sub_vec3(v_2, v_0)
    normal = norm_vec3(cross_vec3(v_a, v_b))

    for img in mip_textures:
        tex_w = len(img[0])
        tex_h = len(img)

        rows = tex_h
        cols = tex_w

        v_03_step = mul_vec3(1.0 / rows, v_03)
        v_12_step = mul_vec3(1.0 / rows, v_12)

        cur_verts = []
        for u in range(rows + 1):
            v_03_p = add_vec3(v_0, mul_vec3(u, v_03_step))
            v_12_p = add_vec3(v_1, mul_vec3(u, v_12_step))
            v_across_step = mul_vec3(1.0 / cols, sub_vec3(v_12_p, v_03_p))
            for v in range(cols + 1):
                p = add_vec3(v_03_p, mul_vec3(v, v_across_step))
                cur_verts.append([*p, 1.0])

        mip_verts.append(cur_verts)

    return {
        "model_type": MODEL_TYPE_TEXTURE_RECT,
        "quad": [[*v, 1.0] for v in quad_points_v3],
        "normal": [*normal, 0],
        "mip_verts": mip_verts,
        "mip_textures": mip_textures,
        "mip_dist": mip_dist,
    }
