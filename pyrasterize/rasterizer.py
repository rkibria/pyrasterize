#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D rasterizer engine
"""

import pygame

DEBUG_FONT = None

from . import vecmat
from . import drawing

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

    if model is not None and "normals" not in model:
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

        model["normals"] = normals
        vert_normals = list(map(vecmat.norm_vec3, sum_normals))
        model["vert_normals"] = vert_normals

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


def project_model_vert_to_view(model_v, model_m):
    """
    Model vertices are vec3 so need a conversion
    Return vec4
    """
    return vecmat.vec4_mat4_mul((model_v[0], model_v[1], model_v[2], 1), model_m)


def project_normal_to_view(model_n, model_m):
    """
    Gets normal vector transformed to current view
    Returns vec3
    """
    normal_vec4 = (model_n[0], model_n[1], model_n[2], 0)
    return vecmat.norm_vec3(vecmat.vec4_mat4_mul(normal_vec4, model_m)[0:3])


DRAW_MODE_WIREFRAME = 0
DRAW_MODE_FLAT = 1
DRAW_MODE_GOURAUD = 2

def render(surface, screen_area, scene_graph, camera_m, persp_m, lighting, billboards, near_clip=-0.5):
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

    ambient = lighting["ambient"]
    diffuse = lighting["diffuse"]

    def get_proj_light_dir():
        light_dir_vec3 = lighting["lightDir"]
        light_dir_vec4 = (light_dir_vec3[0], light_dir_vec3[1], light_dir_vec3[2], 0)
        return vecmat.norm_vec3(vecmat.vec4_mat4_mul(light_dir_vec4, camera_m)[0:3])

    proj_light_dir = get_proj_light_dir()

    # Collect all visible triangles. Elements are tuples:
    # (average z depth, screen points of triangle, lighted color, draw mode)
    # Sorted by depth before drawing, draw mode overrides order so wireframes come last
    scene_triangles = []

    def project_to_screen(view_v):
        """
        Takes vec4
        Returns vec2 or None
        """
        minus_z = -view_v[2]
        if minus_z == 0:
            return None
        else:
            screen_v = vecmat.vec4_mat4_mul(view_v, persp_m)
            return [screen_v[0]/minus_z, screen_v[1]/minus_z]

    def get_visible_instance_tris(model, view_verts, view_normals, vert_normals, no_culling):
        """
        Compute the triangles we can see, i.e. are not back facing or outside view frustum
        - Also returns screen projections of all vertices
        - Also may create new triangles due to clipping.
        Returns: ([indices of visible triangles], [screen verts of visible tris])
        SIDE EFFECTS: model's tris and colors get new entries which should be removed immediately!
        """
        visible_tri_idcs = []
        screen_verts = list(map(project_to_screen, view_verts))

        # May add new triangles as we loop
        tris = model["tris"]
        colors = model["colors"]
        num_orig_tris = len(tris)
        for tri_idx in range(num_orig_tris):
            tri = tris[tri_idx]

            i_0 = tri[0]
            i_1 = tri[1]
            i_2 = tri[2]

            v_0 = view_verts[i_0]
            v_1 = view_verts[i_1]
            v_2 = view_verts[i_2]

            # clip left/right/top/bottom
            sv_0 = screen_verts[i_0]
            sv_1 = screen_verts[i_1]
            sv_2 = screen_verts[i_2]

            if sv_0 is None or sv_1 is None or sv_2 is None:
                continue

            v_0_behind = v_0[2] > near_clip
            v_1_behind = v_1[2] > near_clip
            v_2_behind = v_2[2] > near_clip
            num_behind = (1 if v_0_behind else 0) + (1 if v_1_behind else 0) + (1 if v_2_behind else 0)
            if num_behind == 3:
                # Clip triangles totally behind near clip plane
                continue

            # Back-face culling: visible if dot_product(v_0, normal) < 0
            normal = view_normals[tri_idx]
            if not (no_culling or (v_0[0] * normal[0] + v_0[1] * normal[1] + v_0[2] * normal[2]) < 0):
                continue

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
                screen_verts.append(project_to_screen(new_back_1))
                view_verts.append(new_back_2)
                screen_verts.append(project_to_screen(new_back_2))
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
                screen_verts.append(project_to_screen(new_front_1))
                view_verts.append(new_front_2)
                screen_verts.append(project_to_screen(new_front_2))
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

        return (visible_tri_idcs, screen_verts)

    def get_screen_tris_for_instance(instance, model_m):
        """Get (lighted) triangles from this instance and insert them into scene_triangles"""
        model = instance["model"]
        if not model:
            return

        view_verts = list(map(lambda x: project_model_vert_to_view(x, model_m), model["verts"]))
        view_normals = list(map(lambda x: project_normal_to_view(x, model_m), model["normals"]))

        draw_as_wireframe = ("wireframe" in instance) and instance["wireframe"]
        no_culling = ("noCulling" in instance) and instance["noCulling"]
        model_colors = model["colors"]
        model_tris = model["tris"]
        draw_gouraud_shaded = ("gouraud" in instance) and instance["gouraud"]

        vert_normals = None
        if draw_gouraud_shaded:
            vert_normals = list(map(lambda x: project_normal_to_view(x, model_m), model["vert_normals"]))

        # This function may add temporary triangles due to clipping
        # We reset the model's lists to their original size after processing
        num_orig_model_tris = len(model_tris)
        visible_tri_idcs,screen_verts = get_visible_instance_tris(model, view_verts, view_normals, vert_normals, no_culling)
        screen_verts = [(int(scr_origin_x + v_2[0] * scr_origin_x), int(scr_origin_y - v_2[1] * scr_origin_y)) if v_2 is not None else None for v_2 in screen_verts]

        draw_mode = DRAW_MODE_WIREFRAME if draw_as_wireframe else (DRAW_MODE_GOURAUD if draw_gouraud_shaded else DRAW_MODE_FLAT)

        # Compute colors for each required vertex for Gouraud shading
        vert_colors = [None] * len(view_verts)
        if draw_gouraud_shaded:
            for tri_idx in visible_tri_idcs:
                tri_color = model_colors[tri_idx]
                tri = model_tris[tri_idx]
                for vert_idx in tri:
                    if vert_colors[vert_idx] is None:
                        normal = vert_normals[vert_idx]
                        dot_prd = max(0, proj_light_dir[0] * normal[0]
                            + proj_light_dir[1] * normal[1]
                            + proj_light_dir[2] * normal[2])
                        intensity = min(1, max(0, ambient + diffuse * dot_prd))
                        vert_colors[vert_idx] = (intensity * tri_color[0], intensity * tri_color[1], intensity * tri_color[2])

        for tri_idx in visible_tri_idcs:
            tri = model_tris[tri_idx]

            if draw_mode == DRAW_MODE_WIREFRAME:
                color_data = model_colors[tri_idx]
            elif draw_mode == DRAW_MODE_FLAT:
                color = model_colors[tri_idx]
                normal = view_normals[tri_idx]
                dot_prd = max(0, proj_light_dir[0] * normal[0]
                    + proj_light_dir[1] * normal[1]
                    + proj_light_dir[2] * normal[2])
                intensity = min(1, max(0, ambient + diffuse * dot_prd))
                color_data = (intensity * color[0], intensity * color[1], intensity * color[2])
            else: # draw_mode == DRAW_MODE_GOURAUD
                color_data = [vert_colors[vert_idx] for vert_idx in tri]

            z_order = min(view_verts[tri[0]][2], view_verts[tri[1]][2], view_verts[tri[2]][2])
            scene_triangles.append((
                z_order,
                [screen_verts[tri[i]] for i in range(3)],
                color_data,
                draw_mode))

        # Remove temporary triangles
        if num_orig_model_tris != len(model_tris):
            del model_tris[num_orig_model_tris:]
            del model_colors[num_orig_model_tris:]

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

    # Traverse the scene graph and build scene_triangles values
    traverse_scene_graph(scene_graph, vecmat.get_unit_m4())

    # Sort triangles in ascending z order but wireframe triangles should be drawn last
    scene_triangles.sort(key=lambda x: (1 if x[3] == DRAW_MODE_WIREFRAME else 0, x[0]), reverse=False)

    px_array = None
    for _,points,color_data,draw_mode in scene_triangles:
        if draw_mode == DRAW_MODE_GOURAUD:
            if px_array is None:
                px_array = pygame.PixelArray(surface) # TODO pygbag doesn't like this

            v_a = (points[0][0], points[0][1])
            v_b = (points[1][0], points[1][1])
            v_c = (points[2][0], points[2][1])

            # v_ab = vecmat.sub_vec3(v_b, v_a)
            v_ab_0 = v_b[0] - v_a[0]
            v_ab_1 = v_b[1] - v_a[1]
            # v_ac = vecmat.sub_vec3(v_c, v_a)
            v_ac_0 = v_c[0] - v_a[0]
            v_ac_1 = v_c[1] - v_a[1]
            # v_n = vecmat.cross_vec3(v_ab, v_ac)
            v_n = v_ab_0 * v_ac_1 - v_ab_1 * v_ac_0
            # area_full_sq = vecmat.dot_product_vec3(v_n, v_n)
            area_full_sq = v_n * v_n

            if area_full_sq > 0:
                # p = (x, y, 0)
                # v_bc = vecmat.sub_vec3(v_c, v_b)
                v_bc_0 = v_c[0] - v_b[0]
                v_bc_1 = v_c[1] - v_b[1]
                # v_ca = vecmat.sub_vec3(v_a, v_c)
                v_ca_0 = v_a[0] - v_c[0]
                v_ca_1 = v_a[1] - v_c[1]
                for x,y in drawing.triangle(v_a[0], v_a[1], v_b[0], v_b[1], v_c[0], v_c[1]):
                    if x < scr_min_x or x > scr_max_x or y < scr_min_y or y > scr_max_y:
                        continue
                    # v_bp = vecmat.sub_vec3(p, v_b)
                    v_bp_0 = x - v_b[0]
                    v_bp_1 = y - v_b[1]
                    # v_n1 = vecmat.cross_vec3(v_bc, v_bp)
                    v_n1 = v_bc_0 * v_bp_1 - v_bc_1 * v_bp_0
                    # u = vecmat.dot_product_vec3(v_n, v_n1) / area_full_sq
                    u = (v_n * v_n1) / area_full_sq

                    # v_cp = vecmat.sub_vec3(p, v_c)
                    v_cp_0 = x - v_c[0]
                    v_cp_1 = y - v_c[1]
                    # v_n2 = vecmat.cross_vec3(v_ca, v_cp)
                    v_n2 = v_ca_0 * v_cp_1 - v_ca_1 * v_cp_0
                    # v = vecmat.dot_product_vec3(v_n, v_n2) / area_full_sq
                    v = (v_n * v_n2) / area_full_sq

                    w = 1 - u - v
                    r = max(0, min(255, int(color_data[0][0] * u + color_data[1][0] * v + color_data[2][0] * w)))
                    g = max(0, min(255, int(color_data[0][1] * u + color_data[1][1] * v + color_data[2][1] * w)))
                    b = max(0, min(255, int(color_data[0][2] * u + color_data[1][2] * v + color_data[2][2] * w)))
                    px_array[x, y] = (r << 16) | (g << 8) | b
        elif draw_mode == DRAW_MODE_FLAT:
            pygame.draw.polygon(surface, color_data, points)
        elif draw_mode == DRAW_MODE_WIREFRAME:
            pygame.draw.lines(surface, color_data, True, points)
    if px_array is not None:
        del px_array

    # Draw billboards
    for pos,img in billboards:
        cam_pos = vecmat.vec4_mat4_mul(pos, camera_m)
        if cam_pos[2] > near_clip:
            continue
        scr_pos = project_to_screen(cam_pos)
        if scr_pos is not None:
            scr_pos = (int(scr_origin_x + scr_pos[0] * scr_origin_x - img.get_width()/2),
                       int(scr_origin_y - scr_pos[1] * scr_origin_y - img.get_height()/2))
            surface.blit(img, scr_pos)

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
        for name,instance in subgraph.items():
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
                        selected = (name, instance)
                    else:
                        if ray_t < min_t:
                            min_t = ray_t
                            selected = (name, instance)
            else: # Only check outermost models for selection
                pass_m = vecmat.mat4_mat4_mul(parent_m, instance["xform_m4"])
                if instance["children"]:
                    traverse_scene_graph(instance["children"], pass_m)

    traverse_scene_graph(scene_graph, vecmat.get_unit_m4())
    return selected
