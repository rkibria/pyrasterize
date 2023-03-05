#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D rasterizer engine
"""

import pygame

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
    return {
        "enabled" : True,
        "model": model,
        "preproc_m4": preproc_m4,
        "xform_m4": xform_m4,
        "children": children
        }

DRAW_MODE_WIREFRAME = 0
DRAW_MODE_FLAT = 1
DRAW_MODE_GOURAUD = 2

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

    # Collect all visible triangles. Elements are tuples:
    # (average z depth, screen points of triangle, lighted color, draw mode)
    # Sorted by depth before drawing, draw mode overrides order so wireframes come last
    scene_triangles = []

    def project_to_screen(view_v):
        minus_z = -view_v[2]
        if minus_z == 0:
            return None
        else:
            screen_v = vecmat.vec4_mat4_mul(view_v, persp_m)
            return [screen_v[0]/minus_z, screen_v[1]/minus_z]

    near_clip = -0.5
    far_clip = -100

    def get_visible_instance_tris(tris, view_verts, no_culling, get_vert_normals):
        """
        Compute the triangles we can see, i.e. are not back facing or outside view frustum
        Also returns triangle normals and screen projections of all vertices
        Returns (
            [indices of visible triangles],
            [normals of all tris],
            [screen verts of visible tris])
        """
        visible_tri_idcs = []
        tri_normals = []
        screen_verts = list(map(project_to_screen, view_verts))
        # Sum the normals that touch the respective vertex and normalize at the end
        sum_normals = [[0, 0, 0] for _ in range(len(view_verts))] if get_vert_normals else None

        i = -1
        for tri in tris:
            i += 1

            i_0 = tri[0]
            i_1 = tri[1]
            i_2 = tri[2]

            v_0 = view_verts[i_0]
            v_1 = view_verts[i_1]
            v_2 = view_verts[i_2]

            # clip near/far plane
            if ( (v_0[2] >= near_clip or v_1[2] >= near_clip or v_2[2] >= near_clip)
            or (v_0[2] <= far_clip  or v_1[2] <= far_clip  or v_2[2] <= far_clip)):
                tri_normals.append(None)
                continue

            # clip left/right/top/bottom
            sv_0 = screen_verts[i_0]
            sv_1 = screen_verts[i_1]
            sv_2 = screen_verts[i_2]
            one_scr_v_visible = (
                   ((sv_0[0] >= -1 and sv_0[0] <= 1) and (sv_0[1] >= -1 and sv_0[1] <= 1))
                or ((sv_1[0] >= -1 and sv_1[0] <= 1) and (sv_1[1] >= -1 and sv_1[1] <= 1))
                or ((sv_2[0] >= -1 and sv_2[0] <= 1) and (sv_2[1] >= -1 and sv_2[1] <= 1)))
            if not one_scr_v_visible:
                tri_normals.append(None)
                continue

            # normal = cross_product(v_1 - v_0, v_2 - v_0)
            sub10 = (v_1[0] - v_0[0], v_1[1] - v_0[1], v_1[2] - v_0[2])
            sub20 = (v_2[0] - v_0[0], v_2[1] - v_0[1], v_2[2] - v_0[2])
            normal = (sub10[1]*sub20[2] - sub10[2]*sub20[1],
                sub10[2]*sub20[0] - sub10[0]*sub20[2],
                    sub10[0]*sub20[1] - sub10[1]*sub20[0])
            tri_normals.append(normal)

            if get_vert_normals:
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

            # Back-face culling: visible if dot_product(v_0, normal) < 0
            if no_culling or (v_0[0] * normal[0] + v_0[1] * normal[1] + v_0[2] * normal[2]) < 0:
                visible_tri_idcs.append(i)

        vert_normals = None
        if get_vert_normals:
            vert_normals = list(map(vecmat.norm_vec3, sum_normals))

        return (visible_tri_idcs, tri_normals, screen_verts, vert_normals)

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
        draw_gouraud_shaded = ("gouraud" in instance) and instance["gouraud"]

        visible_tri_idcs,tri_normals,screen_verts,vert_normals = get_visible_instance_tris(model_tris,
                                                                                           view_verts,
                                                                                           no_culling,
                                                                                           draw_gouraud_shaded)
        screen_verts = [(int(scr_origin_x + v_2[0] * scr_origin_x), int(scr_origin_y - v_2[1] * scr_origin_y)) for v_2 in screen_verts]

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
                normal = vecmat.norm_vec3(tri_normals[tri_idx])
                color = model_colors[tri_idx]
                dot_prd = max(0, proj_light_dir[0] * normal[0]
                    + proj_light_dir[1] * normal[1]
                    + proj_light_dir[2] * normal[2])
                intensity = min(1, max(0, ambient + diffuse * dot_prd))
                color_data = (intensity * color[0], intensity * color[1], intensity * color[2])
            else: # draw_mode == DRAW_MODE_GOURAUD
                color_data = [vert_colors[vert_idx] for vert_idx in tri]

            scene_triangles.append((
                (view_verts[tri[0]][2] + view_verts[tri[1]][2] + view_verts[tri[2]][2]) / 3,
                [screen_verts[tri[i]] for i in range(3)],
                color_data,
                draw_mode))

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

    def area(a, b, c):
        """Area of triangle formed by 3 vec3s"""
        t = vecmat.cross_vec3(vecmat.sub_vec3(b, a), vecmat.sub_vec3(c, a))
        return 0.5 * vecmat.mag_vec3(t)

    px_array = pygame.PixelArray(surface)
    for _,points,color_data,draw_mode in scene_triangles:
        if draw_mode == DRAW_MODE_GOURAUD:
            # color_data = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            # pygame.draw.lines(surface, (255, 0, 0), True, points)
            v_a = (points[0][0], points[0][1], 0)
            v_b = (points[1][0], points[1][1], 0)
            v_c = (points[2][0], points[2][1], 0)

            v_ab = vecmat.sub_vec3(v_b, v_a)
            v_ac = vecmat.sub_vec3(v_c, v_a)
            v_n = vecmat.cross_vec3(v_ab, v_ac)

            # color_data is one color for each vertex rather than just one for the whole triangle
            # p0 = (points[0][0], points[0][1], 0)
            # p1 = (points[1][0], points[1][1], 0)
            # p2 = (points[2][0], points[2][1], 0)
            # Compute barycentric coordinates of each point inside triangle
            # area_full = area(p0, p1, p2)
            area_full_sq = vecmat.dot_product_vec3(v_n, v_n)
            if area_full_sq > 0:
                for x,y in drawing.get_triangle_2d_points(v_a[0], v_a[1], v_b[0], v_b[1], v_c[0], v_c[1]):
                    p = (x, y, 0)
                    v_bc = vecmat.sub_vec3(v_c, v_b)
                    v_bp = vecmat.sub_vec3(p, v_b)
                    v_n1 = vecmat.cross_vec3(v_bc, v_bp)
                    u = vecmat.dot_product_vec3(v_n, v_n1) / area_full_sq
                    # a0 = area(p, p1, p2)
                    # u = a0 / area_full
                    # Check point is inside triangle
                    # point_drawn = False
                    # if (u >= 0 and u <= 1):
                    # a1 = area(p0, p, p2)
                    # v = a1 / area_full
                    v_ca = vecmat.sub_vec3(v_a, v_c)
                    v_cp = vecmat.sub_vec3(p, v_c)
                    v_n2 = vecmat.cross_vec3(v_ca, v_cp)
                    v = vecmat.dot_product_vec3(v_n, v_n2) / area_full_sq
                    w = 1 - u - v
                    if (u >= 0 and u <= 1) and (v >= 0 and v <= 1) and (w >= 0 and w <= 1):
                        rgb = (
                            min(255, color_data[0][0] * u + color_data[1][0] * v + color_data[2][0] * w),
                            min(255, color_data[0][1] * u + color_data[1][1] * v + color_data[2][1] * w),
                            min(255, color_data[0][2] * u + color_data[1][2] * v + color_data[2][2] * w),)
                        px_array[x, y] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                        point_drawn = True
                    if not point_drawn:
                        px_array[x, y] = (255, 255, 255)
            # else: # happens rarely
            #     pygame.draw.polygon(surface, (0, 255, 0), points)
        elif draw_mode == DRAW_MODE_FLAT:
            pygame.draw.polygon(surface, color_data, points)
        elif draw_mode == DRAW_MODE_WIREFRAME:
            pygame.draw.lines(surface, color_data, True, points)
    del px_array

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
