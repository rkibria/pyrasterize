"""
Demonstrates mouse selection of scene graph objects
"""

import math

import pygame
import pygame.gfxdraw
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes
from pyrasterize import selecting

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 800, 600
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)
RGB_GREEN = (0, 255, 0)

CAMERA = { "pos": [0,0,0], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}
NUM_OBJECTS = 8
CUR_SELECTED = None

def create_scene_graph():
    """Create the main scene graph"""
    scene_graph = { "root": rasterizer.get_model_instance(None) }
    for i in range(NUM_OBJECTS):
        name = "object_" + str(i)
        phi = vecmat.deg_to_rad(360 / NUM_OBJECTS * i)
        placement_radius = 3
        scene_graph["root"]["children"][name] = rasterizer.get_model_instance(
            meshes.get_sphere_mesh(0.5, 10, 10),
            xform_m4=vecmat.get_transl_m4(placement_radius * math.cos(phi),
                placement_radius * math.sin(phi), 0))
        scene_graph["root"]["children"][name]["bound_sph_r"] = 1
    return scene_graph

def draw_scene_graph(surface, frame, scene_graph):
    """Draw and animate the scene graph"""
    CAMERA["pos"] = [0, 0, 8]

    scene_graph["root"]["xform_m4"] = vecmat.mat4_mat4_mul(
        vecmat.get_rot_z_m4(vecmat.deg_to_rad(25 + frame * 0.5)),
        vecmat.get_rot_x_m4(vecmat.deg_to_rad(25 + frame * 0.5)))

    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    rasterizer.render(surface, SCR_AREA, scene_graph,
        vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)

# MAIN

def on_left_down(pos, scene_graph):
    """Handle left button down"""
    global CUR_SELECTED
    selection = selecting.get_selection(SCR_AREA, pos, scene_graph,
        vecmat.get_simple_camera_m(CAMERA))
    if CUR_SELECTED is not None:
        CUR_SELECTED[1]["wireframe"] = False
        CUR_SELECTED[1]["noCulling"] = False
    if selection:
        CUR_SELECTED = selection
        CUR_SELECTED[1]["wireframe"] = True
        CUR_SELECTED[1]["noCulling"] = True
    else:
        CUR_SELECTED = None

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE)
    pygame.display.set_caption("PyRasterize")
    clock = pygame.time.Clock()

    pygame.mouse.set_cursor(*pygame.cursors.broken_x)

    scene_graph = create_scene_graph()
    # font = pygame.font.Font(None, 30)

    frame = 0
    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_left_down(pygame.mouse.get_pos(), scene_graph)

        screen.fill(RGB_BLACK)

        draw_scene_graph(screen, frame, scene_graph)

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()
