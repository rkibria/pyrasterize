"""
Simulates the classic shell game with 3d models
"""

import math
import random

import pygame
import pygame.gfxdraw
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes

# GAME LOGIC
#
# SHELL GAME https://en.wikipedia.org/wiki/Shell_game
#
# Shell  Shell  Shell
#   0      1      2
#
# /---\  /---\  /---\
# |   |  |   |  |   |
# |   |  |   |  |   |
#
#   O <-- Pea location
#
# Game state: pea location = 0/1/2
#
# Possible shell swaps/game operations:
# 0-1, 1-2, 0-2
# - each move can be animated clockwise or counter-clockwise,
#   but has the same result.
# - a swap swaps the position of the pea if the pea is in
#   either one of the affected shell positions.
#

SWAP_01 = 0
SWAP_12 = 1
SWAP_02 = 2

PEA_LOC = 0

SWAP_RESULT_TABLE = {
    0: {SWAP_01: 1, SWAP_12: 0, SWAP_02: 2},
    1: {SWAP_01: 0, SWAP_12: 2, SWAP_02: 1},
    2: {SWAP_01: 2, SWAP_12: 1, SWAP_02: 0},
}

def get_new_pea_loc(pea_loc, n_swap):
    """Returns new pea location depending on swap"""
    return SWAP_RESULT_TABLE[pea_loc][n_swap]

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 800, 600
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)
RGB_GREEN = (0, 255, 0)

def get_camera_m(cam):
    """Return matrix based on camera dict(rot = [x,y,z],pos = [x,y,z])"""
    cam_pos = cam["pos"]
    cam_rot = cam["rot"]
    camera_m = vecmat.get_transl_m4(-cam_pos[0], -cam_pos[1], -cam_pos[2])
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_z_m4(-cam_rot[2]), camera_m)
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_y_m4(-cam_rot[1]), camera_m)
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_x_m4(-cam_rot[0]), camera_m)
    return camera_m

CAMERA = { "pos": [0, 4, 7],
    "rot": [vecmat.deg_to_rad(-20), 0, 0],
    "fov": 90,
    "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}
CUR_SELECTED = None
SHELL_MESH = meshes.get_cylinder_mesh(2, 1, 50, (100, 100, 230), close_bottom=False)
SHELL_DIST = 2.5
PEA_RADIUS = 0.5
PEA_MESH = meshes.get_sphere_mesh(PEA_RADIUS, 6, 4, (200, 20, 20))

def create_scene_graph():
    """Create the main scene graph"""
    scene_graph = { "root": rasterizer.get_model_instance(None) }
    for i in range(3):
        scene_graph["root"]["children"]["shell_" + str(i)] = rasterizer.get_model_instance(
            SHELL_MESH,
            xform_m4=vecmat.get_transl_m4(-SHELL_DIST + i * SHELL_DIST, 0, 0))
        # scene_graph["root"]["children"][name]["wireframe"] = True
        # scene_graph["root"]["children"][name]["noCulling"] = True
    scene_graph["root"]["children"]["pea"] = rasterizer.get_model_instance(
        PEA_MESH,
        xform_m4=vecmat.get_transl_m4(-SHELL_DIST, 0, 0))

    return scene_graph

def set_shell_pos(scene_graph, n_shell, x, y, z):
    """Set cup n position"""
    inst = scene_graph["root"]["children"]["shell_" + str(n_shell)]
    inst["xform_m4"] = vecmat.get_transl_m4(x, y, z)

def rotate_shell_around_point(scene_graph, n_shell, px, pz, y, angle, radius):
    """Rotate CCW by angle around point on XZ-plane"""
    x = px + radius * math.cos(angle)
    z = pz + radius * math.sin(angle)
    set_shell_pos(scene_graph, n_shell, x, y, z)

def rotate_shell_01(scene_graph, angle):
    """Rotate 1 and 2 around mid point"""
    rotate_shell_around_point(scene_graph, 0, -SHELL_DIST/2, 0, 0, angle, SHELL_DIST/2)
    rotate_shell_around_point(scene_graph, 1, -SHELL_DIST/2, 0, 0, angle + math.pi, SHELL_DIST/2)

def rotate_shell_12(scene_graph, angle):
    """Rotate 2 and 3 around mid point"""
    rotate_shell_around_point(scene_graph, 1, SHELL_DIST/2, 0, 0, angle, SHELL_DIST/2)
    rotate_shell_around_point(scene_graph, 2, SHELL_DIST/2, 0, 0, angle + math.pi, SHELL_DIST/2)

def rotate_shell_02(scene_graph, angle):
    """Rotate 1 and 3 around mid point"""
    rotate_shell_around_point(scene_graph, 0, 0, 0, 0, angle, SHELL_DIST)
    rotate_shell_around_point(scene_graph, 2, 0, 0, 0, angle + math.pi, SHELL_DIST)

def enable_pea(scene_graph, en):
    """Set enable for drawing of pea"""
    scene_graph["root"]["children"]["pea"]["enabled"] = en

def reset_shell_positions(scene_graph):
    """Reset to default"""
    set_shell_pos(scene_graph, 0, -SHELL_DIST, 0, 0)
    set_shell_pos(scene_graph, 1, 0, 0, 0)
    set_shell_pos(scene_graph, 2, SHELL_DIST, 0, 0)

SWAP_DONE = True
CURRENT_SWAP = 0
SWAP_CLOCKWISE = 0
CURRENT_FRAME = 0

def draw_scene_graph(surface, frame, scene_graph):
    """Draw and animate the scene graph"""
    # angle = 10 * vecmat.deg_to_rad(frame)
    enable_pea(scene_graph, False)
    # set_shell_pos(scene_graph, 0, -SHELL_DIST, abs(math.sin(5 * vecmat.deg_to_rad(frame))) * 3, 0)

    global SWAP_DONE
    global CURRENT_SWAP
    global SWAP_CLOCKWISE
    global CURRENT_FRAME
    if SWAP_DONE:
        reset_shell_positions(scene_graph)
        while True:
            new_swap = random.randint(0, 2)
            if new_swap != CURRENT_SWAP:
                break
        CURRENT_SWAP = new_swap
        SWAP_CLOCKWISE = random.randint(0, 1)
        CURRENT_FRAME = 0
        SWAP_DONE = False

    degs_per_frame = 35 if CURRENT_SWAP != SWAP_02 else 30
    degs = min(180, CURRENT_FRAME * degs_per_frame)
    angle = vecmat.deg_to_rad(degs)
    angle = angle if SWAP_CLOCKWISE == 0 else -angle
    if CURRENT_SWAP == SWAP_01:
        rotate_shell_01(scene_graph, angle)
    elif CURRENT_SWAP == SWAP_02:
        rotate_shell_02(scene_graph, angle)
    elif CURRENT_SWAP == SWAP_12:
        rotate_shell_12(scene_graph, angle)
    CURRENT_FRAME += 1
    if degs >= 180:
        SWAP_DONE = True

    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    rasterizer.render(surface, SCR_AREA, scene_graph, get_camera_m(CAMERA), persp_m, LIGHTING)

# MAIN

def on_left_down(pos, scene_graph):
    """Handle left button down"""
    global CUR_SELECTED
    selection = rasterizer.get_selection(SCR_AREA, pos, scene_graph, get_camera_m(CAMERA))
    if CUR_SELECTED is not None:
        CUR_SELECTED["wireframe"] = False
        CUR_SELECTED["noCulling"] = False
    if selection:
        CUR_SELECTED = selection
        CUR_SELECTED["wireframe"] = True
        CUR_SELECTED["noCulling"] = True
    else:
        CUR_SELECTED = None

def main_function():
    """Main"""
    pygame.init()
    random.seed()

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
