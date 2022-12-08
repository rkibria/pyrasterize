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

CAMERA = { "pos": [0,0,0], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}
CUR_SELECTED = None
SHELL_MESH = meshes.get_cylinder_mesh(2, 1, 12, (100, 100, 230), close_bottom=False)
SHELL_DIST = 2.5
PEA_RADIUS = 0.5
PEA_MESH = meshes.get_sphere_mesh(PEA_RADIUS, 16, 12, (200, 20, 20))

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

def rotate_shell_12(scene_graph, angle):
    """Rotate 1 and 2 around mid point"""
    rotate_shell_around_point(scene_graph, 0, -SHELL_DIST/2, 0, 0, angle, SHELL_DIST/2)
    rotate_shell_around_point(scene_graph, 1, -SHELL_DIST/2, 0, 0, angle + math.pi, SHELL_DIST/2)

def rotate_shell_23(scene_graph, angle):
    """Rotate 2 and 3 around mid point"""
    rotate_shell_around_point(scene_graph, 1, SHELL_DIST/2, 0, 0, angle, SHELL_DIST/2)
    rotate_shell_around_point(scene_graph, 2, SHELL_DIST/2, 0, 0, angle + math.pi, SHELL_DIST/2)

def rotate_shell_13(scene_graph, angle):
    """Rotate 1 and 3 around mid point"""
    rotate_shell_around_point(scene_graph, 0, 0, 0, 0, angle, SHELL_DIST)
    rotate_shell_around_point(scene_graph, 2, 0, 0, 0, angle + math.pi, SHELL_DIST)

def draw_scene_graph(surface, frame, scene_graph):
    """Draw and animate the scene graph"""
    CAMERA["pos"][1] = 4
    CAMERA["pos"][2] = 7
    CAMERA["rot"][0] = vecmat.deg_to_rad(-20)

    angle = 10 * vecmat.deg_to_rad(frame)
    rotate_shell_13(scene_graph, angle)

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
