"""
Filled polygons with simple lighting rasterizer demo using pygame
"""

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

def create_scene_graph():
    """Create the main scene graph"""
    scene_graph = { "root": rasterizer.get_model_instance(None) }
    scene_graph["root"]["children"]["cube_1"] = rasterizer.get_model_instance(
        meshes.get_cube_mesh(),
        xform_m4=vecmat.get_transl_m4(-1,0,0))
    scene_graph["root"]["children"]["cube_1"]["bound_sph_r"] = 1

    scene_graph["root"]["children"]["cube_2"] = rasterizer.get_model_instance(
        meshes.get_cube_mesh(),
        xform_m4=vecmat.get_transl_m4(1,0,0))
    scene_graph["root"]["children"]["cube_2"]["bound_sph_r"] = 1

    scene_graph["root"]["children"]["selected_mesh"] = rasterizer.get_model_instance(
        meshes.get_cube_mesh(RGB_GREEN),
        xform_m4=vecmat.get_transl_m4(0,0,0))
    scene_graph["root"]["children"]["selected_mesh"]["enabled"] = False
    scene_graph["root"]["children"]["selected_mesh"]["wireframe"] = True
    scene_graph["root"]["children"]["selected_mesh"]["noCulling"] = True

    return scene_graph

def draw_scene_graph(surface, frame, scene_graph):
    """Draw the scene graph"""
    CAMERA["pos"] = [0, 0, 4]
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    rasterizer.render(surface, SCR_AREA, scene_graph, get_camera_m(CAMERA), persp_m, LIGHTING)

# MAIN

def on_left_down(pos, scene_graph):
    """Handle left button down"""
    selection = rasterizer.get_selection(SCR_AREA, pos, scene_graph, get_camera_m(CAMERA))
    selected_mesh = scene_graph["root"]["children"]["selected_mesh"]
    if selection:
        selected_mesh["enabled"] = True
        selected_mesh["preproc_m4"] = selection["preproc_m4"]
        selected_mesh["xform_m4"] = selection["xform_m4"]
    else:
        selected_mesh["enabled"] = False

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
