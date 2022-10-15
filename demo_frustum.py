"""
Filled polygons with simple lighting rasterizer demo using pygame
"""

import math
import pygame

import vecmat
import rasterizer
import meshes

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 800, 600
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

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
    scene_graph["root"]["children"]["cube"] = rasterizer.get_model_instance(meshes.get_cube_mesh())
    # scene_graph["root"]["children"]["cube"]["wireframe"] = True
    # scene_graph["root"]["children"]["cube"]["noCulling"] = True
    return scene_graph

def draw_scene_graph(surface, frame, scene_graph):
    """Draw the scene graph"""
    # radius = 3
    # CAMERA["pos"][0] = radius * math.cos(deg_to_rad(frame))
    # CAMERA["pos"][2] = radius * math.sin(deg_to_rad(frame))
    CAMERA["pos"][0] = 0
    CAMERA["pos"][1] = 0
    CAMERA["pos"][2] = 2
    CAMERA["rot"][0] = vecmat.deg_to_rad(0)

    angle = 0.4 * vecmat.deg_to_rad(frame)
    cube_m = vecmat.get_rot_x_m4(angle)
    cube_m = vecmat.mat4_mat4_mul(vecmat.get_rot_y_m4(angle * 0.6), cube_m)
    cube_m = vecmat.mat4_mat4_mul(vecmat.get_rot_z_m4(angle * 0.4), cube_m)
    # cube_m = vecmat.mat4_mat4_mul(vecmat.get_transl_m4(1 * math.sin(angle*2), 0, 0), cube_m)
    scene_graph["root"]["children"]["cube"]["xform_m4"] = cube_m

    CAMERA["fov"] = 90 + 30 * math.sin(vecmat.deg_to_rad(frame))

    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])

    rasterizer.render(surface, SCR_AREA, scene_graph, get_camera_m(CAMERA), persp_m, LIGHTING)

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE)
    pygame.display.set_caption("PyRasterize")
    clock = pygame.time.Clock()

    scene_graph = create_scene_graph()

    font = pygame.font.Font(None, 30)

    frame = 0
    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        draw_scene_graph(screen, frame, scene_graph)

        screen.blit(font.render(f"FOV: {float(int(CAMERA['fov'] * 10))/10}", True, RGB_WHITE), (30, 20))

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()
