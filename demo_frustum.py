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

# DEMO CODE

CAMERA = { "pos": [0,0,0], "rot": [0,0,0] }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}
SPRITE_SPEED = 0.1

def create_scene_graph():
    """Create the main scene graph"""
    scene_graph = { "root": rasterizer.get_model_instance(None) }
    scene_graph["root"]["children"]["tri"] = rasterizer.get_model_instance(meshes.get_test_triangle_mesh())
    # size = 11
    # start = -int(size / 2)
    # r_ch = scene_graph["root"]["children"]
    # tile_spacing = 1
    # for r_i in range(size):
    #     y_i = start + r_i
    #     for t_i in range(size):
    #         x_i = start + t_i
    #         color = (200, 0, 0) if ((r_i + t_i) % 2 == 0) else (0, 0, 200)
    #         tile = get_model_instance(get_rect_mesh((1,1), (1,1), (color, color)),
    #             get_rot_x_m4(deg_to_rad(-90)),
    #             get_transl_m4(x_i * tile_spacing, -1, y_i * tile_spacing))
    #         r_ch["tile_" + str(x_i) + "_" + str(y_i)] = tile
    return scene_graph

def draw_scene_graph(surface, frame, scene_graph):
    """Draw the scene graph"""
    # radius = 3
    # CAMERA["pos"][0] = radius * math.cos(deg_to_rad(frame))
    # CAMERA["pos"][2] = radius * math.sin(deg_to_rad(frame))
    CAMERA["pos"][0] = 0
    CAMERA["pos"][1] = 0
    CAMERA["pos"][2] = 1 + 0.2 * math.sin(vecmat.deg_to_rad(frame))
    print(CAMERA["pos"][2])
    CAMERA["rot"][0] = vecmat.deg_to_rad(0)
    rasterizer.render(surface, SCR_AREA, scene_graph, get_camera_m(CAMERA), LIGHTING)

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE)
    pygame.display.set_caption("PyRasterize")
    clock = pygame.time.Clock()

    scene_graph = create_scene_graph()

    frame = 0
    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        draw_scene_graph(screen, frame, scene_graph)

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()
