"""
Animated geometry demo
"""

import pygame
import pygame.gfxdraw

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import model_file_io

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 640, 480
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

CAMERA = { "pos": [0, 1, 3], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE, flags=pygame.SCALED)
    pygame.display.set_caption("pyrasterize demo")
    clock = pygame.time.Clock()

    scene_graph = {"root": rasterizer.get_model_instance(None)}

    meshes = {
        "idle" : model_file_io.get_animation_from_zip_file("assets/dummy-idle.zip", (1, 60)),
        "walk" : model_file_io.get_animation_from_zip_file("assets/dummy-walk.zip", (1, 180)),
        "run" : model_file_io.get_animation_from_zip_file("assets/dummy-run.zip", (1, 35)),
    }
    scene_graph["root"]["children"]["model"] = rasterizer.get_model_instance(meshes["run"][0])

    frame = 0
    done = False
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    while not done:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
        screen.fill(RGB_BLACK)

        scene_graph["root"]["xform_m4"] = vecmat.get_rot_y_m4(vecmat.deg_to_rad(frame * 1.5))

        rasterizer.render(screen, SCR_AREA, scene_graph,
            vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)

        pygame.display.flip()
        frame += 1

if __name__ == '__main__':
    main_function()