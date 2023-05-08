"""
Template for demos
"""

import math
import pygame
import pygame.gfxdraw

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes
from pyrasterize import textures

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 640, 480
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

CAMERA = { "pos": [0,0,0], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE, flags=pygame.SCALED)
    pygame.display.set_caption("pyrasterize demo")
    clock = pygame.time.Clock()

    # mip_textures = textures.get_mip_textures("assets/Mona_Lisa_64x64.png")
    # mip_textures = textures.get_mip_textures("assets/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.png")
    mip_textures = textures.get_mip_textures("assets/Terrestrial-Clouds-EQUIRECTANGULAR-0-64x32.png")
    # mip_textures.pop(0)

    scene_graph = {"root": rasterizer.get_model_instance(None)}

    # scene_graph["root"]["children"]["sprite"] = rasterizer.get_model_instance(meshes.get_test_texture_mesh(mip_textures))
    # scene_graph["root"]["children"]["sprite"]["gouraud"] = True
    # scene_graph["root"]["children"]["sprite"]["gouraud_max_iterations"] = 1

    # scene_graph["root"]["children"]["sprite"] = meshes.get_test_texture_cube_instance(mip_textures, True, 1)

    scene_graph["root"]["children"]["sprite"] = rasterizer.get_model_instance(meshes.get_sphere_mesh(1, 20, 20))
    scene_graph["root"]["children"]["sprite"]["model"]["texture"] = mip_textures
    # scene_graph["root"]["children"]["sprite"]["gouraud"] = True
    # scene_graph["root"]["children"]["sprite"]["gouraud_max_iterations"] = 0

    font = pygame.font.Font(None, 30)
    textblock_fps = font.render("", True, (0,0,0))

    frame = 0
    done = False
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
        screen.fill((0, 0, 255))

        d = 20
        m = vecmat.get_transl_m4(0, 0, -2)
        # m = vecmat.get_transl_m4(0, 0, -1 - d + d * abs(math.sin(vecmat.deg_to_rad(frame * 2))))
        m = vecmat.mat4_mat4_mul(m, vecmat.get_rot_z_m4(vecmat.deg_to_rad(frame * 1.5)))
        m = vecmat.mat4_mat4_mul(m, vecmat.get_rot_y_m4(vecmat.deg_to_rad(frame * 1.5)))
        m = vecmat.mat4_mat4_mul(m, vecmat.get_rot_x_m4(vecmat.deg_to_rad(frame * 1.5)))
        scene_graph["root"]["xform_m4"] = m

        rasterizer.render(screen, SCR_AREA, scene_graph,
            vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING, mip_dist=100)

        screen.blit(textblock_fps, (10, 10))

        pygame.display.flip()
        frame += 1
        if frame % 30 == 0:
            print(f"{clock.get_fps()} fps")
            textblock_fps = font.render(f"{round(clock.get_fps(), 1)} fps", True, (200, 200, 200))

if __name__ == '__main__':
    main_function()
