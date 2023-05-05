"""
Template for demos
"""

import math
import pygame
import pygame.gfxdraw

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 640, 480
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

CAMERA = { "pos": [0,0,3], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE, flags=pygame.SCALED)
    pygame.display.set_caption("pyrasterize demo")
    clock = pygame.time.Clock()

    file_name = "assets/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.png"
    img = pygame.image.load(file_name).convert_alpha()
    tex_data = []
    for y in range(img.get_height()):
        row = []
        for x in range(img.get_width()):
            rgb = img.get_at((x, y))[:3]
            row.append(rgb)
        tex_data.append(row)
    tex_data.reverse()

    scene_graph = {"root": rasterizer.get_model_instance(None)}
    scene_graph["root"]["children"]["cube"] = rasterizer.get_model_instance(meshes.get_test_texture_mesh())
    scene_graph["root"]["children"]["cube"]["model"]["texture"] = tex_data
    scene_graph["root"]["children"]["cube"]["gouraud"] = True
    scene_graph["root"]["children"]["cube"]["gouraud_max_iterations"] = 1
    scene_graph["root"]["children"]["cube"]["textured"] = True

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
        screen.fill(RGB_BLACK)

        scale = 1 + 0.75 * math.sin(vecmat.deg_to_rad(frame))
        m = vecmat.get_scal_m4(scale, scale, scale)
        m = vecmat.mat4_mat4_mul(m, vecmat.get_rot_z_m4(vecmat.deg_to_rad(frame * 1.5)))
        m = vecmat.mat4_mat4_mul(m, vecmat.get_rot_y_m4(vecmat.deg_to_rad(frame * 1.5)))
        m = vecmat.mat4_mat4_mul(m, vecmat.get_rot_x_m4(vecmat.deg_to_rad(frame * 1.5)))
        scene_graph["root"]["xform_m4"] = m

        rasterizer.render(screen, SCR_AREA, scene_graph,
            vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()
