"""
Demonstrates movement in a first person view environment
"""

# import asyncio # PYGBAG

import time

import pygame
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import textures
from pyrasterize.fpscontrols import FpsControls

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 640, 480
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)

# Set up a camera that is at the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0, 1, 5], "rot": [0.0, -0.003490658503988659, 0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

# Light comes from a right, top, and back direction (over the "right shoulder")
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7,
            "pointlight_enabled": True, "pointlight": [0.5, 1, -5.2, 1], "pointlight_falloff": 2.5}

FPSCONTROLS = FpsControls(RASTER_SCR_SIZE, CAMERA)

def main_function(): # PYGBAG: decorate with 'async'
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(RASTER_SCR_SIZE, pygame.SCALED)
    pygame.display.set_caption("pyrasterize first person demo")
    clock = pygame.time.Clock()

    # Use separate scene graphs for sky, ground and everything else to avoid problems with overlapping
    scene_graphs = [
        { "root": rasterizer.get_model_instance(None) },
        { "root": rasterizer.get_model_instance(None) },
        { "root": rasterizer.get_model_instance(None) }
    ]
    # sky_graph = scene_graphs[0]
    ground_graph = scene_graphs[1]
    world_graph = scene_graphs[2]

    grass_texture = textures.get_mip_textures("assets/grass_tile_16x16.png")
    grass_size = 4.0
    half_grass_sz = grass_size / 2
    grass_mesh = rasterizer.get_texture_rect(grass_texture, [(-half_grass_sz, 0, half_grass_sz), (half_grass_sz, 0, half_grass_sz),
                                                             (half_grass_sz, 0, -half_grass_sz), (-half_grass_sz, 0, -half_grass_sz)],
                                                             4)
    for row in range(-2, 3, 1):
        for col in range(-2, 3, 1):
            grass_pos = (row * grass_size, 0, col * grass_size)
            ground_graph["root"]["children"][f"grass_{row}_{col}"] = rasterizer.get_model_instance(grass_mesh,
                xform_m4=vecmat.get_transl_m4(*grass_pos))

    painting_texture = textures.get_mip_textures("assets/Mona_Lisa_64x64.png")
    painting_pos = (0, 0.5, 0)
    painting_mesh = rasterizer.get_texture_rect(painting_texture, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], 3)
    world_graph["root"]["children"]["painting"] = rasterizer.get_model_instance(painting_mesh,
        xform_m4=vecmat.get_transl_m4(*painting_pos))

    font = pygame.font.Font(None, 30)
    TEXT_COLOR = (200, 200, 230)

    frame = 0
    done = False
    paused = False

    FPSCONTROLS.update_hud(font, clock, TEXT_COLOR)

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    while not done:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
            FPSCONTROLS.on_event(event)

        FPSCONTROLS.do_movement()

        screen.fill((0, 0, 0))

        persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
        cam_m = vecmat.get_simple_camera_m(CAMERA)
        t = time.perf_counter()
        for scene_graph in scene_graphs:
            rasterizer.render(screen, RASTER_SCR_AREA, scene_graph,
                cam_m, persp_m, LIGHTING)

        # elapsed_time = time.perf_counter() - t
        # if frame % 30 == 0:
        #     print(f"render time: {round(elapsed_time, 3)} s")

        FPSCONTROLS.draw(screen)

        if frame % 60 == 0:
            FPSCONTROLS.update_hud(font, clock, TEXT_COLOR)

        pygame.display.flip()
        frame += 1 if not paused else 0
        # await asyncio.sleep(0) # PYGBAG

if __name__ == '__main__':
    # asyncio.run(main_function()) # PYGBAG
    main_function()
