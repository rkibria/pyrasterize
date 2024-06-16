"""
Demonstrates movement in a first person view environment
"""

# import asyncio # PYGBAG

import time
import math
import random

import pygame
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes
from pyrasterize import textures

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 640, 480
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)

# Set up a camera that is at the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0, 1, 5], "rot": [0.0, -0.003490658503988659, 0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

# Light comes from a right, top, and back direction (over the "right shoulder")
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7,
            "pointlight_enabled": True, "pointlight": [0.5, 1, -5.2, 1], "pointlight_falloff": 2.5}


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
    move_dir = [0, 0, 0] # xyz delta relative to camera direction
    rot_dir = [0, 0] # horizontal/vertical rotation direction

    textblock_fps = None
    def update_hud():
        global CAMERA
        nonlocal textblock_fps
        pos = [round(p, 2) for p in CAMERA['pos']]
        textblock_fps = font.render(f"{pos} - {round(clock.get_fps(), 1)} fps", True, TEXT_COLOR)
    update_hud()

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    def on_mouse_button_down(event):
        """Handle mouse button down"""

    def on_mouse_movement(x, y):
        """Handle mouse movement"""
        global CAMERA
        rot = CAMERA["rot"]
        rot[0] -= vecmat.deg_to_rad(y * 0.2)
        rot[1] -= vecmat.deg_to_rad(x * 0.2)
        # limit up/down rotation around x-axis to straight up/down at most
        rot[0] = min(math.pi/2, max(-math.pi/2, rot[0]))

    def on_key_down(key):
        """"""
        if key == pygame.K_w:
            move_dir[2] = -1
            return True
        elif key == pygame.K_s:
            move_dir[2] = 1
            return True
        elif key == pygame.K_a:
            move_dir[0] = -1
            return True
        elif key == pygame.K_d:
            move_dir[0] = 1
            return True
        elif key == pygame.K_LEFT:
            rot_dir[0] = 1
            return True
        elif key == pygame.K_RIGHT:
            rot_dir[0] = -1
            return True
        elif key == pygame.K_UP:
            rot_dir[1] = 1
            return True
        elif key == pygame.K_DOWN:
            rot_dir[1] = -1
            return True
        return False

    def on_key_up(key):
        """"""
        if key == pygame.K_w or key == pygame.K_s:
            move_dir[2] = 0
        elif key == pygame.K_a or key == pygame.K_d:
            move_dir[0] = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:
            rot_dir[0] = 0
        elif key == pygame.K_UP or key == pygame.K_DOWN:
            rot_dir[1] = 0

    DELTA_ROT = vecmat.deg_to_rad(3)
    def do_movement():
        """"""
        global CAMERA
        nonlocal rot_dir
        rot = CAMERA["rot"]
        rot[0] += rot_dir[1] * DELTA_ROT
        rot[1] += rot_dir[0] * DELTA_ROT

        nonlocal move_dir
        if move_dir == [0, 0, 0]:
            return
        # forward movement:
        # add vector pointing in the direction of the camera to pos.
        #
        # The camera direction for movement is in the x/z plane (y=0).
        # The relevant rotation axis is Y
        cam_rot_y = CAMERA["rot"][1]
        move_scale = 0.1
        cam_v_forward = [move_scale * math.sin(cam_rot_y), 0, move_scale * math.cos(cam_rot_y)]
        cam_pos = CAMERA["pos"]
        speed = move_dir[2]
        cam_pos[0] += cam_v_forward[0] * speed
        cam_pos[2] += cam_v_forward[2] * speed
        # strafing:
        # add vector perpendicular to camera direction to pos.
        cam_v_right = [-cam_v_forward[2], 0, cam_v_forward[0]] # 90 deg rotate: (-y, x)
        speed = move_dir[0]
        cam_pos[0] -= cam_v_right[0] * speed
        cam_pos[2] -= cam_v_right[2] * speed

    cross_size = 20
    cross_width = 2
    rgb_cross = (255, 255, 255, 100)
    cross_surface = pygame.Surface((2 * cross_size, 2 * cross_size))
    pygame.draw.rect(cross_surface, rgb_cross, (cross_size - cross_width, 0, cross_width * 2, cross_size * 2))
    pygame.draw.rect(cross_surface, rgb_cross, (0, cross_size - cross_width, cross_size * 2, cross_width * 2))
    pygame.draw.rect(cross_surface, (0, 0, 0), (cross_size - 2 * cross_width, cross_size - 2 * cross_width, cross_width * 4, cross_width * 4))

    first_mouse_move = True

    while not done:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_mouse_button_down(event)
            elif event.type == pygame.KEYDOWN:
                if not on_key_down(event.key):
                    if event.key == pygame.K_ESCAPE:
                        done = True
            elif event.type == pygame.KEYUP:
                on_key_up(event.key)
            elif event.type == pygame.MOUSEMOTION:
                mouse_position = pygame.mouse.get_rel()
                if first_mouse_move:
                    first_mouse_move = False
                else:
                    on_mouse_movement(mouse_position[0], mouse_position[1])

        do_movement()

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

        screen.blit(cross_surface, (RASTER_SCR_WIDTH // 2 - cross_size, RASTER_SCR_HEIGHT // 2 - cross_size), special_flags=pygame.BLEND_RGBA_ADD)

        if frame % 30 == 0:
            update_hud()
        screen.blit(textblock_fps, (30, 30))

        pygame.display.flip()
        frame += 1 if not paused else 0
        # await asyncio.sleep(0) # PYGBAG

if __name__ == '__main__':
    # asyncio.run(main_function()) # PYGBAG
    main_function()
