"""
Demonstrates drawing various geometry with different shading algorithms
"""

import time
import math

import pygame
import pygame.gfxdraw
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 800, 600
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)

# Set up a camera that is at the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0, 1, 0], "rot": [0, 0, 0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

# Light comes from a right, top, and back direction (over the "right shoulder")
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}

def draw_scene_graph(surface, frame, scene_graph):
    """Draw and animate the scene graph"""
    # Get perspective matrix and render the scene
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    t = time.perf_counter()
    rasterizer.render(surface, RASTER_SCR_AREA, scene_graph,
        vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)
    elapsed_time = time.perf_counter() - t
    if frame % 30 == 0:
        print(f"render time: {round(elapsed_time, 3)} s")

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(RASTER_SCR_SIZE)
    pygame.display.set_caption("pyrasterize first person demo")
    clock = pygame.time.Clock()

    pygame.mouse.set_cursor(*pygame.cursors.broken_x)

    # The scene graph's top element is the "root" element which has no geometry of its own
    scene_graph = { "root": rasterizer.get_model_instance(None) }
    scene_graph["root"]["children"]["ground"] = rasterizer.get_model_instance(
        meshes.get_rect_mesh((10, 10), (10, 10), ((255,0,0), (0,255,0))),
        vecmat.get_rot_x_m4(vecmat.deg_to_rad(-90))
        )

    font = pygame.font.Font(None, 30)
    TEXT_COLOR = (200, 200, 230)

    frame = 0
    done = False
    paused = False
    move_dir = [0, 0, 0] # xyz delta relative to camera direction

    textblock_fps = None
    def update_hud():
        global CAMERA
        nonlocal textblock_fps
        pos = [round(p, 2) for p in CAMERA['pos']]
        textblock_fps = font.render(f"pos: {pos} - mov: {move_dir} - {round(clock.get_fps(), 1)} fps", True, TEXT_COLOR)
    update_hud()

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    def on_mouse_button_down(event):
        """Handle mouse button down"""

    def on_mouse_movement(x, y):
        """Handle mouse movement"""
        global CAMERA
        CAMERA["rot"][0] -= vecmat.deg_to_rad(y * 0.2)
        CAMERA["rot"][1] -= vecmat.deg_to_rad(x * 0.2)

    def on_key_down(key):
        """"""
        if key == pygame.K_w:
            move_dir[2] = -1
            return True
        return False

    def on_key_up(key):
        """"""
        if key == pygame.K_w:
            move_dir[0] = 0
            move_dir[1] = 0
            move_dir[2] = 0

    def do_movement():
        """"""
        # forward movement:
        # add vector pointing in the direction of the camera to pos.
        # strafing:
        # add vector perpendicular to camera direction to pos.
        #
        # The camera direction for movement is in the x/z plane (y=0).
        # The relevant rotation axis is Y
        global CAMERA
        nonlocal move_dir
        if move_dir == [0, 0, 0]:
            return
        cam_rot_y = CAMERA["rot"][1]
        move_scale = 0.1
        cam_v_forward = [move_scale * math.sin(cam_rot_y), 0, move_scale * math.cos(cam_rot_y)]
        cam_pos = CAMERA["pos"]
        forward_factor = move_dir[2]
        cam_pos[0] += cam_v_forward[0] * forward_factor
        cam_pos[2] += cam_v_forward[2] * forward_factor

    while not done:
        clock.tick(30)
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
                on_mouse_movement(mouse_position[0], mouse_position[1])

        do_movement()

        screen.fill(RGB_BLACK)
        draw_scene_graph(screen, frame, scene_graph)

        if frame % 3 == 0:
            update_hud()
        screen.blit(textblock_fps, (30, 30))

        pygame.display.flip()
        frame += 1 if not paused else 0

if __name__ == '__main__':
    main_function()
