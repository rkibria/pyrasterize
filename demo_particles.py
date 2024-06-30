"""
Demonstrates particles
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

# Set up a camera that is a little back from the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0,0,5], "rot": [0,0,0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

render_settings = rasterizer.get_default_render_settings()

def draw_scene_graph(surface, frame, scene_graph):
    """Draw and animate the scene graph"""
    scene_graph["root"]["xform_m4"] = vecmat.mat4_mat4_mul(
        vecmat.get_rot_z_m4(vecmat.deg_to_rad(frame * 1.5)),
        vecmat.mat4_mat4_mul(
            vecmat.get_rot_y_m4(vecmat.deg_to_rad(frame * 1.5)),
            vecmat.get_rot_x_m4(vecmat.deg_to_rad(frame * 1.5))))

    # Get perspective matrix and render the scene
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    t = time.perf_counter()
    rasterizer.render(surface, RASTER_SCR_AREA, scene_graph,
        vecmat.get_simple_camera_m(CAMERA), persp_m, render_settings)
    elapsed_time = time.perf_counter() - t
    if frame % 30 == 0:
        print(f"render time: {round(elapsed_time, 3)} s")

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(RASTER_SCR_SIZE)
    pygame.display.set_caption("pyrasterize particles demo")
    clock = pygame.time.Clock()

    pygame.mouse.set_cursor(*pygame.cursors.broken_x)

    # The scene graph's top element is the "root" element which has no geometry of its own
    scene_graph = { "root": rasterizer.get_model_instance(None) }

    scene_graph["root"]["children"]["cube"] = rasterizer.get_model_instance(meshes.get_cube_mesh())

    img = pygame.image.load("assets/blue_spot.png").convert_alpha()
    l_divs = 15
    r_divs = 15
    r_phi_step = 2 * math.pi / r_divs
    l_phi_step = math.pi / l_divs
    radius = 1.5
    num_particles = (l_divs - 1) * r_divs
    particles = rasterizer.get_particles(img, num_particles, 1, 1)
    scene_graph["root"]["children"]["particles"] = rasterizer.get_model_instance(particles)
    pos_i = 0
    for l_i in range(l_divs - 1):
        for r_i in range(r_divs):
            # divide surface arc from bottom to top into l_divs
            l_phi = l_phi_step * (l_i + 1)
            y_i = -radius * math.cos(l_phi)
            radius_i = (radius ** 2 - y_i ** 2) ** 0.5
            r_phi = r_phi_step * r_i
            x_i = radius_i * math.cos(r_phi)
            z_i = -radius_i * math.sin(r_phi)
            pos = particles["positions"][pos_i]
            pos[0] = x_i
            pos[1] = y_i
            pos[2] = z_i
            pos_i += 1

    font = pygame.font.Font(None, 30)
    TEXT_COLOR = (200, 200, 230)

    frame = 0
    done = False
    textblock_fps = font.render("", True, TEXT_COLOR)

    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        screen.fill(render_settings["fog_color"])
        draw_scene_graph(screen, frame, scene_graph)
        screen.blit(textblock_fps, (30, 10))

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     textblock_fps = font.render(f"{round(clock.get_fps(), 1)} fps", True, TEXT_COLOR)

if __name__ == '__main__':
    main_function()
