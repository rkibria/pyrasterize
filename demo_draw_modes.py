"""
Demonstrates drawing various geometry with different shading algorithms
"""

import math

import pygame
import pygame.gfxdraw
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 320, 240
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)

# Set up a camera that is a little back from the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0,0,3], "rot": [0,0,0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

# Light comes from a right, top, and back direction (over the "right shoulder")
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}

def draw_scene_graph(surface, frame, scene_graph):
    """Draw and animate the scene graph"""
    # Set the transformation matrix of the root element to a combination of x/y/z rotations
    # This will also rotate all its children, i.e. the cube
    scene_graph["root"]["xform_m4"] = vecmat.mat4_mat4_mul(
        vecmat.get_rot_z_m4(vecmat.deg_to_rad(frame * 1.5)),
        vecmat.mat4_mat4_mul(
            vecmat.get_rot_y_m4(vecmat.deg_to_rad(frame * 1.5)),
            vecmat.get_rot_x_m4(vecmat.deg_to_rad(frame * 1.5))))
    # Get perspective matrix and render the scene
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    rasterizer.render(surface, RASTER_SCR_AREA, scene_graph,
        vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)

def main_function():
    """Main"""
    pygame.init()

    PYGAME_SCR_SIZE = (800, 600)
    screen = pygame.display.set_mode(PYGAME_SCR_SIZE)
    pygame.display.set_caption("pyrasterize drawing modes demo")
    clock = pygame.time.Clock()

    pygame.mouse.set_cursor(*pygame.cursors.broken_x)

    # The scene graph's top element is the "root" element which has no geometry of its own
    scene_graph = { "root": rasterizer.get_model_instance(None) }

    # The root has the various geometric objects as children
    instances = [
        ["Cube", rasterizer.get_model_instance(meshes.get_cube_mesh())],
        ["Sphere", rasterizer.get_model_instance(meshes.get_sphere_mesh(1, 20, 10))],
    ]
    for name,model in instances:
        scene_graph["root"]["children"][name] = model
    cur_inst = 0

    font = pygame.font.Font(None, 30)
    TEXT_COLOR = (200, 200, 230)
    textblock_drawmode = font.render("", True, TEXT_COLOR)
    textblock_model = font.render("", True, TEXT_COLOR)

    drawing_mode_names = ["Gouraud shading", "Flat shading", "Wireframe with backface culling", "Wireframe"]
    drawing_mode = 0

    def regenerate_textblocks():
        nonlocal textblock_drawmode
        nonlocal textblock_model
        textblock_drawmode = font.render(f"Draw mode (left button toggles): {drawing_mode_names[drawing_mode]}", True, TEXT_COLOR)
        textblock_model = font.render(f"Model (right button toggles): {instances[cur_inst][0]}", True, TEXT_COLOR)

    def set_draw_mode():
        """Set the cube instance's drawing parameters according to current mode"""
        nonlocal drawing_mode
        nonlocal instances
        nonlocal cur_inst
        for i in range(len(instances)):
            instances[i][1]["enabled"] = (i == cur_inst)
        instances[cur_inst][1]["gouraud"] = (drawing_mode == 0)
        instances[cur_inst][1]["wireframe"] = (drawing_mode == 2 or drawing_mode == 3)
        instances[cur_inst][1]["noCulling"] = (drawing_mode == 3)

    def on_mouse_button_down(event):
        """Handle mouse button down"""
        if event.button == 1:
            nonlocal drawing_mode
            drawing_mode = drawing_mode + 1 if drawing_mode < 3 else 0
        elif event.button == 3:
            nonlocal cur_inst
            cur_inst = cur_inst + 1 if cur_inst < (len(instances) - 1) else 0
        set_draw_mode()
        regenerate_textblocks()

    set_draw_mode()
    regenerate_textblocks()

    frame = 0
    done = False
    textblock_fps = font.render("", True, TEXT_COLOR)

    offscreen = pygame.Surface(RASTER_SCR_SIZE)

    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_mouse_button_down(event)

        offscreen.fill(RGB_BLACK)
        draw_scene_graph(offscreen, frame, scene_graph)

        screen.blit(pygame.transform.scale(offscreen, PYGAME_SCR_SIZE), (0,0))
        screen.blit(textblock_drawmode, (30, 20))
        screen.blit(textblock_model, (30, 50))
        screen.blit(textblock_fps, (30, 80))

        pygame.display.flip()
        frame += 1
        if frame % 30 == 0:
            textblock_fps = font.render(f"{round(clock.get_fps(), 1)} fps", True, TEXT_COLOR)

if __name__ == '__main__':
    main_function()