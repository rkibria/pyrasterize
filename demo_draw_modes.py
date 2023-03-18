"""
Demonstrates drawing various geometry with different shading algorithms
"""

import time

import pygame
import pygame.gfxdraw
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes
from pyrasterize import model_file_io

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 320, 240
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)
PYGAME_SCR_SIZE = (800, 600)

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
    t = time.perf_counter()
    rasterizer.render(surface, RASTER_SCR_AREA, scene_graph,
        vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)
    elapsed_time = time.perf_counter() - t
    if frame % 30 == 0:
        print(f"render time: {round(elapsed_time, 3)} s")

def main_function():
    """Main"""
    pygame.init()

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
    try:
        teapot_model = model_file_io.get_model_from_obj_file("teapot-low.obj")
        instances.append(["Utah teapot", rasterizer.get_model_instance(teapot_model, vecmat.get_scal_m4(0.1, 0.1, 0.1))])
    except:
        pass
    try:
        teapot_model = model_file_io.get_model_from_obj_file("teapot.obj")
        instances.append(["Utah teapot (hi poly)", rasterizer.get_model_instance(teapot_model, vecmat.get_scal_m4(0.1, 0.1, 0.1))])
    except:
        pass

    for name,instance in instances:
        scene_graph["root"]["children"][name] = instance
        print(f"- {name}: {len(instance['model']['tris'])} triangles")
    cur_inst = 0

    font = pygame.font.Font(None, 30)
    TEXT_COLOR = (200, 200, 230)
    textblock_drawmode = font.render("", True, TEXT_COLOR)
    textblock_model = font.render("", True, TEXT_COLOR)

    drawing_mode_names = ["Gouraud shading", "Flat shading", "Wireframe with backface culling", "Wireframe"]
    OVERLAY_DRAWING_MODE = 2
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
    do_overlay = False
    done = False
    textblock_fps = font.render("", True, TEXT_COLOR)
    paused = False

    offscreen = pygame.Surface(RASTER_SCR_SIZE)
    offscreen_2 = pygame.Surface(RASTER_SCR_SIZE)

    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_mouse_button_down(event)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    do_overlay = not do_overlay
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_b and paused:
                    frame -= 1
                elif event.key == pygame.K_n and paused:
                    frame += 1

        offscreen.fill(RGB_BLACK)
        draw_scene_graph(offscreen, frame, scene_graph)

        if do_overlay:
            offscreen_2.fill(RGB_BLACK)
            saved_draw_mode = drawing_mode
            drawing_mode = OVERLAY_DRAWING_MODE
            set_draw_mode()
            draw_scene_graph(offscreen_2, frame, scene_graph)
            drawing_mode = saved_draw_mode
            offscreen_2.set_alpha(250)
            offscreen.blit(offscreen_2, (0,0), special_flags = pygame.BLEND_RGBA_SUB)
            set_draw_mode()

        screen.blit(pygame.transform.scale(offscreen, PYGAME_SCR_SIZE), (0,0))
        screen.blit(textblock_drawmode, (30, 20))
        screen.blit(textblock_model, (30, 50))
        screen.blit(textblock_fps, (30, 80))

        pygame.display.flip()
        frame += 1 if not paused else 0
        if frame % 30 == 0:
            textblock_fps = font.render(f"{round(clock.get_fps(), 1)} fps", True, TEXT_COLOR)

if __name__ == '__main__':
    main_function()
