"""
Demonstrates movement in a large labyrinth in first person view
"""

# import asyncio # PYGBAG

import time
import math

import pygame
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes

from labyrinth_gen import make_labyrinth, labyrinth_to_string, WALL_NORTH, WALL_SOUTH, WALL_EAST, WALL_WEST

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 800, 600
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)

# Set up a camera that is at the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0, 1, 0], "rot": [0, 0, 0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

# Light comes from a right, top, and back direction (over the "right shoulder")
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}


def main_function(): # PYGBAG: decorate with 'async'
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(RASTER_SCR_SIZE)
    pygame.display.set_caption("pyrasterize first person demo")
    clock = pygame.time.Clock()

    # Generate the labyrinth
    lab_rows = 5
    lab_cols = 5
    labyrinth = make_labyrinth(lab_rows, lab_cols, 20)
    print(labyrinth_to_string(labyrinth))
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(labyrinth)

    # Use separate scene graphs for ground and other objects to avoid problems with overlapping
    scene_graphs = [
        { "root": rasterizer.get_model_instance(None) },
        { "root": rasterizer.get_model_instance(None) }
    ]

    # Ground and ceiling graph

    # Each labyrinth cell's area is a cube with this side length
    cell_size = 3
    # Height of cell walls
    cell_height = 3

    scene_graphs[0]["root"]["children"]["ground"] = rasterizer.get_model_instance(
        meshes.get_rect_mesh((lab_cols * cell_size, lab_rows * cell_size), (1, 1), ((100, 100, 100), (0, 0, 0))),
        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(lab_cols * cell_size / 2, 0, -lab_cols * cell_size / 2),
                             vecmat.get_rot_x_m4(vecmat.deg_to_rad(-90))))

    # Interior: walls
    wall_color_1 = (130, 130, 140)
    wall_color_2 = (120, 120, 120)
    wall_mesh = meshes.get_rect_mesh((cell_size, cell_height), (1, 1), (wall_color_1, wall_color_1))

    cells = labyrinth["cells"]
    for row in range(lab_rows):
        row_cells = cells[row]
        for col in range(lab_cols):
            cell = row_cells[col]
            cell_name = f"cell_{row}_{col}"
            scene_graphs[1]["root"]["children"][cell_name] = rasterizer.get_model_instance(None,
                xform_m4=vecmat.get_transl_m4(cell_size * col, 0, -cell_size * (lab_rows - 1 - row)))
            cell_inst = scene_graphs[1]["root"]["children"][cell_name]

            # cell_inst["children"]["test_cube"] = rasterizer.get_model_instance(meshes.get_cube_mesh(), vecmat.get_scal_m4(0.1, 0.1, 0.1))

            if cell[WALL_NORTH]:
                cell_inst["children"]["north_wall"] = rasterizer.get_model_instance(
                    wall_mesh,
                    vecmat.get_transl_m4(cell_size / 2, cell_height / 2, -cell_size))
            if cell[WALL_SOUTH]:
                cell_inst["children"]["south_wall"] = rasterizer.get_model_instance(
                    wall_mesh,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(cell_size / 2, cell_height / 2, 0),
                                            vecmat.get_rot_y_m4(vecmat.deg_to_rad(180))))
            if cell[WALL_WEST]:
                cell_inst["children"]["west_wall"] = rasterizer.get_model_instance(
                    wall_mesh,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, cell_height / 2, -cell_size / 2),
                                            vecmat.get_rot_y_m4(vecmat.deg_to_rad(90))))
            if cell[WALL_EAST]:
                cell_inst["children"]["east_wall"] = rasterizer.get_model_instance(
                    wall_mesh,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(cell_size, cell_height / 2, -cell_size / 2),
                                            vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90))))

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
        return False

    def on_key_up(key):
        """"""
        if key == pygame.K_w or key == pygame.K_s:
            move_dir[2] = 0
        elif key == pygame.K_a or key == pygame.K_d:
            move_dir[0] = 0

    def do_movement():
        """"""
        global CAMERA
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

        persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
        t = time.perf_counter()
        for scene_graph in scene_graphs:
            rasterizer.render(screen, RASTER_SCR_AREA, scene_graph,
                vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)
        elapsed_time = time.perf_counter() - t
        if frame % 30 == 0:
            print(f"render time: {round(elapsed_time, 3)} s")

        screen.blit(cross_surface, (RASTER_SCR_WIDTH // 2 - cross_size, RASTER_SCR_HEIGHT // 2 - cross_size), special_flags=pygame.BLEND_RGBA_ADD)

        if frame % 3 == 0:
            update_hud()
        screen.blit(textblock_fps, (30, 30))

        pygame.display.flip()
        frame += 1 if not paused else 0
        # await asyncio.sleep(0) # PYGBAG

if __name__ == '__main__':
    # asyncio.run(main_function()) # PYGBAG
    main_function()
