"""
Demonstrates movement through a labyrinth in first person view
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
from pyrasterize import model_file_io

from labyrinth_gen import make_labyrinth, get_blocky_labyrinth

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 800, 600
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)

# Set up a camera that is at the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0.5, 1, 0.5], "rot": [0, 0, 0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

# Light comes from a right, top, and back direction (over the "right shoulder")
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}


def create_labyrinth_floor(root_instance, labyrinth, cell_size):
    """
    """
    lab_rows,lab_cols = labyrinth["size"]

    floor_model = model_file_io.get_model_from_obj_file("assets/floor_1.obj")

    # Original mesh width is 2
    model_width = 2
    scale_factor = cell_size / model_width

    cells = labyrinth["cells"]
    for row in range(lab_rows):
        row_cells = cells[row]
        for col in range(lab_cols):
            cell = row_cells[col]
            if cell != "#":
                cell_name = f"cell_{row}_{col}"
                root_instance["children"][cell_name] = rasterizer.get_model_instance(floor_model,
                    preproc_m4=vecmat.get_scal_m4(scale_factor, 1, scale_factor),
                    xform_m4=vecmat.get_transl_m4(cell_size / 2 + cell_size * col, 0, -cell_size / 2 + -cell_size * (lab_rows - 1 - row)))


def create_labyrinth_instances(root_instance, labyrinth, cell_size):
    """
    """
    lab_rows,lab_cols = labyrinth["size"]

    wall_model = model_file_io.get_model_from_obj_file("assets/wall_1.obj")
    # Original mesh width is 2
    model_width = 2
    scale_factor = cell_size / model_width
    wall_mesh = rasterizer.get_model_instance(wall_model, preproc_m4=vecmat.get_scal_m4(scale_factor, scale_factor, scale_factor))
    # wall_mesh["baked_colors"] = True
    wall_mesh["wireframe"] = True
    wall_mesh["noCulling"] = True
    wall_mesh["instance_normal"] = [0, 0, 1]

    cells = labyrinth["cells"]
    for row in range(lab_rows):
        row_cells = cells[row]
        for col in range(lab_cols):
            # if not (col == 0 and row == lab_rows - 1):
            #     continue

            cell_name = f"cell_{row}_{col}"
            root_instance["children"][cell_name] = rasterizer.get_model_instance(None,
                xform_m4=vecmat.get_transl_m4(cell_size * col, 0, -cell_size * (lab_rows - 1 - row)))
            cell_inst = root_instance["children"][cell_name]

            wall_n = False
            wall_s = False
            wall_w = False
            wall_e = False

            cell = row_cells[col]
            if cell == "#":
                if row != 0 and cells[row - 1][col] != "#":
                    wall_n = True
                if row != lab_rows -1 and cells[row + 1][col] != "#":
                    wall_s = True
                if col != 0 and cells[row][col - 1] != "#":
                    wall_w = True
                if col != lab_cols - 1 and cells[row][col + 1] != "#":
                    wall_e = True

            cell_inst["children"]["test_cube"] = rasterizer.get_model_instance(meshes.get_cube_mesh((255, 0, 0)), vecmat.get_scal_m4(0.1, 0.1, 0.1))

            if wall_n:
                cell_inst["children"]["wall_n"] = rasterizer.get_model_instance(None, None,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(cell_size / 2, 0, -cell_size),
                    vecmat.get_rot_y_m4(vecmat.deg_to_rad(180))),
                    {"wall": wall_mesh})
            if wall_s:
                cell_inst["children"]["wall_s"] = rasterizer.get_model_instance(None, None,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(cell_size / 2, 0, 0),
                    vecmat.get_rot_y_m4(vecmat.deg_to_rad(0))),
                    {"wall": wall_mesh})
            if wall_w:
                cell_inst["children"]["wall_w"] = rasterizer.get_model_instance(None, None,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, 0, -cell_size / 2),
                    vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90))),
                    {"wall": wall_mesh})
            if wall_e:
                cell_inst["children"]["wall_e"] = rasterizer.get_model_instance(None, None,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(cell_size, 0, -cell_size / 2),
                    vecmat.get_rot_y_m4(vecmat.deg_to_rad(90))),
                    {"wall": wall_mesh})

def main_function(): # PYGBAG: decorate with 'async'
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(RASTER_SCR_SIZE)
    pygame.display.set_caption("pyrasterize first person demo")
    clock = pygame.time.Clock()

    # Generate the labyrinth
    labyrinth = get_blocky_labyrinth(make_labyrinth(3, 3, 20))
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(labyrinth)
    lab_rows,lab_cols = labyrinth["size"]

    # Use separate scene graphs for ground and other objects to avoid problems with overlapping
    scene_graphs = [
        { "root": rasterizer.get_model_instance(None) },
        { "root": rasterizer.get_model_instance(None) }
    ]

    # Ground and ceiling graph

    # Each labyrinth cell's area is a cube with an "inner" and "outer" area
    cell_size = 8

    CAMERA["pos"][0] = 0 # cell_size / 2
    CAMERA["pos"][1] = 1.5
    CAMERA["pos"][2] = 0# -(cell_size / 2)

    # scene_graphs[0]["root"]["children"]["ground"] = rasterizer.get_model_instance(
    #     meshes.get_rect_mesh((lab_cols * cell_size, lab_rows * cell_size), (lab_cols, lab_rows), ((100, 0, 0), (0, 0, 100))),
    #     vecmat.mat4_mat4_mul(vecmat.get_transl_m4(lab_cols * cell_size / 2, 0, -lab_cols * cell_size / 2),
    #                          vecmat.get_rot_x_m4(vecmat.deg_to_rad(-90))))
    create_labyrinth_floor(scene_graphs[0]["root"], labyrinth, cell_size)

    # Interior: walls
    create_labyrinth_instances(scene_graphs[1]["root"], labyrinth, cell_size)

    font = pygame.font.Font(None, 30)
    TEXT_COLOR = (200, 200, 230)

    frame = 0
    done = False
    paused = False
    # xyz delta relative to camera direction / xyz camera rotation
    move_dir = [0, 0, 0, 0, 0, 0]

    textblock_fps = None
    def update_hud():
        global CAMERA
        nonlocal textblock_fps
        pos = [round(p, 2) for p in CAMERA['pos']]
        rot = [round(vecmat.rad_to_deg(p), 2) for p in CAMERA['rot']]
        textblock_fps = font.render(f"pos: {pos} - rot(deg): {rot} - {round(clock.get_fps(), 1)} fps", True, TEXT_COLOR)
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
        # limit to 360 degrees
        # rot[0] = divmod(rot[0], 2 * math.pi)[1]
        # rot[1] = divmod(rot[1], 2 * math.pi)[1]

    # key: (index, value)
    key_moves = {
        # WASD
        pygame.K_w: (2, -1),
        pygame.K_s: (2, 1),
        pygame.K_a: (0, -1),
        pygame.K_d: (0, 1),
        # Camera rotation
        pygame.K_v: (4, 1),
        pygame.K_n: (4, -1),
        pygame.K_g: (3, 1),
        pygame.K_b: (3, -1),
    }

    def on_key_down(key):
        """"""
        if key in key_moves:
            index, value = key_moves[key]
            move_dir[index] = value
            return True
        return False

    def on_key_up(key):
        """"""
        if key in key_moves:
            index, _ = key_moves[key]
            move_dir[index] = 0

    def do_movement():
        """"""
        global CAMERA
        nonlocal move_dir
        if not any(move_dir):
            return
        # forward movement:
        # add vector pointing in the direction of the camera to pos.
        #
        # The camera direction for movement is in the x/z plane (y=0).
        # The relevant rotation axis is Y
        cam_rot_y = CAMERA["rot"][1]
        cam_v_forward = [math.sin(cam_rot_y), 0, math.cos(cam_rot_y)]
        speed = move_dir[2]
        total_movement = [0.0, 0.0, 0.0]
        total_movement[0] += cam_v_forward[0] * speed
        total_movement[2] += cam_v_forward[2] * speed
        # strafing:
        # add vector perpendicular to camera direction to pos.
        cam_v_right = [-cam_v_forward[2], 0, cam_v_forward[0]] # 90 deg rotate: (-y, x)
        speed = move_dir[0]
        total_movement[0] -= cam_v_right[0] * speed
        total_movement[2] -= cam_v_right[2] * speed
        # normalize the movement vector so moving diagonally isn't faster than straight moves
        total_movement = vecmat.norm_vec3(total_movement)
        cam_pos = CAMERA["pos"]
        move_scale = 0.2
        cam_pos[0] += total_movement[0] * move_scale
        cam_pos[2] += total_movement[2] * move_scale
        # Camera rotation
        rot_scale = 0.05
        CAMERA["rot"][0] += move_dir[3] * rot_scale
        CAMERA["rot"][1] += move_dir[4] * rot_scale

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
