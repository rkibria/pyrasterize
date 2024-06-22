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

from spritesheet import SpriteSheet

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 640, 480
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)

# Set up a camera that is at the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0.5, 1, 0.5], "rot": [0, 0, 0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

# Light comes from a right, top, and back direction (over the "right shoulder")
LIGHTING = {"lightDir": (1, 1, 1), "ambient": 0.3, "diffuse": 0.7,
            "pointlight_enabled": True, "pointlight": [12, 2, -12, 1], "pointlight_falloff": 5}

# Original mesh width for scaling
MODELS_ORIG_WIDTH = 2

FADE_DISTANCE = 15

def get_ceiling_height(cell_size):
    return 1.25 * cell_size / MODELS_ORIG_WIDTH

def create_labyrinth_floor_and_ceiling(root_instance, labyrinth, cell_size):
    """
    """
    lab_rows,lab_cols = labyrinth["size"]
    scale_factor = cell_size / MODELS_ORIG_WIDTH

    floor_model = model_file_io.get_model_from_obj_file("assets/floor_62tris.obj")
    preproc_m4 = vecmat.get_scal_m4(scale_factor, 1, scale_factor)

    ceil_model = meshes.get_rect_mesh((2, 2), (5,5))
    ceil_preproc_m4 = vecmat.get_rot_x_m4(vecmat.deg_to_rad(90))
    ceil_preproc_m4 = vecmat.mat4_mat4_mul(vecmat.get_scal_m4(scale_factor, 1, scale_factor), ceil_preproc_m4)
    ceil_preproc_m4 = vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, get_ceiling_height(cell_size), 0), ceil_preproc_m4)

    cells = labyrinth["cells"]
    for row in range(lab_rows):
        row_cells = cells[row]
        for col in range(lab_cols):
            cell = row_cells[col]
            if cell != "#":
                cell_name = f"cell_{row}_{col}"
                root_instance["children"][cell_name] = rasterizer.get_model_instance(None)
                root_instance["children"][cell_name]["children"]["floor"] = rasterizer.get_model_instance(floor_model,
                    preproc_m4=preproc_m4,
                    xform_m4=vecmat.get_transl_m4(cell_size / 2 + cell_size * col, 0, -cell_size / 2 + -cell_size * (lab_rows - 1 - row)), create_bbox=False)
                root_instance["children"][cell_name]["children"]["floor"]["fade_distance"] = FADE_DISTANCE

                root_instance["children"][cell_name]["children"]["ceiling"] = rasterizer.get_model_instance(ceil_model,
                    preproc_m4=ceil_preproc_m4,
                    xform_m4=vecmat.get_transl_m4(cell_size / 2 + cell_size * col, 0, -cell_size / 2 + -cell_size * (lab_rows - 1 - row)), create_bbox=False)
                root_instance["children"][cell_name]["children"]["ceiling"]["fade_distance"] = FADE_DISTANCE


def create_labyrinth_instances(root_instance, labyrinth, cell_size):
    lab_rows,lab_cols = labyrinth["size"]
    scale_factor = cell_size / MODELS_ORIG_WIDTH

    wall_model = model_file_io.get_model_from_obj_file("assets/wall_1_145tris.obj")
    preproc_m4 = vecmat.get_scal_m4(scale_factor, scale_factor, scale_factor)

    wall_inst = rasterizer.get_model_instance(wall_model,
        preproc_m4=preproc_m4)
    # Wall meshes are culled if not facing the camera.
    wall_inst["instance_normal"] = [0, 0, 1]
    wall_inst["fade_distance"] = FADE_DISTANCE
    wall_inst["use_minimum_z_order"] = True

    cells = labyrinth["cells"]
    for row in range(lab_rows):
        row_cells = cells[row]
        for col in range(lab_cols):
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

            # cell_inst["children"]["test_cube"] = rasterizer.get_model_instance(meshes.get_cube_mesh((255, 0, 0)), vecmat.get_scal_m4(0.1, 0.1, 0.1))

            if wall_n:
                cell_inst["children"]["wall_n"] = rasterizer.get_model_instance(None, None,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(cell_size / 2, 0, -cell_size),
                    vecmat.get_rot_y_m4(vecmat.deg_to_rad(180))),
                    {"wall": wall_inst})
            if wall_s:
                cell_inst["children"]["wall_s"] = rasterizer.get_model_instance(None, None,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(cell_size / 2, 0, 0),
                    vecmat.get_rot_y_m4(vecmat.deg_to_rad(0))),
                    {"wall": wall_inst})
            if wall_w:
                cell_inst["children"]["wall_w"] = rasterizer.get_model_instance(None, None,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, 0, -cell_size / 2),
                    vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90))),
                    {"wall": wall_inst})
            if wall_e:
                cell_inst["children"]["wall_e"] = rasterizer.get_model_instance(None, None,
                    vecmat.mat4_mat4_mul(vecmat.get_transl_m4(cell_size, 0, -cell_size / 2),
                    vecmat.get_rot_y_m4(vecmat.deg_to_rad(90))),
                    {"wall": wall_inst})

def update_viewable_area(labyrinth, cell_size, view_max, root_instances):
    """
    """
    lab_rows,lab_cols = labyrinth["size"]
    cells = labyrinth["cells"]

    def enable_cell(row, col, enable):
        cell_name = f"cell_{row}_{col}"
        for root_instance in root_instances:
            children = root_instance["children"]
            if cell_name in children:
                root_instance["children"][cell_name]["enabled"] = enable

    # Turn off everything
    for row in range(lab_rows):
        for col in range(lab_cols):
            enable_cell(row, col, False)

    def pos_to_cell(z, x):
        return [lab_rows - 1 + int(z / cell_size), int(x / cell_size)]

    cam_rot_y = CAMERA["rot"][1]
    cam_v_forward = [-math.cos(cam_rot_y), -math.sin(cam_rot_y)]

    step = cell_size / 4.0
    enables = set()
    for delta_angle in range(-60, 60, 2):
        delta_rad = vecmat.deg_to_rad(delta_angle)
        cos = math.cos(delta_rad)
        sin = math.sin(delta_rad)
        rot_forward = [cos * cam_v_forward[0] - sin * cam_v_forward[1], sin * cam_v_forward[0] + cos * cam_v_forward[1]]
        pos_zx = [CAMERA["pos"][2], CAMERA["pos"][0]]
        for _ in range(int(view_max / step)):
            pos_zx[0] += rot_forward[0] * step
            pos_zx[1] += rot_forward[1] * step
            row,col = pos_to_cell(pos_zx[0], pos_zx[1])
            if row < 0:
                break
            if col < 0:
                break
            if row >= lab_rows:
                break
            if col >= lab_cols:
                break
            if cells[row][col] == "#":
                enables.add((row, col))
                break
            enables.add((row, col))

    for row,col in enables:
        enable_cell(row, col, True)

def main_function(): # PYGBAG: decorate with 'async'
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(RASTER_SCR_SIZE, flags=pygame.SCALED)
    pygame.display.set_caption("pyrasterize first person demo")
    clock = pygame.time.Clock()

    labyrinth = {
        'cells': [
        '#################',
        '#.........#.....#',
        '#..########..####',
        '#.#.......#...#.#',
        '#.#....##.###.#.#',
        '#.#.....#.....#.#',
        '#.#####.#####.#.#',
        '#.....#.#.......#',
        '#.....#.#.......#',
        '#.....#.....#...#',
        '#.....#####.#...#',
        '#.#.#...#.#.#.#.#',
        '###.###.#.#.#.#.#',
        '#...#.....#...#.#',
        '#..############.#',
        '#...............#',
        '#################'],
        'size': (17, 17)}

    lab_rows,lab_cols = labyrinth["size"]

    # Use separate scene graphs for ground and other objects to avoid problems with overlapping
    scene_graphs = [
        { "root": rasterizer.get_model_instance(None) },
        { "root": rasterizer.get_model_instance(None) }
    ]

    cell_size = 8
    player_radius = 1

    CAMERA["pos"][0] = cell_size * 1.5
    CAMERA["pos"][1] = 2
    CAMERA["pos"][2] = -cell_size * 1.5

    create_labyrinth_floor_and_ceiling(scene_graphs[0]["root"], labyrinth, cell_size)

    # Interior: walls
    create_labyrinth_instances(scene_graphs[1]["root"], labyrinth, cell_size)

    # Projectile - only one active at any time
    projectile_billboard = rasterizer.get_billboard(0, 0, 0, 4, 4, pygame.image.load("assets/plasmball.png").convert_alpha())
    projectile_inst = rasterizer.get_model_instance(projectile_billboard)
    scene_graphs[1]["root"]["children"]["projectile"] = projectile_inst
    projectile_inst["enabled"] = False
    LIGHTING["pointlight_enabled"] = False

    # Projectile explosion - only one active at any time
    explo_ss = SpriteSheet("assets/explosion_pixelfied.png")
    explo_imgs = []
    for y in range(4):
        for x in range(4):
            explo_imgs.append(explo_ss.get_image(x * 32, y * 32, 32, 32))
    explo_billboard = rasterizer.get_animated_billboard(0, 0, 0, 16, 16, explo_imgs)
    explo_billboard["play_mode"] = rasterizer.BILLBOARD_PLAY_ONCE
    explo_inst = rasterizer.get_model_instance(explo_billboard)
    scene_graphs[1]["root"]["children"]["projectile_explo"] = explo_inst
    explo_inst["enabled"] = False

    # Skeleton
    skeleton_ss = SpriteSheet("assets/zombie_n_skeleton2.png")
    skeleton_imgs = []
    for x in range(3):
        skeleton_imgs.append(skeleton_ss.get_image(3*32 + x * 32, 0 * 64, 32, 64))
    skeleton_billboard = rasterizer.get_animated_billboard(cell_size * (1 + 0.5), 2, -cell_size * (3 + 0.5), 20, 20, skeleton_imgs)
    skeleton_billboard["frame_advance"] = 0.1
    skeleton_inst = rasterizer.get_model_instance(skeleton_billboard)
    scene_graphs[1]["root"]["children"]["skeleton"] = skeleton_inst
    skeleton_inst["fade_distance"] = FADE_DISTANCE

    # List of all enemies
    enemies = [skeleton_inst]

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
        if not projectile_inst["enabled"]:
            projectile_inst["enabled"] = True
            LIGHTING["pointlight_enabled"] = True
            projectile_inst["model"]["translate"][0] = CAMERA["pos"][0]
            projectile_inst["model"]["translate"][1] = CAMERA["pos"][1]
            projectile_inst["model"]["translate"][2] = CAMERA["pos"][2]
            dir = vecmat.vec4_mat4_mul([0.0, 0.0, -1.0, 0.0], vecmat.get_rot_x_m4(CAMERA["rot"][0]))
            dir = vecmat.vec4_mat4_mul(dir, vecmat.get_rot_y_m4(CAMERA["rot"][1]))
            f = 1
            projectile_inst["dir"] = [dir[0] * f, dir[1] * f, dir[2] * f]

    def on_mouse_movement(x, y):
        """Handle mouse movement"""
        global CAMERA
        rot = CAMERA["rot"]
        rot[0] -= vecmat.deg_to_rad(y * 0.2)
        rot[1] -= vecmat.deg_to_rad(x * 0.2)
        # limit up/down rotation around x-axis to straight up/down at most
        rot[0] = min(math.pi/2, max(-math.pi/2, rot[0]))

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

    def get_cell_pos(x, z):
        """
        Lower left corner of the map is at 0,0
        (the cell in the last row and first column)
        """
        row = lab_rows - 1 + int(z / cell_size)
        col = int(x / cell_size)
        return row, col

    def cell_to_world_pos(row, col):
        x = col * cell_size
        z = (lab_rows - 1 - row) * -cell_size
        return x,z

    def is_position_reachable(x, y, z):
        """Is this position in open air (i.e. not inside a wall)"""
        if y < 0 or y > get_ceiling_height(cell_size):
            return False

        row,col = get_cell_pos(x, z)

        if row < 0 or row >= lab_rows or col < 0 or col >= lab_cols:
            return False

        if labyrinth["cells"][row][col] == "#":
            return False

        return True

    def is_position_walkable(x, y, z, char_radius):
        if not is_position_reachable(x, y, z):
            return False

        # We are in a free cell, don't let char get closer than their radius to walls
        row,col = get_cell_pos(x, z)
        cell_x,cell_z = cell_to_world_pos(row, col)

        # Check if we are too close to any surrounding walls
        cells = labyrinth["cells"]
        # NW
        if (cells[row - 1][col - 1] == "#"):
            if x < cell_x + char_radius and z < cell_z - cell_size + char_radius:
                return False
        # N
        if (cells[row - 1][col] == "#"):
            if z < cell_z - cell_size + char_radius:
                return False
        # NE
        if (cells[row - 1][col + 1] == "#"):
            if x > cell_x + cell_size - char_radius and z < cell_z - cell_size + char_radius:
                return False
        # E
        if (cells[row][col + 1] == "#"):
            if x > cell_x + cell_size - char_radius:
                return False
        # SE
        if (cells[row + 1][col + 1] == "#"):
            if x > cell_x + cell_size - char_radius and z > cell_z - char_radius:
                return False
        # S
        if (cells[row + 1][col] == "#"):
            if z > cell_z - char_radius:
                return False
        # SW
        if (cells[row + 1][col - 1] == "#"):
            if x < cell_x + char_radius and z > cell_z - char_radius:
                return False
        # W
        if (cells[row][col - 1] == "#"):
            if x < cell_x + char_radius:
                return False

        return True

    def do_player_movement():
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
        new_pos = [cam_pos[0] + total_movement[0] * move_scale, cam_pos[2] + total_movement[2] * move_scale]

        # Prevent clipping through walls
        if is_position_walkable(new_pos[0], cam_pos[1], new_pos[1], player_radius):
            CAMERA["pos"][0] = new_pos[0]
            CAMERA["pos"][2] = new_pos[1]

        # Camera rotation
        rot_scale = 0.05
        CAMERA["rot"][0] += move_dir[3] * rot_scale
        CAMERA["rot"][1] += move_dir[4] * rot_scale
        CAMERA["rot"][0] = min(math.pi/2, max(-math.pi/2, CAMERA["rot"][0]))

    def projectile_collides_with_enemy(projectile_pos, enemy_pos):
        # For simplicity enemy collision volume is a stack of spheres
        sphere_radius = 0.5
        for i in range(3):
            sphere_pos = [enemy_pos[0], sphere_radius + i * 2 * sphere_radius, enemy_pos[2]]
            dist_sq_v = vecmat.mag_sq_vec3(vecmat.sub_vec3(sphere_pos, projectile_pos))
            if dist_sq_v <= 1:
                return True
        return False

    def do_projectile_movement():
        if projectile_inst["enabled"]:
            mdl_tr = projectile_inst["model"]["translate"]
            mdl_tr_copy = mdl_tr.copy()
            mdl_tr_copy[0] += projectile_inst["dir"][0]
            mdl_tr_copy[1] += projectile_inst["dir"][1]
            mdl_tr_copy[2] += projectile_inst["dir"][2]
            if not is_position_reachable(*mdl_tr_copy[0:3]):
                # Projectile explodes and is removed
                projectile_inst["enabled"] = False
                LIGHTING["pointlight_enabled"] = False
                explo_inst["enabled"] = True
                explo_billboard["cur_frame"] = 0
                explo_billboard["size_scale"] = 1
                explo_tr = explo_billboard["translate"]
                explo_tr[0] = mdl_tr[0]
                explo_tr[1] = mdl_tr[1]
                explo_tr[2] = mdl_tr[2]
            else:
                # Projectile moves
                mdl_tr[0] = mdl_tr_copy[0]
                mdl_tr[1] = mdl_tr_copy[1]
                mdl_tr[2] = mdl_tr_copy[2]
                pl_tr = LIGHTING["pointlight"]
                pl_tr[0] = mdl_tr_copy[0]
                pl_tr[1] = mdl_tr_copy[1]
                pl_tr[2] = mdl_tr_copy[2]
                # Collision check
                nonlocal enemies
                nonlocal projectile_billboard
                projectile_pos = projectile_billboard["translate"]
                for enemy_inst in enemies:
                    if enemy_inst["enabled"]:
                        enemy_billboard = enemy_inst["model"]
                        enemy_pos = enemy_billboard["translate"]
                        if projectile_collides_with_enemy(projectile_pos, enemy_pos):
                            projectile_inst["enabled"] = False
                            enemy_inst["enabled"] = False
                            LIGHTING["pointlight_enabled"] = False
                            explo_inst["enabled"] = True
                            explo_billboard["cur_frame"] = 0
                            explo_billboard["size_scale"] = 3
                            explo_tr = explo_billboard["translate"]
                            explo_tr[0] = projectile_pos[0]
                            explo_tr[1] = projectile_pos[1]
                            explo_tr[2] = projectile_pos[2]

    cross_size = 20
    cross_width = 2
    rgb_cross = (255, 255, 255, 100)
    cross_surface = pygame.Surface((2 * cross_size, 2 * cross_size))
    pygame.draw.rect(cross_surface, rgb_cross, (cross_size - cross_width, 0, cross_width * 2, cross_size * 2))
    pygame.draw.rect(cross_surface, rgb_cross, (0, cross_size - cross_width, cross_size * 2, cross_width * 2))
    pygame.draw.rect(cross_surface, (0, 0, 0), (cross_size - 2 * cross_width, cross_size - 2 * cross_width, cross_width * 4, cross_width * 4))

    view_max = 3 * cell_size
    near_clip = -0.5
    far_clip = -view_max
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

        do_player_movement()
        do_projectile_movement()

        screen.fill(RGB_BLACK)

        persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
        t = time.perf_counter()
        update_viewable_area(labyrinth, cell_size, view_max, [scene_graph["root"] for scene_graph in scene_graphs])
        for scene_graph in scene_graphs:
            rasterizer.render(screen, RASTER_SCR_AREA, scene_graph,
                vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING,
                near_clip, far_clip)
        elapsed_time = time.perf_counter() - t
        if frame % 60 == 0:
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
