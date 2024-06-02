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
from pyrasterize import model_file_io

from spritesheet import SpriteSheet

# CONSTANTS

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 640, 480
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)

# Set up a camera that is at the origin point, facing forward (i.e. to negative z)
CAMERA = { "pos": [0, 1, 2.5], "rot": [0, 0, 0], "fov": 90, "ar": RASTER_SCR_WIDTH/RASTER_SCR_HEIGHT }

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
    sky_graph = scene_graphs[0]
    ground_graph = scene_graphs[1]
    world_graph = scene_graphs[2]

    # Sky graph
    sky_color_1 = (98, 207, 244)
    sky_color_2 = (44, 103, 242)
    wall_divs = (1, 20)
    sky_graph["root"]["children"]["blue_sky"] = rasterizer.get_model_instance(None)
    blue_sky_instance = sky_graph["root"]["children"]["blue_sky"]
    sky_width = 11 * 5
    sky_height = 1 * 5
    blue_sky_instance["children"]["north"] = rasterizer.get_model_instance(
        meshes.get_rect_mesh((sky_width, sky_height), wall_divs, (sky_color_1, sky_color_2), make_gradient=2),
        xform_m4=vecmat.get_transl_m4(0, sky_height / 2, -5.5))
    blue_sky_instance["children"]["north"]["ignore_lighting"] = True

    sky_graph["root"]["children"]["billboards"] = rasterizer.get_model_instance(None)
    sky_billboards_instance = sky_graph["root"]["children"]["billboards"]
    sky_billboards_instance["children"]["sun"] = rasterizer.get_model_instance(
        rasterizer.get_billboard(0, 1.5, -1, 1, 1, pygame.image.load("assets/sun.png").convert_alpha()))
    sky_billboards_instance["children"]["cloud_1"] = rasterizer.get_model_instance(
        rasterizer.get_billboard(-1, 1.2, -1, 0.7, 0.5, pygame.image.load("assets/smoke4.png").convert_alpha()))
    sky_billboards_instance["children"]["cloud_2"] = rasterizer.get_model_instance(
        rasterizer.get_billboard(1, 1.2, -1, 0.7, 0.5, pygame.image.load("assets/smoke2.png").convert_alpha()))

    # Ground graph
    ground_graph["root"]["children"]["ground"] = rasterizer.get_model_instance(
        meshes.get_rect_mesh((11 * 4, 11 * 4), (11, 11), ((180, 180, 180), (60, 60, 60))),
        vecmat.get_rot_x_m4(vecmat.deg_to_rad(-90)))

    # World graph
    # Interior: pedestal and spheres
    stone_color = (100, 100, 110)
    world_graph["root"]["children"]["pedestal"] = rasterizer.get_model_instance(
        meshes.get_cylinder_mesh(0.5, 0.5, 5, stone_color, True, False),
        xform_m4=vecmat.get_transl_m4(0, 0.25, 0))
    world_graph["root"]["children"]["pedestal"]["gouraud"] = True
    world_graph["root"]["children"]["pedestal"]["subdivide_max_iterations"] = 2

    planet_mip_textures = textures.get_mip_textures("assets/Terrestrial-Clouds-EQUIRECTANGULAR-0-64x32.png")
    planet_mip_textures.pop(0)
    world_graph["root"]["children"]["blue_sphere"] = rasterizer.get_model_instance(
        meshes.get_sphere_mesh(0.2, 20, 10))
    blue_sphere = world_graph["root"]["children"]["blue_sphere"]
    blue_sphere["model"]["texture"] = planet_mip_textures

    img = pygame.transform.scale(pygame.image.load("assets/blue_spot.png").convert_alpha(), (5, 5))
    l_divs = 15
    r_divs = 15
    r_phi_step = 2 * math.pi / r_divs
    l_phi_step = math.pi / l_divs
    radius = 0.7
    num_particles = (l_divs - 1) * r_divs
    particles = rasterizer.get_particles(img, num_particles, 1, 1)
    blue_sphere["children"]["particles"] = rasterizer.get_model_instance(particles)
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

    blue_sphere["children"]["red_sphere"] = rasterizer.get_model_instance(
        meshes.get_sphere_mesh(0.1, 10, 5, (80, 60, 20)))
    red_sphere = blue_sphere["children"]["red_sphere"]
    red_sphere["gouraud"] = True

    red_sphere["children"]["green_sphere"] = rasterizer.get_model_instance(
        meshes.get_sphere_mesh(0.05, 10, 5, (90, 80, 30)),
        xform_m4=vecmat.get_transl_m4(0.15, 0, 0))
    red_sphere["children"]["green_sphere"]["gouraud"] = True

    # Interior: columns
    column_positions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    for x,y in column_positions:
        world_graph["root"]["children"][f"column_{y}_{x}"] = rasterizer.get_model_instance(
            meshes.get_cylinder_mesh(5, 0.5, 10, stone_color, False, False),
            xform_m4=vecmat.get_transl_m4(y, 2.5, x))
        world_graph["root"]["children"][f"column_{y}_{x}"]["gouraud"] = True

    ground_graph["root"]["children"]["column_top"] = rasterizer.get_model_instance(meshes.get_cube_mesh(stone_color))
    column_top_inst = ground_graph["root"]["children"]["column_top"]
    column_top_inst["xform_m4"] = vecmat.get_transl_m4(0, 5, 0)
    column_top_inst["preproc_m4"] = vecmat.get_scal_m4(6, 1, 6)
    column_top_inst["gouraud"] = True
    column_top_inst["subdivide_max_iterations"] = 2


    # Interior: billboards
    img = pygame.image.load("assets/LampStand.png").convert_alpha()
    lamp_positions = [(-1.5, -4.7), (1.5, -4.7)]
    for x,y in lamp_positions:
        world_graph["root"]["children"][f"lamp_{x}_{y}"] = rasterizer.get_model_instance(
            rasterizer.get_billboard(x, 1, y, 3.5, 3.5, img))

    fire_ss = SpriteSheet("assets/fire1_64.png")
    fire_imgs = []
    for y in range(6):
        for x in range(10):
            fire_imgs.append(fire_ss.get_image(x * 64, y * 64, 64, 64))
    for x,y in lamp_positions:
        world_graph["root"]["children"][f"fire_{x}_{y}"] = rasterizer.get_model_instance(
            rasterizer.get_animated_billboard(x, 2.3, y-0.1, 6, 6, fire_imgs))

    # Interior: painting
    mip_textures = textures.get_mip_textures("assets/Mona_Lisa_64x64.png")
    painting_pos = (0, 1, -5.2)
    world_graph["root"]["children"]["painting"] = rasterizer.get_model_instance(meshes.get_test_texture_mesh(mip_textures),
        xform_m4=vecmat.get_transl_m4(*painting_pos))
    world_graph["root"]["children"]["painting"]["subdivide_max_iterations"] = 6

    world_graph["root"]["children"]["painting_wall"] = rasterizer.get_model_instance(meshes.get_cube_mesh(stone_color))
    painting_wall_inst = world_graph["root"]["children"]["painting_wall"]
    painting_wall_inst["xform_m4"] = vecmat.get_transl_m4(0, 1.5, painting_pos[2] - 0.6)
    painting_wall_inst["preproc_m4"] = vecmat.get_scal_m4(3, 3, 0.3)
    painting_wall_inst["gouraud"] = True
    painting_wall_inst["subdivide_max_iterations"] = 3

    # Interior: NPC
    npc_animation = model_file_io.load_animation({
        "idle": ("assets/dummy-idle.zip", (1, 60)),
        "walk": ("assets/dummy-walk.zip", (63, 123)),
        "run": ("assets/dummy-run.zip", (1, 35))})
    world_graph["root"]["children"]["npc"] = rasterizer.get_model_instance(npc_animation,
                                                                           xform_m4=vecmat.get_transl_m4(4, 0, 0))
    world_graph["root"]["children"]["npc"]["animation"] = "walk"

    world_graph["root"]["children"]["npc_2"] = rasterizer.get_model_instance(npc_animation,
                                                                             vecmat.get_rot_y_m4(vecmat.deg_to_rad(-45)),
                                                                             xform_m4=vecmat.get_transl_m4(-7, 0, -2))
    world_graph["root"]["children"]["npc_2"]["animation"] = "idle"

    world_graph["root"]["children"]["npc_3"] = rasterizer.get_model_instance(npc_animation,
                                                                             vecmat.get_rot_y_m4(vecmat.deg_to_rad(-45)),
                                                                             xform_m4=vecmat.get_transl_m4(-7, 0, -4))
    world_graph["root"]["children"]["npc_3"]["animation"] = "walk"

    world_graph["root"]["children"]["npc_4"] = rasterizer.get_model_instance(npc_animation,
                                                                             vecmat.get_rot_y_m4(vecmat.deg_to_rad(-45)),
                                                                             xform_m4=vecmat.get_transl_m4(-7, 0, -6))
    world_graph["root"]["children"]["npc_4"]["animation"] = "run"


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
        # pos = [round(p, 2) for p in CAMERA['pos']]
        textblock_fps = font.render(f"{round(clock.get_fps(), 1)} fps", True, TEXT_COLOR)
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

    def do_sky():
        """The sky moves along with the camera's x/z position & y rotation"""
        cam_pos = CAMERA["pos"]
        sky_m = vecmat.get_transl_m4(cam_pos[0], 0, cam_pos[2])
        sky_billboards_instance["xform_m4"] = sky_m
        cam_rot = CAMERA["rot"]
        sky_m = vecmat.mat4_mat4_mul(sky_m, vecmat.get_rot_y_m4(cam_rot[1]))
        blue_sky_instance["xform_m4"] = sky_m

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

    npc_phases = ["idle", "walk", "run"]
    def do_animation():
        nonlocal frame
        nonlocal blue_sphere
        rot_m = vecmat.get_rot_y_m4(vecmat.deg_to_rad(frame * 1.5))
        m = vecmat.get_transl_m4(0, 1.5, 0)
        m = vecmat.mat4_mat4_mul(m, rot_m)
        blue_sphere["xform_m4"] = m
        m = vecmat.get_transl_m4(0.5, 0, 0)
        m = vecmat.mat4_mat4_mul(m, rot_m)
        red_sphere["xform_m4"] = m
        if frame % 9 == 0:
            d = 0.08
            LIGHTING["pointlight"] = [random.uniform(-d, d) + 1,
                                      random.uniform(-d, d) + 1,
                                      random.uniform(-d, d) + -5, 1]
            LIGHTING["pointlight_falloff"] = random.uniform(1.5, 1.6)

        npc_dur = 300
        if frame % npc_dur == 0:
            world_graph["root"]["children"]["npc"]["animation"] = npc_phases[(frame // npc_dur) % 3]

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

        do_animation()
        do_movement()
        do_sky()

        screen.fill(sky_color_2)

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
