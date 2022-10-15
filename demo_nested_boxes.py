"""
Filled polygons with simple lighting rasterizer demo using pygame
"""

import math
import pygame
import pygame.gfxdraw
from demo_instances import deg_to_rad

import vecmat
import rasterizer
import meshes

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 800, 600
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

def get_camera_m(cam):
    """Return matrix based on camera dict(rot = [x,y,z],pos = [x,y,z])"""
    cam_pos = cam["pos"]
    cam_rot = cam["rot"]
    camera_m = vecmat.get_transl_m4(-cam_pos[0], -cam_pos[1], -cam_pos[2])
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_z_m4(-cam_rot[2]), camera_m)
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_y_m4(-cam_rot[1]), camera_m)
    camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_x_m4(-cam_rot[0]), camera_m)
    return camera_m

CAMERA = { "pos": [0,0,0], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}

def get_sprite_instance():
    """Make sprite instance"""
    body_width = 0.75
    instance = rasterizer.get_model_instance(meshes.get_cube_mesh())
    instance["preproc_m4"] = vecmat.get_scal_m4(body_width, 1, 0.5)
    children = instance["children"]
    #
    head_size = 0.4
    children["head"] = rasterizer.get_model_instance(meshes.get_cube_mesh((242,212,215)))
    children["head"]["xform_m4"] = vecmat.get_transl_m4(0, 1 - head_size, 0)
    children["head"]["preproc_m4"] = vecmat.get_scal_m4(head_size, head_size, head_size)
    #
    leg_w = 0.25
    stance_w = 1.2
    for name,side in [("leftLeg", -1), ("rightLeg", 1)]:
        children[name] = rasterizer.get_model_instance(meshes.get_cube_mesh())
        m_leg = vecmat.mat4_mat4_mul(vecmat.get_transl_m4(side * leg_w / 2 * stance_w, -1, 0),
            vecmat.get_scal_m4(leg_w, 1, leg_w))
        children[name]["preproc_m4"] = m_leg
    #
    arm_w = 0.2
    arm_len = 0.9
    children["leftArm"] = rasterizer.get_model_instance(meshes.get_cube_mesh())
    children["leftArm"]["xform_m4"] = vecmat.get_transl_m4(-body_width/2-arm_w/2, 0, 0)
    children["leftArm"]["preproc_m4"] = vecmat.get_scal_m4(arm_w, arm_len, arm_w)
    children["rightArm"] = rasterizer.get_model_instance(meshes.get_cube_mesh())
    children["rightArm"]["xform_m4"] = vecmat.get_transl_m4(body_width/2+arm_w/2, 0, 0)
    children["rightArm"]["preproc_m4"] = vecmat.get_scal_m4(arm_w, arm_len, arm_w)
    return instance

SPRITE_INSTANCE = get_sprite_instance()

def animate_sprite(frame):
    """Animate the main sprite instance"""
    for name,side in [("leftLeg", 0), ("rightLeg", 1)]:
        leg = SPRITE_INSTANCE["children"][name]
        leg["xform_m4"] = vecmat.get_rot_x_m4(vecmat.deg_to_rad(20
            * math.sin(vecmat.deg_to_rad((side*180) + (frame*10) % 360))))
    radius = 1.4
    BOX["children"]["sprite"]["xform_m4"] = vecmat.mat4_mat4_mul(
        vecmat.get_transl_m4(radius * math.cos(vecmat.deg_to_rad(frame)),
            0.3,
            radius * math.sin(vecmat.deg_to_rad(frame))),
        vecmat.mat4_mat4_mul(vecmat.get_rot_y_m4(vecmat.deg_to_rad(-frame)), vecmat.get_scal_m4(0.2, 0.2, 0.2))
        )

BOX = rasterizer.get_model_instance(None)

def create_scene_graph():
    """Create the main scene graph"""
    BOX["children"]["cube"] = rasterizer.get_model_instance(None)
    cube = BOX["children"]["cube"]

    cube["children"]["face_top"] = rasterizer.get_model_instance(meshes.get_rect_mesh((3, 3), (15, 15),
        ((100, 100, 200), (90, 90, 150))),
        vecmat.get_rot_x_m4(vecmat.deg_to_rad(-90)))
    cube["children"]["face_front"] = rasterizer.get_model_instance(meshes.get_rect_mesh((3, 0.25), (15, 2),
        ((160,50,50), (90, 90, 150))),
        None,
        vecmat.get_transl_m4(0, -0.125, 1.5))
    cube["children"]["face_back"] = rasterizer.get_model_instance(meshes.get_rect_mesh((3, 0.25), (15, 2),
        ((160,50,50), (90, 90, 150))),
        vecmat.get_rot_y_m4(vecmat.deg_to_rad(180)),
        vecmat.get_transl_m4(0, -0.125, -1.5))
    cube["children"]["face_left"] = rasterizer.get_model_instance(meshes.get_rect_mesh((3, 0.25), (15, 2),
        ((160,50,50), (90, 90, 150))),
        vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90)),
        vecmat.get_transl_m4(-1.5, -0.125, 0))

    BOX["children"]["sprite"] = rasterizer.get_model_instance(None, children={"sprite_inst": SPRITE_INSTANCE})

    scene_graph = { "root": rasterizer.get_model_instance(None) }
    scene_graph["root"]["children"]["box"] = BOX

    inst = scene_graph["root"]
    for _ in range(4):
        inst["children"]["box_ref"] = rasterizer.get_model_instance(None, None,
            vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, 0.5, 0), vecmat.get_scal_m4(0.5, 0.5, 0.5)),
            children={"box": BOX})
        inst = inst["children"]["box_ref"]

    return scene_graph

def draw_scene_graph(surface, frame, scene_graph):
    """Draw the scene graph"""
    i = 200
    CAMERA["pos"] = [0, 1 + 2 * abs(math.sin(vecmat.deg_to_rad(i))), 2]
    CAMERA["rot"] = [vecmat.deg_to_rad(-60 + 20 * abs(math.cos(vecmat.deg_to_rad(i)))), 0, 0]
    #
    animate_sprite(frame)
    #
    scene_graph["root"]["xform_m4"] = vecmat.get_rot_y_m4(vecmat.deg_to_rad(frame))
    #
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    rasterizer.render(surface, SCR_AREA, scene_graph, get_camera_m(CAMERA), persp_m, LIGHTING)

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE)
    pygame.display.set_caption("PyRasterize")
    clock = pygame.time.Clock()

    scene_graph = create_scene_graph()
    font = pygame.font.Font(None, 30)

    frame = 0
    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        draw_scene_graph(screen, frame, scene_graph)

        screen.blit(font.render(f"FOV: {float(int(CAMERA['fov'] * 10))/10}",
            True, RGB_WHITE), (30, 20))
        pygame.draw.circle(screen, (100,100,100), (745,55), 50)
        pygame.gfxdraw.pie(screen, 745, 55, 50, -int(CAMERA['fov']/2), int(CAMERA['fov']/2), RGB_WHITE)

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()
