"""
Filled polygons with simple lighting rasterizer demo using pygame
"""

import math
import pygame
import random
import time

import pyrasterize.vecmat as vecmat
import pyrasterize.rasterizer as rasterizer
import pyrasterize.meshes as meshes

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 320, 240
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}
SPRITE_SPEED = 0.1
CUBE_COLOR_1 = (200, 0, 0)
CUBE_COLOR_2 = (0, 0, 200)
CUBE_FACES = [
    ("faceTop",    (0,5,0),  (0,0,0)),
    ("faceBottom", (0,-5,0), (vecmat.deg_to_rad(-180),0,0)),
    ("faceFront",  (0,0,5),  (vecmat.deg_to_rad(90),0,0)),
    ("faceBack",   (0,0,-5), (vecmat.deg_to_rad(-90),0,0)),
    ("faceLeft",   (-5,0,0), (0,0,vecmat.deg_to_rad(90))),
    ("faceRight",  (5,0,0),  (0,0,vecmat.deg_to_rad(-90))),
    ]

def create_scene_graph():
    """Return scene graph to draw"""

    def get_sprite_instance():
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
        #
        instance["pos"] = [0,0]
        instance["target"] = [1, 0]
        return instance

    scene_graph = { "cubeRoot": rasterizer.get_model_instance(None) }

    for face_name,face_tran,face_rot in CUBE_FACES:
        xform_m4 = vecmat.get_rot_x_m4(face_rot[0])
        xform_m4 = vecmat.mat4_mat4_mul(vecmat.get_rot_y_m4(face_rot[1]), xform_m4)
        xform_m4 = vecmat.mat4_mat4_mul(vecmat.get_rot_z_m4(face_rot[2]), xform_m4)
        xform_m4 = vecmat.mat4_mat4_mul(vecmat.get_transl_m4(*face_tran), xform_m4)
        scene_graph["cubeRoot"]["children"][face_name] = rasterizer.get_model_instance(
            meshes.get_rect_mesh((10, 10), (10, 10), (CUBE_COLOR_1, CUBE_COLOR_2)),
            vecmat.get_rot_x_m4(vecmat.deg_to_rad(-90)),
            xform_m4)
        face = scene_graph["cubeRoot"]["children"][face_name]
        face["children"]["sprite"] = get_sprite_instance()

    orbiter_model = meshes.get_sphere_mesh(3, 10, 5, (255, 255, 0))
    scene_graph["orbiterRoot"] = rasterizer.get_model_instance(None)
    for pos in [(11,0,0), (-11,0,0), (0,0,11), (0,0,-11)]:
        scene_graph["orbiterRoot"]["children"][f"orbiter_{pos}"] = rasterizer.get_model_instance(orbiter_model,
            vecmat.mat4_mat4_mul(vecmat.get_transl_m4(*pos),
                vecmat.mat4_mat4_mul(vecmat.get_scal_m4(0.5, 0.5, 0.5),
                    vecmat.get_transl_m4(*meshes.get_model_centering_offset(orbiter_model)))))

    return scene_graph

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE, pygame.SCALED)
    pygame.display.set_caption("PyRasterize")
    clock = pygame.time.Clock()

    scene_graph = create_scene_graph()
    sum_triangles = 0
    def acc_tris(_,instance):
        nonlocal sum_triangles
        if instance["model"]:
            sum_triangles += len(instance["model"]["tris"])
    rasterizer.visit_instances(scene_graph, acc_tris)
    print(f"total triangles in scene: {sum_triangles}")

    font = pygame.font.Font(None, 30)
    title1 = font.render("A SCENE GRAPH organizes 3D objects as a tree structure,", True, RGB_WHITE)
    title2 = font.render("propagating transformations from parent to child objects.", True, RGB_WHITE)
    title3 = font.render("Each cube face is a child of the cube's abstract parent,", True, RGB_WHITE)
    title4 = font.render("providing an independent coordinate system for the figures to move in.", True, RGB_WHITE)

    bytes_render_image = bytearray(SCR_WIDTH * SCR_HEIGHT * 4)
    render_surface = pygame.image.frombuffer(bytes_render_image, SCR_SIZE, 'RGBA')

    bytes_fog_image = bytearray(SCR_WIDTH * SCR_HEIGHT * 4)
    fog_surface = pygame.image.frombuffer(bytes_fog_image, SCR_SIZE, 'RGBA')

    frame = 0
    done = False
    while not done:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        camera_m = vecmat.get_rot_x_m4(vecmat.deg_to_rad(0))
        camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_y_m4(0), camera_m)
        camera_m = vecmat.mat4_mat4_mul(vecmat.get_rot_z_m4(0), camera_m)
        camera_m = vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, -3, -17), camera_m)

        for face_name,_,_ in CUBE_FACES:
            instance = scene_graph["cubeRoot"]["children"][face_name]["children"]["sprite"]
            pos = instance["pos"]
            target = instance["target"]
            phi = math.atan2(target[1] - pos[1], target[0] - pos[0])
            d_p = (SPRITE_SPEED * math.cos(phi), SPRITE_SPEED * math.sin(phi))
            pos[0] += d_p[0]
            pos[1] += d_p[1]
            if abs(target[0] - pos[0]) + abs(target[1] - pos[1]) < 0.1:
                pos[0] = target[0]
                pos[1] = target[1]
                target[0] = random.uniform(-4, 4)
                target[1] = random.uniform(-4, 4)
            mat = vecmat.get_rot_y_m4(-phi - math.pi/2)
            mat = vecmat.mat4_mat4_mul(vecmat.get_transl_m4(pos[0], 1.6, pos[1]), mat)
            instance["xform_m4"] = mat

            for name,side in [("leftLeg", 0), ("rightLeg", 1)]:
                leg = instance["children"][name]
                leg["xform_m4"] = vecmat.get_rot_x_m4(vecmat.deg_to_rad(20
                    * math.sin(vecmat.deg_to_rad((side*180) + (frame*10) % 360))))

        angle = 0.4 * vecmat.deg_to_rad(frame)
        cube_m = vecmat.get_rot_x_m4(angle)
        cube_m = vecmat.mat4_mat4_mul(vecmat.get_rot_y_m4(angle * 0.6), cube_m)
        cube_m = vecmat.mat4_mat4_mul(vecmat.get_rot_z_m4(angle * 0.4), cube_m)
        scene_graph["cubeRoot"]["xform_m4"] = cube_m

        scene_graph["orbiterRoot"]["xform_m4"] = vecmat.get_rot_y_m4(-angle)

        persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(90), SCR_WIDTH/SCR_HEIGHT)

        render_surface.fill((0, 0, 0, 255))
        fog_surface.fill((255, 0, 0, 0))

        t = time.perf_counter()
        rasterizer.render(render_surface, fog_surface, SCR_AREA, scene_graph, camera_m, persp_m, LIGHTING)
        elapsed_time = time.perf_counter() - t
        # if frame % 30 == 0:
        #     print(f"render time: {round(elapsed_time, 3)} s")

        fog_color = (255, 255, 255)
        for i in range(0, SCR_HEIGHT * SCR_WIDTH * 4, 4):
            f = bytes_fog_image[i] / 255.0
            bytes_render_image[i] = int(f * fog_color[0] + (1-f) * bytes_render_image[i])
            bytes_render_image[i+1] = int(f * fog_color[1] + (1-f) * bytes_render_image[i+1])
            bytes_render_image[i+2] = int(f * fog_color[2] + (1-f) * bytes_render_image[i+2])

        screen.blit(render_surface, (0, 0))

        # screen.blit(title1, (30, 20))
        # screen.blit(title2, (30, 50))
        # screen.blit(title3, (30, 80))
        # screen.blit(title4, (30, 110))

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{round(clock.get_fps(), 2)} fps")

if __name__ == '__main__':
    main_function()
