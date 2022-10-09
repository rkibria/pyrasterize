"""
Filled polygons with simple lighting rasterizer demo using pygame
"""

import math
import random
import pygame

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 800, 600
SCR_ASPECT_RATIO = SCR_WIDTH / SCR_HEIGHT
SCR_ORIGIN_X = SCR_WIDTH / 2
SCR_ORIGIN_Y = SCR_HEIGHT / 2

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

# MATHS

def norm_vec3(v_3):
    """Return normalized vec3"""
    mag = v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2]
    if mag == 0:
        return [0, 0, 0]
    mag = 1.0 / math.sqrt(mag)
    return [v_3[0] * mag, v_3[1] * mag, v_3[2] * mag]

def mat4_mat4_mul(m4_1, m4_2):
    """Return multiplication of 4x4 matrices
    Unrolled form is faster than loops"""
    result = [0] * 16
    result[0] += m4_1[0] * m4_2[0] # row 1 x column 1
    result[0] += m4_1[1] * m4_2[4]
    result[0] += m4_1[2] * m4_2[8]
    result[0] += m4_1[3] * m4_2[12]
    result[1] += m4_1[0] * m4_2[1] # row 1 x column 2
    result[1] += m4_1[1] * m4_2[5]
    result[1] += m4_1[2] * m4_2[9]
    result[1] += m4_1[3] * m4_2[13]
    result[2] += m4_1[0] * m4_2[2]
    result[2] += m4_1[1] * m4_2[6]
    result[2] += m4_1[2] * m4_2[10]
    result[2] += m4_1[3] * m4_2[14]
    result[3] += m4_1[0] * m4_2[3]
    result[3] += m4_1[1] * m4_2[7]
    result[3] += m4_1[2] * m4_2[11]
    result[3] += m4_1[3] * m4_2[15]
    result[4] += m4_1[4] * m4_2[0]
    result[4] += m4_1[5] * m4_2[4]
    result[4] += m4_1[6] * m4_2[8]
    result[4] += m4_1[7] * m4_2[12]
    result[5] += m4_1[4] * m4_2[1]
    result[5] += m4_1[5] * m4_2[5]
    result[5] += m4_1[6] * m4_2[9]
    result[5] += m4_1[7] * m4_2[13]
    result[6] += m4_1[4] * m4_2[2]
    result[6] += m4_1[5] * m4_2[6]
    result[6] += m4_1[6] * m4_2[10]
    result[6] += m4_1[7] * m4_2[14]
    result[7] += m4_1[4] * m4_2[3]
    result[7] += m4_1[5] * m4_2[7]
    result[7] += m4_1[6] * m4_2[11]
    result[7] += m4_1[7] * m4_2[15]
    result[8] += m4_1[8] * m4_2[0]
    result[8] += m4_1[9] * m4_2[4]
    result[8] += m4_1[10] * m4_2[8]
    result[8] += m4_1[11] * m4_2[12]
    result[9] += m4_1[8] * m4_2[1]
    result[9] += m4_1[9] * m4_2[5]
    result[9] += m4_1[10] * m4_2[9]
    result[9] += m4_1[11] * m4_2[13]
    result[10] += m4_1[8] * m4_2[2]
    result[10] += m4_1[9] * m4_2[6]
    result[10] += m4_1[10] * m4_2[10]
    result[10] += m4_1[11] * m4_2[14]
    result[11] += m4_1[8] * m4_2[3]
    result[11] += m4_1[9] * m4_2[7]
    result[11] += m4_1[10] * m4_2[11]
    result[11] += m4_1[11] * m4_2[15]
    result[12] += m4_1[12] * m4_2[0]
    result[12] += m4_1[13] * m4_2[4]
    result[12] += m4_1[14] * m4_2[8]
    result[12] += m4_1[15] * m4_2[12]
    result[13] += m4_1[12] * m4_2[1]
    result[13] += m4_1[13] * m4_2[5]
    result[13] += m4_1[14] * m4_2[9]
    result[13] += m4_1[15] * m4_2[13]
    result[14] += m4_1[12] * m4_2[2]
    result[14] += m4_1[13] * m4_2[6]
    result[14] += m4_1[14] * m4_2[10]
    result[14] += m4_1[15] * m4_2[14]
    result[15] += m4_1[12] * m4_2[3]
    result[15] += m4_1[13] * m4_2[7]
    result[15] += m4_1[14] * m4_2[11]
    result[15] += m4_1[15] * m4_2[15]
    return result

def vec4_mat4_mul(v_4, m_4):
    """Return vec4 multiplied by 4x4 matrix
    This form was more than twice as fast as a nested loop"""
    v_4_0 = v_4[0]
    v_4_1 = v_4[1]
    v_4_2 = v_4[2]
    v_4_3 = v_4[3]
    return [m_4[ 0] * v_4_0 + m_4[ 1] * v_4_1 + m_4[ 2] * v_4_2 + m_4[ 3] * v_4_3,
            m_4[ 4] * v_4_0 + m_4[ 5] * v_4_1 + m_4[ 6] * v_4_2 + m_4[ 7] * v_4_3,
            m_4[ 8] * v_4_0 + m_4[ 9] * v_4_1 + m_4[10] * v_4_2 + m_4[11] * v_4_3,
            m_4[12] * v_4_0 + m_4[13] * v_4_1 + m_4[14] * v_4_2 + m_4[15] * v_4_3]

def get_unit_m4():
    """Return 4x4 unit matrix"""
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]

def get_transl_m4(d_x, d_y, d_z):
    """Return 4x4 translation matrix"""
    return [1.0, 0.0, 0.0, float(d_x),
            0.0, 1.0, 0.0, float(d_y),
            0.0, 0.0, 1.0, float(d_z),
            0.0, 0.0, 0.0, 1.0]

def get_scal_m4(s_x, s_y, s_z):
    """Return 4x4 scaling matrix"""
    return [float(s_x), 0.0,       0.0,       0.0,
            0.0,       float(s_y), 0.0,       0.0,
            0.0,       0.0,       float(s_z), 0.0,
            0.0,       0.0,       0.0,        1.0]

def get_rot_x_m4(phi):
    """Return 4x4 x-rotation matrix"""
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    return [1.0, 0.0,     0.0,      0.0,
            0.0, cos_phi, -sin_phi, 0.0,
            0.0, sin_phi, cos_phi,  0.0,
            0.0, 0.0,     0.0,      1.0]

def get_rot_y_m4(phi):
    """Return 4x4 y-rotation matrix"""
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    return [cos_phi,  0.0,  sin_phi, 0.0,
            0.0,      1.0,  0.0,     0.0,
            -sin_phi, 0.0,  cos_phi, 0.0,
            0.0,      0.0,  0.0,     1.0]

def get_rot_z_m4(phi):
    """Return 4x4 z-rotation matrix"""
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    return [cos_phi, -sin_phi, 0.0, 0.0,
            sin_phi, cos_phi,  0.0, 0.0,
            0.0,     0.0,      1.0, 0.0,
            0.0,     0.0,      0.0, 1.0]

# def get_perp_m4(d, ar):
#     return [cos_phi, -sin_phi, 0.0, 0.0,
#             sin_phi, cos_phi,  0.0, 0.0,
#             0.0,     0.0,      1.0, 0.0,
#             0.0,     0.0,      0.0, 1.0]

def deg_to_rad(degrees):
    """Return degrees converted to radians"""
    return degrees * (math.pi / 180)

# MODELS

DEFAULT_COLOR = (200, 200, 200)

def get_test_triangle_mesh():
    """triangle to 1,1,0"""
    return {
        "verts" : [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
        ],
        "tris" : [(0, 1, 2)],
        "colors" : [DEFAULT_COLOR]
    }

# SCENE GRAPH RENDERING

def get_camera_m(cam):
    """Return matrix based on camera dict(rot = [x,y,z],pos = [x,y,z])"""
    cam_pos = cam["pos"]
    cam_rot = cam["rot"]
    camera_m = get_transl_m4(-cam_pos[0], -cam_pos[1], -cam_pos[2])
    camera_m = mat4_mat4_mul(get_rot_z_m4(-cam_rot[2]), camera_m)
    camera_m = mat4_mat4_mul(get_rot_y_m4(-cam_rot[1]), camera_m)
    camera_m = mat4_mat4_mul(get_rot_x_m4(-cam_rot[0]), camera_m)
    return camera_m

def render_model(surface, model, camera_m):
    """Render the scene graph"""

    world_verts = list(map(lambda v: vec4_mat4_mul((v[0], v[1], v[2], 1), camera_m),
        model["verts"]))

    print("transformed points:")
    for i in range(len(model["verts"])):
        print(f"{model['verts'][i]} to {world_verts[i]}")

# DEMO CODE

CAMERA = { "pos": [0,0,0], "rot": [0,0,0] }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}
SPRITE_SPEED = 0.1

def create_model():
    """Create the main scene graph"""
    return get_test_triangle_mesh()

def draw_model(surface, frame, model):
    """Draw the scene graph"""
    # radius = 3
    # CAMERA["pos"][0] = radius * math.cos(deg_to_rad(frame))
    # CAMERA["pos"][2] = radius * math.sin(deg_to_rad(frame))
    CAMERA["pos"][0] = 0
    CAMERA["pos"][1] = 0
    CAMERA["pos"][2] = 1 + frame % 5 # + 0.2 * math.sin(deg_to_rad(frame))
    # print(CAMERA["pos"][2])
    CAMERA["rot"][0] = deg_to_rad(0)
    render_model(surface, model, get_camera_m(CAMERA))

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE)
    pygame.display.set_caption("PyRasterize")
    clock = pygame.time.Clock()

    model = create_model()

    frame = 0
    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        draw_model(screen, frame, model)

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()
