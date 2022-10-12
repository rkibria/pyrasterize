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

def get_cube_mesh(color=DEFAULT_COLOR):
    """Return a unit cube mesh model dictionary:
    'verts': vertices (float vec3s for point positions in local coordinates)
    'tris': triangles (int vec3s indexing the 3 vertices in 'verts' of triangle)
    'colors': triangle colors (float vec3s of triangle RGB color)
    """
    return {
        "verts" : [
            ( 0.5,  0.5, 0.5),  # front top right     0
            ( 0.5, -0.5, 0.5),  # front bottom right  1
            (-0.5, -0.5, 0.5),  # front bottom left   2
            (-0.5,  0.5, 0.5),  # front top left      3
            ( 0.5,  0.5, -0.5), # back top right      4
            ( 0.5, -0.5, -0.5), # back bottom right   5
            (-0.5, -0.5, -0.5), # back bottom left    6
            (-0.5,  0.5, -0.5)  # back top left       7
            ],
        "tris" : [ # CCW winding order
            (0, 3, 1), # front face
            (2, 1, 3), #
            (3, 7, 2), # left face
            (6, 2, 7), #
            (4, 0, 5), # right face
            (1, 5, 0), #
            (4, 7, 0), # top face
            (3, 0, 7), #
            (1, 2, 5), # bottom face
            (6, 5, 2), #
            (7, 4, 6), # back face
            (5, 6, 4)  #
            ],
        "colors": [[color[0], color[1], color[2]]] * 12
        }

def get_rect_mesh(r_size, r_divs, colors=(DEFAULT_COLOR, DEFAULT_COLOR)):
    """Return 2D rectangle mesh of given size and subdivision
    with checkerboard coloring"""
    mesh = { "verts": [], "tris": [], "colors": []}
    d_x,d_y = r_divs

    start_x = -r_size[0] / 2.0
    step_x = r_size[0] / d_x
    start_y = -r_size[1] / 2.0
    step_y = r_size[1] / d_y
    for i_y in range(d_y + 1):
        for i_x in range(d_x + 1):
            mesh["verts"].append((start_x + step_x * i_x, start_y + step_y * i_y, 0))

    for i_y in range(d_y):
        for i_x in range(d_x):
            u_l = i_x + i_y * (d_x + 1)
            mesh["tris"].append((u_l, u_l + 1, u_l + 1 + (d_x + 1)))
            mesh["tris"].append((u_l, u_l + 1 + (d_x + 1), u_l + (d_x + 1)))
            color = colors[0] if (i_x + i_y) % 2 == 0 else colors[1]
            mesh["colors"].append(color)
            mesh["colors"].append(color)
    return mesh

def get_model_instance(model, preproc_m4=None, xform_m4=None):
    """Return model instance
    These are the key values in a scene graph {name_1: instance_1, ...} dictionary"""
    if preproc_m4 is None:
        preproc_m4 = get_unit_m4()
    if xform_m4 is None:
        xform_m4 = get_unit_m4()
    return { "model": model,
        "preproc_m4": preproc_m4,
        "xform_m4": xform_m4,
        "children": {} }

# FILE IO

def get_model_from_obj_file(fname):
    """Return model loaded from a Wavefront .obj file"""
    with open(fname, encoding="utf-8") as file:
        content = file.readlines()
    content = [x.strip() for x in content]

    mesh = {"verts": [], "tris": [], "colors": []}
    cur_color = DEFAULT_COLOR
    for line in content:
        if line.startswith("v "):
            tokens = line.split()
            mesh["verts"].append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
        elif line.startswith("usemtl "):
            tokens = line.split()[1:]
            mtl = tokens[0]
            if len(mtl) == 6:
                cur_color = (int(mtl[0:2], 16), int(mtl[2:4], 16), int(mtl[4:6], 16))
        elif line.startswith("f "):
            indices = []
            tokens = line.split()[1:]
            for face_token in tokens:
                indices.append(int(face_token.split("/")[0]) - 1)
            if len(indices) == 3:
                mesh["tris"].append((indices[0], indices[1], indices[2]))
                mesh["colors"].append(cur_color)
            elif len(indices) >= 4:
                for i in range(len(indices) - 2):
                    mesh["tris"].append((indices[0], indices[i+1], indices[i+2]))
                    mesh["colors"].append(cur_color)
            else:
                print("? indices " + str(indices))
    print(f"--- loaded {fname}: {len(mesh['verts'])} vertices, {len(mesh['tris'])} triangles")
    return mesh

def get_model_centering_offset(model):
    """Get vec3 to center position (with translate matrix) of the model"""
    avg = [0, 0, 0]
    for v_3 in model["verts"]:
        for i in range(3):
            avg[i] += v_3[i]
    for i in range(3):
        avg[i] /= len(model["verts"])
        avg[i] *= -1
    return avg

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

def get_visible_tris(tri_list, world_vec4_list, clip_planes=(-0.5,-100)):
    """Returns ([indices of visible triangles],[normals of all triangles])"""
    idcs = []
    normals = []
    i = -1
    near_clip = clip_planes[0]
    far_clip = clip_planes[1]
    for tri in tri_list:
        i += 1
        v_0 = world_vec4_list[tri[0]]
        v_1 = world_vec4_list[tri[1]]
        v_2 = world_vec4_list[tri[2]]
        if ( (v_0[2] >= near_clip or v_1[2] >= near_clip or v_2[2] >= near_clip)
          or (v_0[2] <= far_clip  or v_1[2] <= far_clip  or v_2[2] <= far_clip)):
            normals.append((0,0,0))
            continue
        # normal = cross_product(v_1 - v_0, v_2 - v_0)
        sub10 = (v_1[0] - v_0[0], v_1[1] - v_0[1], v_1[2] - v_0[2])
        sub20 = (v_2[0] - v_0[0], v_2[1] - v_0[1], v_2[2] - v_0[2])
        normal = (sub10[1]*sub20[2] - sub10[2]*sub20[1],
            sub10[2]*sub20[0] - sub10[0]*sub20[2],
            sub10[0]*sub20[1] - sub10[1]*sub20[0])
        normals.append(normal)
        # Back-face culling: visible if dot_product(v_0, normal) < 0
        if (v_0[0] * normal[0] + v_0[1] * normal[1] + v_0[2] * normal[2]) < 0:
            idcs.append(i)
    return (idcs, normals)

def render_scene_graph(surface, scene_graph, camera_m, lighting):
    """Render the scene graph"""

    ambient = lighting["ambient"]
    diffuse = lighting["diffuse"]

    def get_proj_light_dir():
        light_dir_vec3 = lighting["lightDir"]
        light_dir_vec4 = (light_dir_vec3[0], light_dir_vec3[1], light_dir_vec3[2], 0)
        return norm_vec3(vec4_mat4_mul(light_dir_vec4, camera_m)[0:3])

    proj_light_dir = get_proj_light_dir()

    scene_triangles = []

    def get_instance_tris(instance, model_m):
        """Get lighted triangles from this instance"""
        model = instance["model"]
        if not model:
            return

        world_verts = list(map(lambda v: vec4_mat4_mul((v[0], v[1], v[2], 1), model_m),
            model["verts"]))

        model_tris = model["tris"]
        idcs,normals = get_visible_tris(model_tris, world_verts)

        use_dyn_light = not "precompColors" in instance
        model_colors = model["colors"]

        view_plane = 1
        for idx in idcs:
            tri = model_tris[idx]
            points = []
            for i in range(3):
                v_3 = world_verts[tri[i]]
                v_2 = (view_plane * v_3[0] / -v_3[2], view_plane * v_3[1] / -v_3[2]) # perspective divide
                scr_x = SCR_ORIGIN_X + v_2[0] * SCR_ORIGIN_X
                scr_y = SCR_ORIGIN_Y - v_2[1] * SCR_ORIGIN_Y * SCR_ASPECT_RATIO
                points.append((int(scr_x), int(scr_y)))
            if use_dyn_light:
                normal = norm_vec3(normals[idx])
                color = model_colors[idx]
                dot_prd = max(0, proj_light_dir[0] * normal[0]
                    + proj_light_dir[1] * normal[1]
                    + proj_light_dir[2] * normal[2])
                intensity = min(1, max(0, ambient + diffuse * dot_prd))
                lighted_color = (intensity * color[0], intensity * color[1], intensity * color[2])
            else:
                lighted_color = instance["bakedColors"][idx]
            scene_triangles.append((
                (world_verts[tri[0]][2] + world_verts[tri[1]][2] + world_verts[tri[2]][2]) / 3,
                points,
                lighted_color))

    def traverse_scene_graph(subgraph, parent_m):
        for _,instance in subgraph.items():
            proj_m = mat4_mat4_mul(instance["xform_m4"], instance["preproc_m4"])
            proj_m = mat4_mat4_mul(parent_m, proj_m)
            proj_m = mat4_mat4_mul(camera_m, proj_m)

            get_instance_tris(instance, proj_m)

            pass_m = mat4_mat4_mul(parent_m, instance["xform_m4"])
            if instance["children"]:
                traverse_scene_graph(instance["children"], pass_m)

    traverse_scene_graph(scene_graph, get_unit_m4())
    scene_triangles.sort(key=lambda x: x[0], reverse=False)
    for _,points,color in scene_triangles:
        pygame.draw.polygon(surface, color, points)

# DEMO CODE

CAMERA = { "pos": [0,0,0], "rot": [0,0,0] }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}
SPRITE_SPEED = 0.1

def create_scene_graph():
    """Create the main scene graph"""
    scene_graph = { "root": get_model_instance(None) }
    scene_graph["root"]["children"]["tri"] = get_model_instance(get_test_triangle_mesh())
    # size = 11
    # start = -int(size / 2)
    # r_ch = scene_graph["root"]["children"]
    # tile_spacing = 1
    # for r_i in range(size):
    #     y_i = start + r_i
    #     for t_i in range(size):
    #         x_i = start + t_i
    #         color = (200, 0, 0) if ((r_i + t_i) % 2 == 0) else (0, 0, 200)
    #         tile = get_model_instance(get_rect_mesh((1,1), (1,1), (color, color)),
    #             get_rot_x_m4(deg_to_rad(-90)),
    #             get_transl_m4(x_i * tile_spacing, -1, y_i * tile_spacing))
    #         r_ch["tile_" + str(x_i) + "_" + str(y_i)] = tile
    return scene_graph

def draw_scene_graph(surface, frame, scene_graph):
    """Draw the scene graph"""
    # radius = 3
    # CAMERA["pos"][0] = radius * math.cos(deg_to_rad(frame))
    # CAMERA["pos"][2] = radius * math.sin(deg_to_rad(frame))
    CAMERA["pos"][0] = 0
    CAMERA["pos"][1] = 0
    CAMERA["pos"][2] = 1 + 0.2 * math.sin(deg_to_rad(frame))
    print(CAMERA["pos"][2])
    CAMERA["rot"][0] = deg_to_rad(0)
    render_scene_graph(surface, scene_graph, get_camera_m(CAMERA), LIGHTING)

# MAIN

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE)
    pygame.display.set_caption("PyRasterize")
    clock = pygame.time.Clock()

    scene_graph = create_scene_graph()

    frame = 0
    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        draw_scene_graph(screen, frame, scene_graph)

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()