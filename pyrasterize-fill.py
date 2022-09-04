"""
Filled polygons with simple lighting rasterizer demo using pygame
"""

import math
import pygame

# MATHS

def norm_vec3(v_3):
    """Return normalized vec3"""
    mag = v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2]
    if mag == 0:
        return [0, 0, 0]
    mag = 1.0 / math.sqrt(mag)
    return [v_3[0] * mag, v_3[1] * mag, v_3[2] * mag]

def mat4_mat4_mul(m4_1, m4_2):
    """Return multiplication of 4x4 matrices"""
    result = [0] * 16
    for row in range(4):
        for col in range(4):
            for i in range(4):
                result[4 * row + col] += m4_1[4 * row + i] * m4_2[4 * i + col]
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

# RENDERING ALGORITHMS

NEAR_CLIP_PLANE = -0.5
FAR_CLIP_PLANE = -100

def getVisibleTris(tris, worldVerts):
    """
    Return:
    - tri indices that aren't culled/outside view
    - normals indexed same as tri's
    - assumes the camera is at origin
    """
    idcs = []
    normals = []
    i = -1
    for tri in tris:
        i += 1
        v0 = worldVerts[tri[0]]
        v1 = worldVerts[tri[1]]
        v2 = worldVerts[tri[2]]
        if (v0[2] >= NEAR_CLIP_PLANE or v1[2] >= NEAR_CLIP_PLANE or v2[2] >= NEAR_CLIP_PLANE
          or v0[2] <= FAR_CLIP_PLANE or v1[2] <= FAR_CLIP_PLANE or v1[2] <= FAR_CLIP_PLANE):
            normals.append((0,0,0))
            continue
        # normal = crossProduct(subVec(v1, v0), subVec(v2, v0))
        sub10 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        sub20 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        normal = (sub10[1]*sub20[2] - sub10[2]*sub20[1],
            sub10[2]*sub20[0] - sub10[0]*sub20[2],
            sub10[0]*sub20[1] - sub10[1]*sub20[0])
        normals.append(normal)
        # isVisible = (dotProduct(v0, normal) < 0)
        if ((v0[0]*normal[0] + v0[1]*normal[1] + v0[2]*normal[2]) < 0):
            idcs.append(i)
    return (idcs, normals)

def getCameraTransform(rot, tran):
    m = get_rot_x_m4(rot[0])
    m = mat4_mat4_mul(get_rot_y_m4(rot[1]), m)
    m = mat4_mat4_mul(get_rot_z_m4(rot[2]), m)
    m = mat4_mat4_mul(get_transl_m4(*tran), m)
    return m

# DRAWING

def drawEdge(surface, p0, p1, color):
    x1 = o_x + p0[0] * o_x
    y1 = o_y - p0[1] * o_y * (width/height)
    x2 = o_x + p1[0] * o_x
    y2 = o_y - p1[1] * o_y * (width/height)
    pygame.draw.aaline(surface, color, (x1, y1), (x2, y2), 1)

def drawCoordGrid(surface, m, color):
    darkColor = (color[0]/2, color[1]/2, color[2]/2)
    def gridLine(v0, v1, color):
        v0 = vec4_mat4_mul(v0, m)
        v1 = vec4_mat4_mul(v1, m)
        if v0[2] >= NEAR_CLIP_PLANE or v1[2] >= NEAR_CLIP_PLANE:
            return
        p0 = (v0[0]/-v0[2], v0[1]/-v0[2]) # perspective divide
        p1 = (v1[0]/-v1[2], v1[1]/-v1[2])
        drawEdge(surface, p0, p1, color)
    numLines = 11
    for i in range(numLines):
        d = 1
        s = (numLines - 1) / 2
        t = -s + i * d
        gridLine((t, 0, s, 1), (t, 0, -s, 1), darkColor)
        gridLine((s, 0, t, 1), (-s, 0, t, 1), darkColor)
    origin = (0, 0, 0, 1)
    gridLine(origin, (5, 0, 0, 1), color)
    gridLine(origin, (0, 5, 0, 1), color)
    gridLine(origin, (0, 0, 5, 1), color)
    gridLine((5, 0, 1, 1), (5, 0, -1, 1), color)

def getInstanceTris(sceneTriangles, modelInstance, cameraM, modelM, lighting):
    """return times {project, cull, draw}"""
    ambient = lighting["ambient"]
    diffuse = lighting["diffuse"]
    model = modelInstance["model"]

    modelVerts = model["verts"]
    modelColors = model["colors"]
    worldVerts = list(map(lambda v: vec4_mat4_mul((v[0], v[1], v[2], 1), modelM), modelVerts))

    modelTris = model["tris"]
    drawIdcs,normals = getVisibleTris(modelTris, worldVerts)

    useDynamicLighting = not ("precompColors" in modelInstance)
    lightDir = lighting["lightDir"]
    lightDirVec4 = (lightDir[0], lightDir[1], lightDir[2], 0) # direction vector! w=0
    projLight = norm_vec3(vec4_mat4_mul(lightDirVec4, cameraM)[0:3])

    for idx in drawIdcs:
        tri = modelTris[idx]
        z0 = worldVerts[tri[0]][2]
        z1 = worldVerts[tri[1]][2]
        z2 = worldVerts[tri[2]][2]
        triZ = (z0 + z1 + z2) / 3
        points = []
        aspectRatio = width/height
        for i in range(3):
            v0 = worldVerts[tri[i]]
            p0 = (v0[0]/-v0[2], v0[1]/-v0[2]) # perspective divide
            x1 = o_x + p0[0] * o_x
            y1 = o_y - p0[1] * o_y * aspectRatio
            points.append((int(x1), int(y1)))
        if useDynamicLighting:
            normal = norm_vec3(normals[idx])
            color = modelColors[idx]
            lightNormalDotProduct = max(0, projLight[0]*normal[0]+projLight[1]*normal[1]+projLight[2]*normal[2])
            intensity = min(1, max(0, ambient + diffuse * lightNormalDotProduct))
            lightedColor = (intensity * color[0], intensity * color[1], intensity * color[2])
        else:
            lightedColor = modelInstance["bakedColors"][idx]
        sceneTriangles.append((triZ, points, lightedColor))

def drawSceneGraph(surface, sg, cameraM, lighting):
    sceneTriangles = []
    def traverseSg(subgraph, parentM):
        for _,instance in subgraph.items():
            projM = mat4_mat4_mul(instance["xform_m4"], instance["preproc_m4"])
            projM = mat4_mat4_mul(parentM, projM)
            projM = mat4_mat4_mul(cameraM, projM)
            getInstanceTris(sceneTriangles, instance, cameraM, projM, lighting)

            passM = mat4_mat4_mul(parentM, instance["xform_m4"])
            if instance["children"]:
                traverseSg(instance["children"], passM)
    traverseSg(sg, get_unit_m4())

    sceneTriangles.sort(key=lambda x: x[0], reverse=False)

    for _,points,color in sceneTriangles:
        pygame.draw.polygon(surface, color, points)

def getModelCenterPos(model):
    avg = [0, 0, 0]
    for v in model["verts"]:
        for i in range(3):
            avg[i] += v[i]
    for i in range(3):
        avg[i] /= len(model["verts"])
    return avg

# MAIN

RGB_BLACK = (0, 0, 0)
RGB_DARKGREEN = (0, 128, 0)

if __name__ == '__main__':
    pygame.init()

    size = width, height = 800, 600
    o_x = width/2
    o_y = height/2

    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("PyRasterize")

    done = False
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 30)

    lighting = {"lightDir" : (1, 1, 1), "ambient": 0.1, "diffuse": 0.9}

    # mesh = loadObjFile("teapot.obj") # teapot-low.obj
    # mesh = loadObjFile("Goldfish_01.obj") # https://poly.pizza/m/52s3JpUSjmX
    # meshSg = { "mesh_1" : MakeModelInstance(mesh) }
    # meshCenterVec = mulVec(-1, getModelCenterPos(mesh))
    # meshSg["mesh_1"]["preproc_m4"] = getTranslationMatrix(*meshCenterVec)
    # def drawMesh(surface, frame):
    #     angle = degToRad(frame)
    #     cameraM = getCameraTransform((degToRad(20), 0, 0), (0, -0.5, -17.5))
    #     drawCoordGrid(surface, cameraM, RGB_DARKGREEN)
    #     m = getRotateXMatrix(angle)
    #     m = matMatMult(getRotateYMatrix(angle), m)
    #     m = matMatMult(getRotateZMatrix(angle), m)
    #     meshSg["mesh_1"]["xform_m4"] = m
    #     return drawSceneGraph(surface, meshSg, cameraM, lighting)

    def MakeSpriteInstance():
        bodyWidth = 0.75
        spriteInstance = get_model_instance(get_cube_mesh())
        spriteInstance["preproc_m4"] = get_scal_m4(bodyWidth, 1, 0.5)
        bodyChildren = spriteInstance["children"]
        #
        headSize = 0.4
        bodyChildren["head"] = get_model_instance(get_cube_mesh((242,212,215)))
        bodyChildren["head"]["xform_m4"] = get_transl_m4(0, 1 - headSize, 0)
        bodyChildren["head"]["preproc_m4"] = get_scal_m4(headSize, headSize, headSize)
        #
        legWidth = 0.25
        stanceWidth = 1.2
        bodyChildren["leftLeg"] = get_model_instance(get_cube_mesh())
        bodyChildren["leftLeg"]["xform_m4"] = get_transl_m4(legWidth/2*stanceWidth, -1, 0)
        bodyChildren["leftLeg"]["preproc_m4"] = get_scal_m4(legWidth, 1, legWidth)
        bodyChildren["rightLeg"] = get_model_instance(get_cube_mesh())
        bodyChildren["rightLeg"]["xform_m4"] = get_transl_m4(-legWidth/2*stanceWidth, -1, 0)
        bodyChildren["rightLeg"]["preproc_m4"] = get_scal_m4(legWidth, 1, legWidth)
        #
        armWidth = 0.2
        armLength = 0.9
        bodyChildren["leftArm"] = get_model_instance(get_cube_mesh())
        bodyChildren["leftArm"]["xform_m4"] = get_transl_m4(-bodyWidth/2-armWidth/2, 0, 0)
        bodyChildren["leftArm"]["preproc_m4"] = get_scal_m4(armWidth, armLength, armWidth)
        bodyChildren["rightArm"] = get_model_instance(get_cube_mesh())
        bodyChildren["rightArm"]["xform_m4"] = get_transl_m4(bodyWidth/2+armWidth/2, 0, 0)
        bodyChildren["rightArm"]["preproc_m4"] = get_scal_m4(armWidth, armLength, armWidth)
        #
        return spriteInstance

    sceneGraph = { "ground": get_model_instance(get_rect_mesh((10, 10), (10, 10), ((200,0,0), (0,200,0))),
        get_rot_x_m4(deg_to_rad(-90))) }
    sceneGraph["ground"]["children"]["sprite_1"] = MakeSpriteInstance()

    def drawSprite(surface, frame):
        angle = deg_to_rad(frame)
        cameraM = getCameraTransform((deg_to_rad(20), 0, 0), (0, 0, -10))
        drawCoordGrid(surface, cameraM, RGB_DARKGREEN)
        y = math.sin(angle)*5
        sceneGraph["ground"]["children"]["sprite_1"]["xform_m4"] = get_transl_m4(y, 1.6, y)
        return drawSceneGraph(surface, sceneGraph, cameraM, lighting)

    frame = 0
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        drawSprite(screen, frame)

        pygame.display.flip()
        frame += 1
