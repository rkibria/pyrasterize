import pygame, math, sys, time

# MATHS

def subVec(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def mulVec(a, v):
    return [a * v[0], a * v[1], a * v[2]]

def normVec(v):
    mag = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    if mag == 0:
        return (0, 0, 0)
    mag = 1.0 / math.sqrt(mag)
    return [v[0] * mag, v[1] * mag, v[2] * mag]

def dotProduct(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def crossProduct(a, b):
    return [a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]

def matMatMult(m1, m2):
    newM = [0] * 16
    for r in range(4):
        for c in range(4):
            for i in range(4):
                v1 = m1[4 * r + i]
                v2 = m2[4 * i + c]
                newM[4 * r + c] += v1 * v2
    return newM

def vecMatMult(v, m):
    """This form was more than twice as fast as a nested loop"""
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    v3 = v[3]
    return [m[ 0] * v0 + m[ 1] * v1 + m[ 2] * v2 + m[ 3] * v3,
            m[ 4] * v0 + m[ 5] * v1 + m[ 6] * v2 + m[ 7] * v3,
            m[ 8] * v0 + m[ 9] * v1 + m[10] * v2 + m[11] * v3,
            m[12] * v0 + m[13] * v1 + m[14] * v2 + m[15] * v3]

def GetUnitMatrix():
        return [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,]

def getTranslationMatrix(dx, dy, dz):
        return [1.0, 0.0, 0.0, float(dx),
                0.0, 1.0, 0.0, float(dy),
                0.0, 0.0, 1.0, float(dz),
                0.0, 0.0, 0.0, 1.0,]

def GetScalingMatrix(sx, sy, sz):
        return [float(sx), 0.0,       0.0,       0.0,
                0.0,       float(sy), 0.0,       0.0,
                0.0,       0.0,       float(sz), 0.0,
                0.0,       0.0,       0.0,       1.0,]

def getRotateXMatrix(phi):
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        return [1.0,    0.0,            0.0,            0.0,
                0.0,    cos_phi,        -sin_phi,       0.0,
                0.0,    sin_phi,        cos_phi,        0.0,
                0.0,    0.0,            0.0,            1.0,]

def getRotateYMatrix(phi):
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        return [cos_phi,        0.0,    sin_phi,        0.0,
                0.0,            1.0,    0.0,            0.0,
                -sin_phi,       0.0,    cos_phi,        0.0,
                0.0,            0.0,    0.0,            1.0,]

def getRotateZMatrix(phi):
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        return [cos_phi,        -sin_phi,       0.0,     0.0,
                sin_phi,        cos_phi,        0.0,     0.0,
                0.0,            0.0,            1.0,     0.0,
                0.0,            0.0,            0.0,     1.0,]

def degToRad(d):
    return d * (math.pi / 180)

# MODELS

DEFAULT_COLOR = (200,200,200)

def GetCubeMesh(color=DEFAULT_COLOR):
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

def MakeModelInstance(model, preprocessM=GetUnitMatrix(), transformM=GetUnitMatrix()):
    return { "model": model,
        "preprocessM": preprocessM,
        "transformM": transformM,
        "children": {} }

# FILE IO

def loadObjFile(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    vertices = []
    triangles = []
    colors = []
    curColor = DEFAULT_COLOR
    for line in content:
        if line.startswith("v "):
            tokens = line.split()
            vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
        elif line.startswith("usemtl "):
            tokens = line.split()[1:]
            mtl = tokens[0]
            if len(mtl) == 6:
                r = int(mtl[0:2], 16)
                g = int(mtl[2:4], 16)
                b = int(mtl[4:6], 16)
                curColor = (r, g, b)
        elif line.startswith("f "):
            indices = []
            tokens = line.split()[1:]
            for faceToken in tokens:
                indices.append(int(faceToken.split("/")[0]) - 1)
            if len(indices) == 3:
                triangles.append((indices[0], indices[1], indices[2]))
                colors.append(curColor)
            elif len(indices) >= 4:
                for i in range(len(indices) - 2):
                    triangles.append((indices[0], indices[i+1], indices[i+2]))
                    colors.append(curColor)
            else:
                print("? indices " + str(indices))
    print("--- loaded %s: %d vertices, %d triangles" % (fname, len(vertices), len(triangles)))
    return {"verts": vertices, "tris": triangles, "colors": colors}

# RENDERING ALGORITHMS

def projectVerts(m, modelVerts):
    """Transform the model's vec3's into projected vec4's"""
    return list(map(lambda v: vecMatMult((v[0], v[1], v[2], 1), m), modelVerts))

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

def sortTrisByZ(idcs, tris, worldVerts):
    """Painter's Algorithm"""
    def _sortByZ(i):
        tri = tris[i]
        z0 = worldVerts[tri[0]][2]
        z1 = worldVerts[tri[1]][2]
        z2 = worldVerts[tri[2]][2]
        return (z0 + z1 + z2) / 3
    idcs.sort(key=_sortByZ, reverse=False)

def getCameraTransform(rot, tran):
    m = getRotateXMatrix(rot[0])
    m = matMatMult(getRotateYMatrix(rot[1]), m)
    m = matMatMult(getRotateZMatrix(rot[2]), m)
    m = matMatMult(getTranslationMatrix(*tran), m)
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
        v0 = vecMatMult(v0, m)
        v1 = vecMatMult(v1, m)
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

def drawModelFilled(surface, modelInstance, cameraM, modelM, lighting):
    """return times {project, cull, draw}"""
    lightDir = lighting["lightDir"]
    ambient = lighting["ambient"]
    diffuse = lighting["diffuse"]
    model = modelInstance["model"]

    times = []
    modelVerts = model["verts"]
    modelColors = model["colors"]
    st = time.time()
    worldVerts = projectVerts(modelM, modelVerts)
    times.append(time.time() - st) # projection time

    modelTris = model["tris"]
    st = time.time()
    drawIdcs,normals = getVisibleTris(modelTris, worldVerts)
    times.append(time.time() - st) # culling time

    st = time.time()
    sortTrisByZ(drawIdcs, modelTris, worldVerts)
    times.append(time.time() - st) # sorting time

    usePrecompColors = "precompColors" in modelInstance
    lightDirVec4 = (lightDir[0], lightDir[1], lightDir[2], 0) # direction vector! w=0
    projLight = normVec(vecMatMult(lightDirVec4, cameraM)[0:3])

    st = time.time()
    for idx in drawIdcs:
        tri = modelTris[idx]
        points = []
        aspectRatio = width/height
        for i in range(3):
            v0 = worldVerts[tri[i]]
            p0 = (v0[0]/-v0[2], v0[1]/-v0[2]) # perspective divide
            x1 = o_x + p0[0] * o_x
            y1 = o_y - p0[1] * o_y * aspectRatio
            points.append((int(x1), int(y1)))
        if not usePrecompColors:
            # Dynamic lighting
            normal = normVec(normals[idx])
            color = modelColors[idx]
            lightNormalDotProduct = max(0, projLight[0]*normal[0]+projLight[1]*normal[1]+projLight[2]*normal[2])
            intensity = min(1, max(0, ambient + diffuse * lightNormalDotProduct))
            lightedColor = (intensity * color[0], intensity * color[1], intensity * color[2])
        else:
            lightedColor = modelInstance["bakedColors"][idx]

        pygame.draw.polygon(surface, lightedColor, points)
    times.append(time.time() - st) # drawing time
    return times

# - Every instance has its own matrix that is applied before the position matrix
#   (e.g. for changing the size/shape/scaling/rotation)
# - After changing shape the position matrix is applied to bring the instance to
#   its position in the world
# - Child instances are positioned relative to their parent, e.g. if the parent
#   is at (1,2,3) and the child's position is (1,0,-1), the absolute position
#   of the child should be (2,2,2).
#  -> must pass parent position through to children
#  -> the camera matrix is used only for the z sorting

def drawSceneGraph(surface, sg, cameraM, lighting):
    """return times {project, cull, draw}"""
    # Get the z's of the center positions (0,0,0) of all instances
    # so we can sort them for painter's algorithm
    projPositions = [] # instance/center position pairs
    def findPositions(subgraph, parentM):
        """adds instance/center position to projPositions"""
        for _,instance in subgraph.items():
            projM = matMatMult(instance["transformM"], instance["preprocessM"])
            projM = matMatMult(parentM, projM)
            projM = matMatMult(cameraM, projM)

            instance["_projM"] = projM
            curPos = vecMatMult((0, 0, 0, 1), projM)
            projPositions.append((instance, curPos))

            passM = matMatMult(parentM, instance["transformM"])

            if instance["children"]:
                findPositions(instance["children"], passM)
    findPositions(sg, cameraM)

    def _sortByZ(p):
        return p[1][2]
    projPositions.sort(key=_sortByZ, reverse=False)

    times = []
    for instance,_ in projPositions:
        curTimes = drawModelFilled(surface, instance, cameraM, instance["_projM"], lighting)
        if len(times) == 0:
            times = curTimes
        else:
            for i in range(len(times)):
                times[i] += curTimes[i]
    return times

def precomputeColors(instance, lighting):
    model = instance["model"]
    lightDir = lighting["lightDir"]
    ambient = lighting["ambient"]
    diffuse = lighting["diffuse"]
    modelColor = instance["color"]
    modelM = instance["preprocessM"]

    instance["precompColors"] = True

    lightDirVec4 = (lightDir[0], lightDir[1], lightDir[2], 0) # direction vector! w=0
    projLight = normVec(vecMatMult(lightDirVec4, modelM)[0:3])

    worldVerts = projectVerts(modelM, model["verts"])
    modelTris = model["tris"]
    bakedColors = []
    for i in range(len(modelTris)):
        tri = modelTris[i]
        v0 = worldVerts[tri[0]]
        v1 = worldVerts[tri[1]]
        v2 = worldVerts[tri[2]]
        sub10 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        sub20 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        normal = (sub10[1]*sub20[2] - sub10[2]*sub20[1],
            sub10[2]*sub20[0] - sub10[0]*sub20[2],
            sub10[0]*sub20[1] - sub10[1]*sub20[0])
        normal = normVec(normal)
        lightNormalDotProduct = max(0, projLight[0]*normal[0]+projLight[1]*normal[1]+projLight[2]*normal[2])
        intensity = min(1, max(0, ambient + diffuse * lightNormalDotProduct))
        lightedColor = (int(intensity * modelColor[0]), int(intensity * modelColor[1]), int(intensity * modelColor[2]))
        bakedColors.append(lightedColor)
    instance["bakedColors"] = bakedColors

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
    # meshSg["mesh_1"]["preprocessM"] = getTranslationMatrix(*meshCenterVec)
    # def drawMesh(surface, frame):
    #     angle = degToRad(frame)
    #     cameraM = getCameraTransform((degToRad(20), 0, 0), (0, -0.5, -17.5))
    #     drawCoordGrid(surface, cameraM, RGB_DARKGREEN)
    #     m = getRotateXMatrix(angle)
    #     m = matMatMult(getRotateYMatrix(angle), m)
    #     m = matMatMult(getRotateZMatrix(angle), m)
    #     meshSg["mesh_1"]["transformM"] = m
    #     return drawSceneGraph(surface, meshSg, cameraM, lighting)

    def MakeSpriteInstance():
        bodyWidth = 0.75
        spriteInstance = MakeModelInstance(GetCubeMesh())
        spriteInstance["preprocessM"] = GetScalingMatrix(bodyWidth, 1, 0.5)
        bodyChildren = spriteInstance["children"]
        #
        headSize = 0.4
        bodyChildren["head"] = MakeModelInstance(GetCubeMesh((242,212,215)))
        bodyChildren["head"]["transformM"] = getTranslationMatrix(0, 1 - headSize, 0)
        bodyChildren["head"]["preprocessM"] = GetScalingMatrix(headSize, headSize, headSize)
        #
        legWidth = 0.25
        stanceWidth = 1.2
        bodyChildren["leftLeg"] = MakeModelInstance(GetCubeMesh())
        bodyChildren["leftLeg"]["transformM"] = getTranslationMatrix(legWidth/2*stanceWidth, -1, 0)
        bodyChildren["leftLeg"]["preprocessM"] = GetScalingMatrix(legWidth, 1, legWidth)
        bodyChildren["rightLeg"] = MakeModelInstance(GetCubeMesh())
        bodyChildren["rightLeg"]["transformM"] = getTranslationMatrix(-legWidth/2*stanceWidth, -1, 0)
        bodyChildren["rightLeg"]["preprocessM"] = GetScalingMatrix(legWidth, 1, legWidth)
        #
        armWidth = 0.2
        armLength = 0.9
        bodyChildren["leftArm"] = MakeModelInstance(GetCubeMesh())
        bodyChildren["leftArm"]["transformM"] = getTranslationMatrix(-bodyWidth/2-armWidth/2, 0, 0)
        bodyChildren["leftArm"]["preprocessM"] = GetScalingMatrix(armWidth, armLength, armWidth)
        bodyChildren["rightArm"] = MakeModelInstance(GetCubeMesh())
        bodyChildren["rightArm"]["transformM"] = getTranslationMatrix(bodyWidth/2+armWidth/2, 0, 0)
        bodyChildren["rightArm"]["preprocessM"] = GetScalingMatrix(armWidth, armLength, armWidth)
        #
        return spriteInstance

    spriteSg = { "sprite_1": MakeSpriteInstance() }
    def drawSprite(surface, frame):
        angle = degToRad(frame)
        cameraM = getCameraTransform((degToRad(20), 0, 0), (0, 0, -3))
        drawCoordGrid(surface, cameraM, RGB_DARKGREEN)
        m = getRotateXMatrix(angle)
        m = matMatMult(getRotateYMatrix(angle), m)
        m = matMatMult(getRotateZMatrix(angle), m)
        spriteSg["sprite_1"]["transformM"] = m
        return drawSceneGraph(surface, spriteSg, cameraM, lighting)

    frame = 0
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        times = drawSprite(screen, frame)
        # print("project %f, cull %f, sort %f, draw %f" % (times[0], times[1], times[2], times[3]))

        pygame.display.flip()
        frame += 1
