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

def getTranslationMatrix(dx, dy, dz):
        return [1.0, 0.0, 0.0, float(dx),
                0.0, 1.0, 0.0, float(dy),
                0.0, 0.0, 1.0, float(dz),
                0.0, 0.0, 0.0, 1.0,]

def getScalingMatrix(sx, sy, sz):
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

# FILE IO

def loadObjFile(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    vertices = []
    triangles = []
    for line in content:
        if line.startswith("v "):
            tokens = line.split()
            vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
        elif line.startswith("f "):
            indices = []
            tokens = line.split()[1:]
            for faceToken in tokens:
                indices.append(int(faceToken.split("/")[0]) - 1)
            if len(indices) == 3:
                triangles.append((indices[0], indices[1], indices[2]))
            elif len(indices) == 4:
                triangles.append((indices[0], indices[1], indices[2]))
                triangles.append((indices[2], indices[3], indices[0]))
    print("--- loaded %s: %d vertices, %d triangles" % (fname, len(vertices), len(triangles)))
    return {"verts" : vertices, "tris" : triangles}

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
    modelColor = modelInstance["color"]

    times = []
    modelVerts = model["verts"]
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
        for i in range(3):
            v0 = worldVerts[tri[i]]
            p0 = (v0[0]/-v0[2], v0[1]/-v0[2]) # perspective divide
            x1 = o_x + p0[0] * o_x
            y1 = o_y - p0[1] * o_y * (width/height)
            points.append((int(x1), int(y1)))
        if not usePrecompColors:
            # Dynamic lighting
            normal = normVec(normals[idx])
            lightNormalDotProduct = projLight[0]*normal[0]+projLight[1]*normal[1]+projLight[2]*normal[2]
            intensity = min(1, max(0, ambient + diffuse * lightNormalDotProduct))
            lightedColor = [intensity * modelColor[0], intensity * modelColor[1], intensity * modelColor[2]]
        else:
            lightedColor = modelColor

        pygame.draw.polygon(surface, lightedColor, points)
    times.append(time.time() - st) # drawing time
    return times

def drawModelList(surface, modelList, cameraM, lighting):
    """return times {project, cull, draw}"""
    projPositions = []
    for i in range(len(modelList)):
        pos = modelList[i]["pos"]
        pos4 = (pos[0], pos[1], pos[2], 1)
        projPositions.append(vecMatMult(pos4, cameraM))
    posIdcs = [*range(len(modelList))]
    def _sortByZ(i):
        return projPositions[i][2]
    posIdcs.sort(key=_sortByZ, reverse=False)

    times = []
    for i in range(len(modelList)):
        modelInstance = modelList[posIdcs[i]]
        modelM = modelInstance["matrix"]
        modelM = matMatMult(getTranslationMatrix(*modelInstance["pos"]), modelM)
        modelM = matMatMult(cameraM, modelM)
        curTimes = drawModelFilled(surface, modelInstance, cameraM, modelM, lighting)
        if len(times) == 0:
            times = curTimes
        else:
            for i in range(len(times)):
                times[i] += curTimes[i]
    return times

# MAIN

RGB_BLACK = (0, 0, 0)
RGB_DARKGREEN = (0, 128, 0)

def precomputeColors(instance):
    model = instance["model"]
    instance["precompColors"] = True

if __name__ == '__main__':
    teapot = loadObjFile("teapot.obj") # teapot-low.obj

    pygame.init()

    size = width, height = 800, 600
    o_x = width/2
    o_y = height/2

    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("PyRasterize")

    done = False
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 30)

    lighting = {"lightDir" : (1, 0, 0), "ambient": 0.3, "diffuse": 0.7}

    def getFourStaticPots():
        scaleRotM = matMatMult(getRotateXMatrix(-math.pi/2), getScalingMatrix(0.125, 0.125, 0.125))
        d = 2
        mlist = [
            { "model": teapot, "pos": (-d, 0, -d), "matrix": scaleRotM, "color": (255, 0, 0) },
            { "model": teapot, "pos": (-d, 0,  d), "matrix": scaleRotM, "color": (0, 255, 0) },
            { "model": teapot, "pos": (d,  0, -d), "matrix": scaleRotM, "color": (0, 0, 255) },
            { "model": teapot, "pos": (d,  0,  d), "matrix": scaleRotM, "color": (255, 255, 255) },]
        precomputeColors(mlist[0])
        precomputeColors(mlist[1])
        return mlist
    fourStaticPots = getFourStaticPots()
    def drawFourStaticPotsRotatingCamera(surface, frame):
        angle = degToRad(frame)
        cameraM = getCameraTransform((degToRad(20), angle, 0), (0, -2.5, -7.5))
        drawCoordGrid(surface, cameraM, RGB_DARKGREEN)
        return drawModelList(surface, fourStaticPots, cameraM, lighting)

    def getModelCenterPos(model):
        avg = [0, 0, 0]
        for v in model["verts"]:
            for i in range(3):
                avg[i] += v[i]
        for i in range(3):
            avg[i] /= len(model["verts"])
        return avg
    teapotAdjust = mulVec(-1, getModelCenterPos(teapot))

    singleRotatingPot = [{ "model": teapot, "pos": (0, 0, 0), "matrix": None, "color": (255, 0, 0) }]
    def drawSingleRotatingPotFixedCamera(surface, frame):
        angle = degToRad(frame)
        cameraM = getCameraTransform((degToRad(20), 0, 0), (0, -2.5, -7.5))
        drawCoordGrid(surface, cameraM, RGB_DARKGREEN)
        m = matMatMult(getRotateXMatrix(-math.pi/2), getTranslationMatrix(*teapotAdjust))
        m = matMatMult(getScalingMatrix(0.25, 0.25, 0.25), m)
        m = matMatMult(getRotateXMatrix(angle), m)
        m = matMatMult(getRotateYMatrix(angle), m)
        m = matMatMult(getRotateZMatrix(angle), m)
        singleRotatingPot[0]["matrix"] = m
        return drawModelList(surface, singleRotatingPot, cameraM, lighting)

    frame = 0
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        times = drawFourStaticPotsRotatingCamera(screen, frame)
        # times = drawSingleRotatingPotFixedCamera(screen, frame)
        print("project %f, cull %f, sort %f, draw %f" % (times[0], times[1], times[2], times[3]))

        pygame.display.flip()
        frame += 1
