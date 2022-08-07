import pygame, math, sys

NEAR_CLIP_PLANE = -0.5

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

def subVec(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def dotProduct(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def crossProduct(a, b):
    return (a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0])

def matMatMult(m1, m2):
    """
    Rows and columns correspond to as written in source
    """
    newM = []
    for r in range(4):
        for c in range(4):
            s = 0
            for i in range(4):
                # multiply values in r-th row, c-th column
                v1 = m1[4 * r + i]
                v2 = m2[4 * i + c]
                s += v1 * v2
            newM.append(s)
    return newM

def vecMatMult(v, m):
        return (
                m[ 0] * v[0] + m[ 1] * v[1] + m[ 2] * v[2] + m[ 3] * v[3],
                m[ 4] * v[0] + m[ 5] * v[1] + m[ 6] * v[2] + m[ 7] * v[3],
                m[ 8] * v[0] + m[ 9] * v[1] + m[10] * v[2] + m[11] * v[3],
                m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3],
                )

def getTranslationMatrix(dx, dy, dz):
        return [
                1.0, 0.0, 0.0, float(dx),
                0.0, 1.0, 0.0, float(dy),
                0.0, 0.0, 1.0, float(dz),
                0.0, 0.0, 0.0, 1.0,
                ]

def getScalingMatrix(sx, sy, sz):
        return [
                float(sx), 0.0, 0.0, 0,
                0.0, float(sy), 0.0, 0,
                0.0, 0.0, float(sz), 0,
                0.0, 0.0, 0.0, 1.0,
                ]

def getRotateXMatrix(phi):
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        return [
                1.0,    0.0,            0.0,            0.0,
                0.0,    cos_phi,        -sin_phi,       0.0,
                0.0,    sin_phi,        cos_phi,        0.0,
                0.0,    0.0,            0.0,            1.0,
                ]

def getRotateYMatrix(phi):
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        return [
                cos_phi,        0.0,            sin_phi,        0.0,
                0.0,            1.0,            0.0,            0.0,
                -sin_phi,       0.0,            cos_phi,        0.0,
                0.0,            0.0,            0.0,            1.0,
                ]

def getRotateZMatrix(phi):
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        return [
                cos_phi,        -sin_phi,       0.0,            0.0,
                sin_phi,        cos_phi,        0.0,            0.0,
                0.0,            0.0,            1.0,            0.0,
                0.0,            0.0,            0.0,            1.0,
                ]

def projectVerts(m, srcV):
    """
    Takes vec3 verts
    Returns vec4
    """
    dstV = []
    for vec in srcV:
        v4 = (vec[0], vec[1], vec[2], 1)
        pvec = vecMatMult(v4, m)
        dstV.append(pvec)
    return dstV

def cullBackfaces(viewPoint, tris, worldVerts):
    """
    """
    idcs = []
    culled = []
    i = -1
    for tri in tris:
        i += 1
        v0 = worldVerts[tri[0]]
        v1 = worldVerts[tri[1]]
        v2 = worldVerts[tri[2]]
        if v0[2] >= NEAR_CLIP_PLANE or v1[2] >= NEAR_CLIP_PLANE or v2[2] >= NEAR_CLIP_PLANE:
            continue
        viewPointToTriVec = subVec(v0, viewPoint)
        normal = crossProduct(subVec(v1, v0), subVec(v2, v0))
        isVisible = (dotProduct(viewPointToTriVec, normal) < 0)
        if isVisible:
            idcs.append(i)
        else:
            culled.append(i)
    return (idcs, culled)

def perspDiv(vert):
    z = -vert[2]
    return (vert[0] / z, vert[1] / z)

if __name__ == '__main__':
    teapot = loadObjFile("teapot.obj") # teapot-low.obj

    pygame.init()

    size = width, height = 800, 600
    o_x = width/2
    o_y = height/2

    RGB_BLACK = (0, 0, 0)
    RGB_WHITE = (255, 255, 255)
    RGB_DARKGREEN = (0, 128, 0)
    RGB_RED = (255, 0, 0)
    RGB_GRAY = (200, 200, 200)

    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("PyRasterize")

    done = False
    clock = pygame.time.Clock()

    def drawEdge(p0, p1, color):
        x1 = o_x + p0[0] * o_x
        y1 = o_y - p0[1] * o_y * (width/height)
        x2 = o_x + p1[0] * o_x
        y2 = o_y - p1[1] * o_y * (width/height)
        pygame.draw.aaline(screen, color, (x1, y1), (x2, y2), 1)

    def getTransform(rot, tran):
        m = getRotateXMatrix(rot[0])
        m = matMatMult(getRotateYMatrix(rot[1]), m)
        m = matMatMult(getRotateZMatrix(rot[2]), m)
        m = matMatMult(getTranslationMatrix(*tran), m)
        return m

    def drawGround(m, color):
        darkColor = (color[0]/2, color[1]/2, color[2]/2)
        def gridLine(v0, v1, color):
            v0 = vecMatMult(v0, m)
            v1 = vecMatMult(v1, m)
            if v0[2] >= NEAR_CLIP_PLANE or v1[2] >= NEAR_CLIP_PLANE:
                return
            p0 = perspDiv(v0)
            p1 = perspDiv(v1)
            drawEdge(p0, p1, color)
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

    def drawMesh(drawTriIdcs, worldVerts, tris, color):
        for idx in drawTriIdcs:
            tri = tris[idx]
            points = []
            for i in range(3):
                p0 = perspDiv(worldVerts[tri[i]])
                x1 = o_x + p0[0] * o_x
                y1 = o_y - p0[1] * o_y * (width/height)
                points.append((x1, y1))
            pygame.draw.aalines(screen, color, True, points)

    font = pygame.font.Font(None, 30)

    legend = "BACK-FACE CULLING optimizes rendering by removing triangles that"
    title1 = font.render(legend, True, RGB_WHITE)
    legend = "are facing away from the viewpoint."
    title2 = font.render(legend, True, RGB_WHITE)

    frame = 0
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        def degToRad(d):
            return d * (math.pi / 180)

        def radToDeg(r):
            deg = r / (math.pi / 180)
            deg = divmod(deg, 360)[1]
            return deg

        angle = degToRad(frame)
        rot = (degToRad(20), angle, 0)
        tran = (0, -2.5, -7.5)

        m = getTransform(rot, tran)
        drawGround(m, RGB_DARKGREEN)
        mt = getScalingMatrix(0.25, 0.25, 0.25)
        mt = matMatMult(getRotateXMatrix(-math.pi/2), mt)
        m = matMatMult(m, mt)

        worldVerts = projectVerts(m, teapot["verts"])
        drawIdcs, cullIdcs = cullBackfaces((0, 0, 0), teapot["tris"], worldVerts)

        # legend = "transl (%.1f, %.1f, %.1f) rot[deg] (%.1f, %.1f, %.1f)" % (tran[0], tran[1], tran[2],
        #     radToDeg(rot[0]), radToDeg(rot[1]), radToDeg(rot[2]))
        if frame % 10 == 0:
            legend = "triangles drawn: %d" % (len(drawIdcs))
            text1 = font.render(legend, True, RGB_GRAY)
            legend = "triangles removed: %d" % (len(cullIdcs))
            text2 = font.render(legend, True, RGB_RED)
        screen.blit(title1, (30, 20))
        screen.blit(title2, (30, 50))
        screen.blit(text1, (30, 100))
        screen.blit(text2, (300, 100))

        seconds = int(frame/30)
        if seconds % 2 == 0:
            drawMesh(cullIdcs, worldVerts, teapot["tris"], RGB_RED)

        drawMesh(drawIdcs, worldVerts, teapot["tris"], RGB_WHITE)

        pygame.display.flip()
        frame += 1
