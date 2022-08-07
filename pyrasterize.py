import pygame, math, sys

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
                triangles.append((indices[0], indices[2], indices[3]))
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
    idcs = []
    i = -1
    for tri in tris:
        i += 1
        v0 = worldVerts[tri[0]]
        v1 = worldVerts[tri[1]]
        v2 = worldVerts[tri[2]]
        if v0[2] == 0 or v1[2] == 0 or v2[2] == 0:
            continue
        idcs.append(i)
        # nearClip = 0.5
        # if v0[2] <= nearClip and v1[2] <= nearClip and v2[2] <= nearClip:
        #     continue
        # viewPointToTriVec = subVec(v0, viewPoint)
        # normal = crossProduct(subVec(v1, v0), subVec(v2, v0))
        # if dotProduct(viewPointToTriVec, normal) > 0:
        #     idcs.append(i)
    return idcs

def perspDiv(vert):
    return (vert[0] / vert[2], vert[1] / vert[2])

if __name__ == '__main__':
    teapot = loadObjFile("C:/svn/pyrasterize/teapot-low.obj")

    pygame.init()

    size = width, height = 800, 600
    o_x = width/2
    o_y = height/2

    RGB_BLACK = (0, 0, 0)
    RGB_WHITE = (255, 255, 255)
    RGB_DARKGREEN = (0, 128, 0)

    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("PyRasterize")

    done = False
    clock = pygame.time.Clock()

    def drawEdge(p0, p1, color):
        x1 = o_x + p0[0] * o_x
        y1 = o_y - p0[1] * o_y * (width/height)
        x2 = o_x + p1[0] * o_x
        y2 = o_y - p1[1] * o_y * (width/height)
        pygame.draw.line(screen, color, (x1, y1), (x2, y2), 1)

    def getTransform(rot, tran):
        m = getRotateXMatrix(rot[0])
        m = matMatMult(getRotateYMatrix(rot[1]), m)
        m = matMatMult(getRotateZMatrix(rot[2]), m)
        m = matMatMult(getTranslationMatrix(*tran), m)
        return m

    def drawGround(m, color):
        pass

    def drawMesh(m, mesh, color):
        worldVerts = projectVerts(m, mesh["verts"])
        drawTriIdcs = cullBackfaces((0, 0, 0), mesh["tris"], worldVerts)
        for idx in drawTriIdcs:
            tri = mesh["tris"][idx]
            p0 = perspDiv(worldVerts[tri[0]])
            p1 = perspDiv(worldVerts[tri[1]])
            p2 = perspDiv(worldVerts[tri[2]])
            drawEdge(p0, p1, color)
            drawEdge(p0, p2, color)
            drawEdge(p1, p2, color)

    frame = 0
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        angle = math.pi / 180 * frame
        rot = (angle, 0, 0)
        m = getTransform(rot, (0, 0, 25))
        drawGround(m, RGB_DARKGREEN)
        drawMesh(m, teapot, RGB_WHITE)

        pygame.display.flip()
        frame += 1
