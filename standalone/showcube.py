import pygame, math, random

# Unit cube centered at origin

# OpenGL coord system: -z goes into the screen
# 8 vertices form corners of the unit cube
cubeVertices = [
( 0.5,  0.5, 0.5),  # front top right     0
( 0.5, -0.5, 0.5),  # front bottom right  1
(-0.5, -0.5, 0.5),  # front bottom left   2
(-0.5,  0.5, 0.5),  # front top left      3
( 0.5,  0.5, -0.5), # back top right      4
( 0.5, -0.5, -0.5), # back bottom right   5
(-0.5, -0.5, -0.5), # back bottom left    6
(-0.5,  0.5, -0.5)  # back top left       7
]

cubeFaces = [
(0, 1, 3), # front face  3 0
(2, 3, 1), #             2 1
(3, 2, 7), # left face   7 3
(6, 7, 2), #             6 2
(4, 5, 0), # right face  0 4
(1, 0, 5), #             1 5
(4, 0, 7), # top face    7 4
(3, 7, 0), #             3 0
(1, 5, 2), # bottom face 2 1
(6, 2, 5), #             6 5
(7, 6, 4), # back face   4 7
(5, 4, 6)  #             5 6
]

def subVec(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def dotProduct(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def crossProduct(a, b):
    return (a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0])

def get_unit_m4():
    """Return 4x4 unit matrix"""
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]

def matMatMult(m1, m2):
    """
    m1 on left, m2 on right
    Matrix list layout: [row 1],[row 2],[row 3],[row 4]
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

def vecMatMult(m, v):
        """
        4x4 matrix on the left side
        4x1 (column) vector is on the right side
        (other way around doesn't work)
        """
        return (
                m[ 0] * v[0] + m[ 1] * v[1] + m[ 2] * v[2] + m[ 3] * v[3], # (row 1 of m) X (col 1 of v)
                m[ 4] * v[0] + m[ 5] * v[1] + m[ 6] * v[2] + m[ 7] * v[3], # (row 2 of m) X (col 1 of v)
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

def getScaleMatrix(s_x, s_y, s_z):
    return [float(s_x), 0.0,       0.0,       0.0,
            0.0,       float(s_y), 0.0,       0.0,
            0.0,       0.0,       float(s_z), 0.0,
            0.0,       0.0,       0.0,        1.0]

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

def getOrthoMatrix():
        return [
                1.0,            0.0,       0.0,            0.0,
                0.0,            1.0,       0.0,            0.0,
                0.0,            0.0,       0.0,            0.0,
                0.0,            0.0,       0.0,            1.0,
                ]

def projectVerts(m, srcV):
    """
    Takes vec3 or vec4 verts
    Returns vec4
    """
    dstV = []
    for vec in srcV:
        v4 = (vec[0], vec[1], vec[2], 1 if len(vec) < 4 else vec[3])
        pvec = vecMatMult(m, v4)
        dstV.append(pvec)
    return dstV

def cullBackfaces(viewPoint, srcIdcs, worldVerts, disable=False):
    idcs = []
    i = 0
    for tri in srcIdcs:
        v0 = worldVerts[tri[0]]
        v1 = worldVerts[tri[1]]
        v2 = worldVerts[tri[2]]
        viewPointToTriVec = subVec(v0, viewPoint)
        normal = crossProduct(subVec(v1, v0), subVec(v2, v0))
        if disable or dotProduct(viewPointToTriVec, normal) > 0:
            idcs.append(i)
        i += 1
    return idcs

def perspectiveDivide(srcV):
    """
    Takes vec4
    Returns vec3
    """
    dstV = []
    for vec4 in srcV:
        x = vec4[0]
        y = vec4[1]
        z = vec4[2]
        w = vec4[3]
        x /= w
        y /= w
        z /= w
        if z != 0:
            x /= z
            y /= z
        dstV.append((x, y, z))
    return dstV

def getNDCVerts(screenspaceVerts, w, h):
    result = []
    for x,y,_ in screenspaceVerts:
        result.append(((x + w/2)/w, (y + h/2)/h))
    return result

def getRasterVerts(ndcVerts, w, h):
    result = []
    for x,y in ndcVerts:
        result.append((w * x, h * (1 - y)))
    return result

def getPerspectiveMatrix(l, r, b, t, n, f):
    """
    In contrast to the other matrices this will affect the w component of the vector!
    We will also need a division by w to get a non-homogenous point.
    """
    return [
            2*n/(r-l), 0.0,       -(r+l)/(r-l),                      0.0,
            0.0,        2*n/(t-b),  -(t+b)/(t-b),                      0.0,
            0.0,        0.0,       (f+n)/(f-n), -2*f*n/(f-n),
            0.0,        0.0,       1.0,                     0.0,
            ]

def getSimplePerspectiveMatrix(d=1):
    """
    In contrast to the other matrices this will affect the w component of the vector!
    We will also need a division by w to get a non-homogenous point.
    """
    return [
            1.0, 0.0, 0.0,    0.0,
            0.0, 1.0, 0.0,    0.0,
            0.0, 0.0, 1.0,    0.0,
            0.0, 0.0, -1.0/d, 0.0,
            ]

def drawText():
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/mathematics-computing-2d-coordinates-of-3d-points.html
    # World space to camera space.
    # Camera space to screen space.
    # Screen space to NDC space.
    # NDC space to raster space.
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/projection-matrix-introduction.html

    frustumVrts = [
    (3,  3, -3),  # front top right
    (3, -3, -3),  # front bottom right
    (-3, -3, -3),  # front bottom left
    (-3, 3, -3),  # front top left
    (13,  13, -13),  # back top right
    (13, -13, -13),  # back bottom right
    (-13, -13, -13),  # back bottom left
    (-13, 13, -13),  # back top left
    ]

    viewspaceVrts = projectVerts(get_unit_m4(), frustumVrts)
    print(f"Frustum in view space: {viewspaceVrts}")

    # clipspaceVrts = projectVerts(getSimplePerspectiveMatrix(), viewspaceVrts)
    clipspaceVrts = projectVerts(getPerspectiveMatrix(-3, 3,
                                                      -3, 3,
                                                      -3, -13), viewspaceVrts)
    print(f"Frustum in clip space: {clipspaceVrts}")

    screenspaceVrts = perspectiveDivide(clipspaceVrts)
    print(f"Frustum in screen space: {screenspaceVrts}")

def drawMain():
    pygame.init()

    size = width, height = 800, 600

    RGB_BLACK = (0, 0, 0)
    RGB_WHITE = (255, 255, 255)

    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Cube")

    done = False
    clock = pygame.time.Clock()

    def drawEdge(p0, p1, color):
        # x1 = o_x + p0[0] * o_x
        # y1 = o_y - p0[1] * o_y
        # x2 = o_x + p1[0] * o_x
        # y2 = o_y - p1[1] * o_y
        x1 = p0[0]
        y1 = p0[1]
        x2 = p1[0]
        y2 = p1[1]
        pygame.draw.line(screen, color, (x1, y1), (x2, y2), 1)

    colors = [(255,0,0), (0,255,0), (0,0,255),
              (255,255,0), (0,255,255), (255,0,255),
              (255,255,255), (127,255,0), (127,0,255),
              (127,0,0), (0,127,0), (0,0,127),
              ]
    def drawObject(culledIdxList, idcs, perspVerts):
        for idx in culledIdxList:
            idc = idcs[idx]
            i0 = idc[0]
            i1 = idc[1]
            i2 = idc[2]
            p0 = perspVerts[i0]
            p1 = perspVerts[i1]
            p2 = perspVerts[i2]
            color = colors[idx % len(colors)]
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

        angle = math.pi / 180 * frame # math.pi / 180 * frame

        m = getRotateXMatrix(angle)
        m = matMatMult(getRotateYMatrix(angle), m)
        m = matMatMult(getRotateZMatrix(angle), m)
        # m = matMatMult(getTranslationMatrix(math.sin(0), math.cos(0), -3), m)
        m = matMatMult(getTranslationMatrix(1, 0, -2.5), m)
        # m = matMatMult(getPerspectiveMatrix(0.5, 5000.0, 1.0, 1.0), m)
        prePerspectiveVerts = projectVerts(m, cubeVertices)
        m = matMatMult(getSimplePerspectiveMatrix(), m)
        # m = matMatMult(getOrthoMatrix(), m)
        # operation order: (P * (T * (Z * (Y * X)))) * point => rotate first, then translate
        # Same result: ((T * Z) * (Y * X)) * point
        #              (((T * Z) * Y) * X) * point

        clipVerts = projectVerts(m, cubeVertices)
        drawIdxList = cullBackfaces((0, 0, 0), cubeFaces, clipVerts, True)
        screenspaceVerts = perspectiveDivide(clipVerts)
        ndcVerts = getNDCVerts(screenspaceVerts, 2, 2)
        rasterVerts = getRasterVerts(ndcVerts, width, height)
        drawObject(drawIdxList, cubeFaces, rasterVerts)

        pygame.display.flip()
        frame += 1

if __name__ == '__main__':
    drawText()
