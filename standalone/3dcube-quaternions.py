import pygame
import math

# OpenGL coord system: -z goes into the screen
# 8 vertices form corners of the unit cube
cubeVerts = [
    (0.5,  0.5, 0.5),  # front top right     0
    (0.5, -0.5, 0.5),  # front bottom right  1
    (-0.5, -0.5, 0.5),  # front bottom left   2
    (-0.5,  0.5, 0.5),  # front top left      3
    (0.5,  0.5, -0.5),  # back top right      4
    (0.5, -0.5, -0.5),  # back bottom right   5
    (-0.5, -0.5, -0.5),  # back bottom left    6
    (-0.5,  0.5, -0.5)  # back top left       7
    ]

cubeIdxs = [
    (0, 1, 3),  # front face  3 0
    (2, 3, 1),  # .           2 1
    (3, 2, 7),  # left face   7 3
    (6, 7, 2),  # .           6 2
    (4, 5, 0),  # right face  0 4
    (1, 0, 5),  # .           1 5
    (4, 0, 7),  # top face    7 4
    (3, 7, 0),  # .           3 0
    (1, 5, 2),  # bottom face 2 1
    (6, 2, 5),  # .           6 5
    (7, 6, 4),  # back face   4 7
    (5, 4, 6)   # .           5 6
    ]


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
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3],
            m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7] * v[3],
            m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11] * v[3],
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


def perspectiveDivide(srcV):
    """
    Takes vec4
    Returns vec2
    """
    dstV = []
    for vec4 in srcV:
        dstV.append((vec4[0] / vec4[2], vec4[1] / vec4[2]))
    return dstV


def cullBackfaces(viewPoint, srcIdcs, worldVerts):
    idcs = []
    i = 0
    for tri in srcIdcs:
        v0 = worldVerts[tri[0]]
        v1 = worldVerts[tri[1]]
        v2 = worldVerts[tri[2]]
        viewPointToTriVec = subVec(v0, viewPoint)
        normal = crossProduct(subVec(v1, v0), subVec(v2, v0))
        if dotProduct(viewPointToTriVec, normal) > 0:
            idcs.append(i)
        i += 1
    return idcs


def get_rot_quat(angle_x, angle_y, angle_z):
    """
    Returns a quaternion for rotating around the x, y, and z axes in that order

    Parameters:
    - angle_x: Rotation around the x-axis in radians.
    - angle_y: Rotation around the y-axis in radians.
    - angle_z: Rotation around the z-axis in radians.

    Returns:
    - q: Combined rotation quaternion [w, x, y, z].
    """

    # Quaternion for rotation around the x-axis
    q_x = [math.cos(angle_x / 2), math.sin(angle_x / 2), 0, 0]

    # Quaternion for rotation around the y-axis
    q_y = [math.cos(angle_y / 2), 0, math.sin(angle_y / 2), 0]

    # Quaternion for rotation around the z-axis
    q_z = [math.cos(angle_z / 2), 0, 0, math.sin(angle_z / 2)]

    # Combined rotation: First z, then y, then x (applied in reverse order)
    q = quat_mul(q_z, quat_mul(q_y, q_x))

    return q


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]


def rotate_vector_by_quaternion(vector, quat):
    """
    Rotates a 3-component vector by a given quaternion.

    Parameters:
    - vector: A list representing the 3D vector [x, y, z].
    - quaternion: A list representing the quaternion [w, x, y, z].

    Returns:
    - rotated_vector: The rotated 3D vector as a list [x', y', z'].
    """
    # Convert the vector to a quaternion with a zero scalar part
    vector_quat = [0, vector[0], vector[1], vector[2]]

    # Compute the conjugate of the quaternion
    q_conjug = [quat[0], -quat[1], -quat[2], -quat[3]]

    # Apply the rotation: q * v * q_conjugate
    rotated_quat = quat_mul(quat_mul(quat, vector_quat), q_conjug)

    # Return the vector part of the resulting quaternion
    rotated_vector = rotated_quat[1:]  # Extract the x, y, z components

    return rotated_vector


def quaternion_to_matrix(quaternion):
    """
    Converts a quaternion into a 4x4 rotation matrix.

    Parameters:
    - quaternion: A list representing the quaternion [w, x, y, z].

    Returns:
    - matrix: A 4x4 list of lists representing the rotation matrix.
    """
    w, x, y, z = quaternion
    return [
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0,
        2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, 0,
        2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0,
        0, 0, 0, 1
    ]


def quaternion_dot(q1, q2):
    """Returns the dot product of two quaternions."""
    return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]


def quaternion_slerp(q1, q2, t):
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions.

    Parameters:
    - q1: The starting quaternion as a list [w, x, y, z].
    - q2: The ending quaternion as a list [w, x, y, z].
    - t: The interpolation parameter between 0 and 1.

    Returns:
    - q_interp: The interpolated quaternion as a list [w, x, y, z].
    """

    # Compute the dot product
    dot = quaternion_dot(q1, q2)

    # Clamp dot product to avoid numerical errors
    # due to floating point precision
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0

    # If the dot product is negative, invert one quaternion
    # to take the shorter path
    if dot < 0.0:
        q2 = [-q for q in q2]
        dot = -dot

    # If the quaternions are very close, linearly
    # interpolate to avoid division by zero
    if dot > 0.9995:
        q_interp = [(1.0 - t) * q1[i] + t * q2[i] for i in range(4)]
        return normalize_quaternion(q_interp)

    # Calculate the angle between the quaternions
    theta_0 = math.acos(dot)
    theta = theta_0 * t

    # Compute the second quaternion in the interpolation
    q2_orthogonal = [q2[i] - q1[i] * dot for i in range(4)]
    q2_orthogonal = normalize_quaternion(q2_orthogonal)

    # Perform the interpolation
    q_interp = [
        math.cos(theta) * q1[i] + math.sin(theta) * q2_orthogonal[i]
        for i in range(4)
    ]

    return q_interp


def normalize_quaternion(q):
    """Normalizes a quaternion to unit length."""
    length = math.sqrt(sum(x * x for x in q))
    return [x / length for x in q]


if __name__ == '__main__':
    pygame.init()

    size = width, height = 800, 600
    o_x = width/2
    o_y = height/2

    RGB_BLACK = (0, 0, 0)
    RGB_WHITE = (255, 255, 255)

    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Cube")

    done = False
    clock = pygame.time.Clock()

    def drawEdge(p0, p1):
        x1 = o_x + p0[0] * o_x
        y1 = o_y - p0[1] * o_y * (width/height)
        x2 = o_x + p1[0] * o_x
        y2 = o_y - p1[1] * o_y * (width/height)
        pygame.draw.line(screen, RGB_WHITE, (x1, y1), (x2, y2), 1)

    def drawObject(culledIdxList, idcs, perspVerts):
        for idx in culledIdxList:
            idc = idcs[idx]
            i0 = idc[0]
            i1 = idc[1]
            i2 = idc[2]
            p0 = perspVerts[i0]
            p1 = perspVerts[i1]
            p2 = perspVerts[i2]
            drawEdge(p0, p1)
            drawEdge(p0, p2)
            drawEdge(p1, p2)

    frame = 0
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill(RGB_BLACK)

        angle = math.pi / 180 * frame

        # With individual rotation matrices it would be:
        # m = getRotateXMatrix(angle)
        # m = matMatMult(getRotateYMatrix(angle), m)
        # m = matMatMult(getRotateZMatrix(angle), m)

        m = quaternion_to_matrix(get_rot_quat(angle, angle, angle))
        m = matMatMult(getTranslationMatrix(0, 0, -3), m)

        worldVerts = projectVerts(m, cubeVerts)
        drawIdxList = cullBackfaces((0, 0, 0), cubeIdxs, worldVerts)
        perspVerts = perspectiveDivide(worldVerts)
        drawObject(drawIdxList, cubeIdxs, perspVerts)

        pygame.display.flip()
        frame += 1
