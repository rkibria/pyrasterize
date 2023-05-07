#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vectors, matrices and other math
"""

from collections import deque
import math

def sub_vec3(v_1, v_2):
    """Return v1 - v2"""
    return [v_1[0] - v_2[0], v_1[1] - v_2[1], v_1[2] - v_2[2]]

def cross_vec3(a, b):
    """Return vec3 result of cross product of 2 vec3's"""
    return [a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]

def dot_product_vec3(a, b):
    """Return dot product of vec3"""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def mag_sq_vec3(v_3):
    """Return squared magnitude of vec3"""
    return (v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2])

def mag_vec3(v_3):
    """Return magnitude of vec3"""
    return (v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2]) ** 0.5

def norm_vec3(v_3):
    """Return normalized vec3"""
    mag = v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2]
    if mag == 0:
        return [0, 0, 0]
    mag = mag ** -0.5 # 1.0 / math.sqrt(mag)
    return [v_3[0] * mag, v_3[1] * mag, v_3[2] * mag]

def norm_vec3_from_vec4(v_4):
    """
    Take first elements of vec4 as a vec3, normalize,
    return vec4 with normalized and 4th element from original
    """
    mag = v_4[0]*v_4[0] + v_4[1]*v_4[1] + v_4[2]*v_4[2]
    if mag == 0:
        return [0, 0, 0]
    mag = mag ** -0.5 # 1.0 / math.sqrt(mag)
    return [v_4[0] * mag, v_4[1] * mag, v_4[2] * mag, v_4[3]]

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

def rad_to_deg(radians):
    """Return radians converted to degrees"""
    return radians / math.pi * 180

def get_persp_m4(d, ar):
    """Return perspective transformation matrix"""
    return [d,    0.0,   0.0,  0.0,
            0.0,  d*ar,  0.0,  0.0,
            0.0,  0.0,   1.0,  0.0,
            0.0,  0.0,   1.0,  0.0]

def get_view_plane_from_fov(fov):
    """Return view plane distance"""
    return 1 / math.tan(deg_to_rad(fov / 2))

def ray_sphere_intersect(r_orig3, r_dir3, sph_orig3, sph_r, t_min=0.001, t_max=10**6):
    """Return ray direction multi t if ray intersects sphere or None"""
    oc = [r_orig3[0] - sph_orig3[0], r_orig3[1] - sph_orig3[1], r_orig3[2] - sph_orig3[2]]
    a = r_dir3[0] * r_dir3[0] + r_dir3[1] * r_dir3[1] + r_dir3[2] * r_dir3[2]
    b = oc[0] * r_dir3[0] + oc[1] * r_dir3[1] + oc[2] * r_dir3[2]
    c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sph_r * sph_r
    discriminant = b * b - a * c
    if discriminant > 0:
        sqrt_discriminant = discriminant ** 0.5
        temp_1 = (-b - sqrt_discriminant) / a
        if temp_1 < t_max and temp_1 > t_min:
            return temp_1
        temp_2 = (-b + sqrt_discriminant) / a
        if temp_2 < t_max and temp_2 > t_min:
            return temp_2
    return None

def mouse_pos_to_ray(pos, scr_size):
    """Get ray vec3 into scene from mouse position"""
    ndc_x = 2 * pos[0] / scr_size[0] - 1
    ndc_y = 1 - (2 * pos[1]) / scr_size[1]
    return norm_vec3([ndc_x, ndc_y, -1])

def get_rot_xyz_m4(x, y, z):
    """Return rotation order xyz matrix"""
    m = get_rot_z_m4(z)
    m = mat4_mat4_mul(get_rot_y_m4(y), m)
    return mat4_mat4_mul(get_rot_x_m4(x), m)

def get_simple_camera_m(cam):
    """Return matrix based on camera dict(rot = [x,y,z],pos = [x,y,z])"""
    cam_pos = cam["pos"]
    cam_rot = cam["rot"]
    # Operation order: rot_x, rot_y, rot_z, translate
    camera_m = get_transl_m4(-cam_pos[0], -cam_pos[1], -cam_pos[2])
    camera_m = mat4_mat4_mul(get_rot_z_m4(-cam_rot[2]), camera_m)
    camera_m = mat4_mat4_mul(get_rot_y_m4(-cam_rot[1]), camera_m)
    camera_m = mat4_mat4_mul(get_rot_x_m4(-cam_rot[0]), camera_m)
    return camera_m

def get_triangle_area(a, b, c):
    """Area of triangle formed by 3 vec3s"""
    t = cross_vec3(sub_vec3(b, a), sub_vec3(c, a))
    return 0.5 * mag_vec3(t)

def get_2d_triangle_area(v1, v2, v3):
    """Area of triangle formed by 3 vec2s"""
    v_12 = (v2[0] - v1[0], v2[1] - v1[1]) # a,b
    v_13 = (v3[0] - v1[0], v3[1] - v1[1]) # c,d
    cross = v_12[0] * v_13[1] - v_12[1] * v_13[0] # ad-bc
    return abs(cross / 2)

def get_vec2_triangle_centroid(v_a, v_b, v_c):
    """Get centroid point of a two-dimensional triangle"""
    cx = (v_a[0] + v_b[0] + v_c[0]) / 3
    cy = (v_a[1] + v_b[1] + v_c[1]) / 3
    return (cx, cy)

def get_vec3_triangle_centroid(v_a, v_b, v_c):
    """Get centroid point of a three-dimensional triangle"""
    cx = (v_a[0] + v_b[0] + v_c[0]) / 3
    cy = (v_a[1] + v_b[1] + v_c[1]) / 3
    cz = (v_a[2] + v_b[2] + v_c[2]) / 3
    return [cx, cy, cz]

def get_average_color(c_0, c_1, c_2):
    """Average of 3 RGB colors"""
    return [(i + j + k) / 3.0 for i, j, k in zip(c_0, c_1, c_2)]

class Barycentric2dTriangle:
    def __init__(self, v_a, v_b, v_c) -> None:
        self.v_a = v_a
        self.v_b = v_b
        self.v_c = v_c

        # v_ab = vecmat.sub_vec3(v_b, v_a)
        v_ab_0 = v_b[0] - v_a[0]
        v_ab_1 = v_b[1] - v_a[1]
        # v_ac = vecmat.sub_vec3(v_c, v_a)
        v_ac_0 = v_c[0] - v_a[0]
        v_ac_1 = v_c[1] - v_a[1]
        # v_n = vecmat.cross_vec3(v_ab, v_ac)
        self.v_n = v_ab_0 * v_ac_1 - v_ab_1 * v_ac_0
        # area_full_sq = vecmat.dot_product_vec3(v_n, v_n)
        self.area_sq = self.v_n ** 2

        # p = (x, y, 0)
        # v_bc = vecmat.sub_vec3(v_c, v_b)
        self.v_bc_0 = v_c[0] - v_b[0]
        self.v_bc_1 = v_c[1] - v_b[1]
        # v_ca = vecmat.sub_vec3(v_a, v_c)
        self.v_ca_0 = v_a[0] - v_c[0]
        self.v_ca_1 = v_a[1] - v_c[1]

    def get_uvw(self, x, y):
        # v_bp = vecmat.sub_vec3(p, v_b)
        v_bp_0 = x - self.v_b[0]
        v_bp_1 = y - self.v_b[1]
        # v_n1 = vecmat.cross_vec3(v_bc, v_bp)
        v_n1 = self.v_bc_0 * v_bp_1 - self.v_bc_1 * v_bp_0
        # u = vecmat.dot_product_vec3(v_n, v_n1) / area_full_sq
        u = (self.v_n * v_n1) / self.area_sq
        # v_cp = vecmat.sub_vec3(p, v_c)
        v_cp_0 = x - self.v_c[0]
        v_cp_1 = y - self.v_c[1]
        # v_n2 = vecmat.cross_vec3(v_ca, v_cp)
        v_n2 = self.v_ca_0 * v_cp_1 - self.v_ca_1 * v_cp_0
        # v = vecmat.dot_product_vec3(v_n, v_n2) / area_full_sq
        v = (self.v_n * v_n2) / self.area_sq
        return u, v, 1 - u - v

class TextureInterpolation:
    def __init__(self, uv, mip_textures, z_order, mip_dist) -> None:
        self.uv = uv
        self.uv_extent = TextureInterpolation.get_uv_extent(*uv)
        num_mip_levels = len(mip_textures)
        mip_level = num_mip_levels * abs(z_order) / mip_dist
        mip_level = max(0, min(num_mip_levels - 1, int(mip_level)))
        self.texture = mip_textures[mip_level]
        self.tex_w,self.tex_h = len(self.texture[0]), len(self.texture)
        
    @staticmethod
    def get_uv_extent(uv_0, uv_1, uv_2):
        s_min = min(uv_0[0], uv_1[0], uv_2[0])
        s_max = max(uv_0[0], uv_1[0], uv_2[0])
        t_min = min(uv_0[1], uv_1[1], uv_2[1])
        t_max = max(uv_0[1], uv_1[1], uv_2[1])
        return s_max - s_min, t_max - t_min

    def get_color(self, u, v, w):
        s = self.uv[0][0] * u + self.uv[1][0] * v + self.uv[2][0] * w
        t = self.uv[0][1] * u + self.uv[1][1] * v + self.uv[2][1] * w
        s_i = min(self.tex_w - 1, max(0, int(s * self.tex_w)))
        t_i = min(self.tex_h - 1, max(0, int(t * self.tex_h)))
        color = self.texture[t_i][s_i]
        return color

def subdivide_2d_triangle(v_a, v_b, v_c, callback):
    """
    Callback arguments: vec2: point, vec2: point, vec2: point, int: iteration
    If returns true, don't split current triangle further
    """
    tri_stack = deque()
    tri_stack.append((v_a, v_b, v_c, 0))
    while tri_stack:
        tri = tri_stack.popleft()
        if callback(*tri):
            continue

        # Split and recurse
        iteration = tri[3]
        iteration += 1
        v_01 = [tri[1][0] - tri[0][0], tri[1][1] - tri[0][1]]
        v_02 = [tri[2][0] - tri[0][0], tri[2][1] - tri[0][1]]

        v_01_h = [v_01[0] / 2, v_01[1] / 2]
        v_02_h = [v_02[0] / 2, v_02[1] / 2]

        h_01 = [tri[0][0] + v_01_h[0], tri[0][1] + v_01_h[1]]
        h_02 = [tri[0][0] + v_02_h[0], tri[0][1] + v_02_h[1]]
        h_12 = [tri[0][0] + v_01_h[0] + v_02_h[0], tri[0][1] + v_01_h[1] + v_02_h[1]]

        tri_stack.append((tri[0], h_01, h_02, iteration))
        tri_stack.append((h_02, h_01, h_12, iteration))
        tri_stack.append((h_01, tri[1], h_12, iteration))
        tri_stack.append((h_02, h_12, tri[2], iteration))
