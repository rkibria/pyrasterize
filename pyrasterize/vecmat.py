#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vectors, matrices and other math
"""

import math

def norm_vec3(v_3):
    """Return normalized vec3"""
    mag = v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2]
    if mag == 0:
        return [0, 0, 0]
    mag = mag ** -0.5 # 1.0 / math.sqrt(mag)
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

def get_persp_m4(d, ar):
    """Return perspective transformation matrix"""
    return [d,    0.0,   0.0,  0.0,
            0.0,  d*ar,  0.0,  0.0,
            0.0,  0.0,   1.0,  0.0,
            0.0,  0.0,   1.0,  0.0]

def get_view_plane_from_fov(fov):
    """Return view plane distance"""
    return 1 / math.tan(deg_to_rad(fov / 2))
