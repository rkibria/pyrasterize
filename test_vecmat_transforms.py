import pytest
import math

from pyrasterize import vecmat

def test_rot_y_no_rot():
    assert vecmat.vec4_mat4_mul([0, 0, -1, 1], vecmat.get_rot_y_m4(vecmat.deg_to_rad(0))) == [0, 0, -1, 1]

def test_rot_y_x1_90_ccw():
    assert vecmat.vec4_mat4_mul([1, 0, 0, 1], vecmat.get_rot_y_m4(vecmat.deg_to_rad(90))) == pytest.approx([0, 0, -1, 1])

def test_rot_y_x1_90_cw():
    assert vecmat.vec4_mat4_mul([1, 0, 0, 1], vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90))) == pytest.approx([0, 0, 1, 1])

def test_transl():
    assert vecmat.vec4_mat4_mul([0, 0, 0, 1], vecmat.get_transl_m4(1, 2, 3)) == pytest.approx([1, 2, 3, 1])

def test_transl_rot():
    v = [1, 0, 0, 1]
    v = vecmat.vec4_mat4_mul(v, vecmat.get_transl_m4(0, 0, -1))
    assert v == pytest.approx([1, 0, -1, 1])
    v = vecmat.vec4_mat4_mul(v, vecmat.get_rot_z_m4(math.pi / 2))
    assert v == pytest.approx([0, 1, -1, 1])

def test_camera():
    cam = { "pos": [0, 1.5, 3], "rot": [0, vecmat.deg_to_rad(-90), 0], "fov": 90, "ar": 800 / 600 }
    m = vecmat.get_simple_camera_m(cam)
    verts = [(-1.5, -1.5, 0), (1.5, -1.5, 0), (-1.5, 1.5, 0), (1.5, 1.5, 0)]
    view_verts = list(map(lambda model_v: vecmat.vec4_mat4_mul((model_v[0], model_v[1], model_v[2], 1), m), verts))
    assert view_verts[0] == pytest.approx([-3.0, -3.0, 1.5, 1.0])
    assert view_verts[1] == pytest.approx([-3.0, -3.0, -1.5, 1.0])
    assert view_verts[2] == pytest.approx([-3.0, 0.0, 1.5, 1.0])
    assert view_verts[3] == pytest.approx([-3.0, 0.0, -1.5, 1.0])
