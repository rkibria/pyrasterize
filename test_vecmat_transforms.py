import pytest

from pyrasterize import vecmat

def test_rot_y_no_rot():
    assert vecmat.vec4_mat4_mul([0, 0, -1, 1], vecmat.get_rot_y_m4(vecmat.deg_to_rad(0))) == [0, 0, -1, 1]

def test_rot_y_x1_90_ccw():
    assert vecmat.vec4_mat4_mul([1, 0, 0, 1], vecmat.get_rot_y_m4(vecmat.deg_to_rad(90))) == pytest.approx([0, 0, -1, 1])

def test_rot_y_x1_90_cw():
    assert vecmat.vec4_mat4_mul([1, 0, 0, 1], vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90))) == pytest.approx([0, 0, 1, 1])
