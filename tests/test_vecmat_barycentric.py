import pytest

from pyrasterize import vecmat

def test_bary_2d_():
    v_a = (0, 0)
    v_b = (1, 0)
    v_c = (0, 1)
    bt = vecmat.Barycentric2dTriangle(v_a, v_b, v_c)

    assert bt.get_uvw(*v_a) == pytest.approx((1, 0, 0))
    assert bt.get_uvw(*v_b) == pytest.approx((0, 1, 0))
    assert bt.get_uvw(*v_c) == pytest.approx((0, 0, 1))

    assert bt.get_uvw(0.5, 0) == pytest.approx((0.5, 0.5, 0))
    assert bt.get_uvw(0, 0.5) == pytest.approx((0.5, 0, 0.5))
    assert bt.get_uvw(0.5, 0.5) == pytest.approx((0, 0.5, 0.5))

def test_bary_2d():
    v_a = (0, 0)
    v_b = (1, 0)
    v_c = (0, 1)
    
    assert vecmat.get_barycentric_vec2(v_a, v_b, v_c, v_a) == pytest.approx((1, 0, 0))
    assert vecmat.get_barycentric_vec2(v_a, v_b, v_c, v_b) == pytest.approx((0, 1, 0))
    assert vecmat.get_barycentric_vec2(v_a, v_b, v_c, v_c) == pytest.approx((0, 0, 1))

    assert vecmat.get_barycentric_vec2(v_a, v_b, v_c, (0.5, 0)) == pytest.approx((0.5, 0.5, 0))
    assert vecmat.get_barycentric_vec2(v_a, v_b, v_c, (0, 0.5)) == pytest.approx((0.5, 0, 0.5))
    assert vecmat.get_barycentric_vec2(v_a, v_b, v_c, (0.5, 0.5)) == pytest.approx((0, 0.5, 0.5))

def test_bary_3d_xy():
    v_a = (0, 0, 1)
    v_b = (1, 0, 1)
    v_c = (0, 1, 1)
    
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, v_a) == pytest.approx((1, 0, 0))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, v_b) == pytest.approx((0, 1, 0))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, v_c) == pytest.approx((0, 0, 1))

    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0.5, 0, 1)) == pytest.approx((0.5, 0.5, 0))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0, 0.5, 1)) == pytest.approx((0.5, 0, 0.5))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0.5, 0.5, 1)) == pytest.approx((0, 0.5, 0.5))

    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0.5, 0, -1)) == pytest.approx((0.5, 0.5, 0))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0, 0.5, 0)) == pytest.approx((0.5, 0, 0.5))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0.5, 0.5, 1)) == pytest.approx((0, 0.5, 0.5))

def test_bary_3d_xyz():
    v_a = (1, 0, 0)
    v_b = (0, 1, 0)
    v_c = (0, 0, 1)
    
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, v_a) == pytest.approx((1, 0, 0))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, v_b) == pytest.approx((0, 1, 0))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, v_c) == pytest.approx((0, 0, 1))

    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0.5, 0.5, 0)) == pytest.approx((0.5, 0.5, 0))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0, 0.5, 0.5)) == pytest.approx((0, 0.5, 0.5))
    assert vecmat.get_barycentric_vec3(v_a, v_b, v_c, (0.5, 0, 0.5)) == pytest.approx((0.5, 0, 0.5))
