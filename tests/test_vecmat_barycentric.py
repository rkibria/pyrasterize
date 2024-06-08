import pytest

from pyrasterize import vecmat

def test_bary_2d():
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
