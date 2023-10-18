import pytest
from pyrasterize.raytracer import *

def test_inverval():
    i1 = make_interval()
    assert interval_size(i1) == float('-inf')

    i1 = make_interval(0, 10)
    i2 = interval_expand(i1, 10)
    assert interval_size(i2) == pytest.approx(20)

def test_aabb():
    a = AABB()
    assert interval_size(a.x) == float('-inf')

    a = AABB(make_interval(1, 2), make_interval(3, 4), make_interval(5, 6))
    assert a.x == make_interval(1, 2)
    assert a.y == make_interval(3, 4)
    assert a.z == make_interval(5, 6)

    a = AABB([1, 2, 3], [4, 5, 6])
    assert a.x == make_interval(1, 4)
    assert a.y == make_interval(2, 5)
    assert a.z == make_interval(3, 6)

    a = AABB([-2, 0, -1], [-1, 1, -2])
    b = AABB([1, 0, -1], [2, 1, -2])
    c = AABB(a, b)
    assert c == AABB([-2, 0, -1], [2, 1, -2])

def test_aabb_hit():
    box = AABB([-10, -10, -1], [20, 20, -2])
    assert box.hit(Ray([0, 0, 0], [-1, -1, -1]), make_interval(0, math.inf))
    assert not box.hit(Ray([0, 0, 0], [1, 1, 1]), make_interval(0, math.inf))

def test_sphere_bbox():
    sphere = Sphere([2, 3, 4], 1, None)
    assert sphere.bounding_box() == AABB(make_interval(1, 3), make_interval(2, 4), make_interval(3, 5))
