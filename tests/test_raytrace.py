import pytest
from pyrasterize.raytracer import *

def test_inverval():
    i1 = Interval()
    assert i1.size() == float('-inf')

    i1 = Interval(0, 10)
    i2 = i1.expand(10)
    assert i2.size() == pytest.approx(20)

def test_aabb():
    a = AABB()
    assert a.x.size() == float('-inf')

    a = AABB(Interval(1, 2), Interval(3, 4), Interval(5, 6))
    assert a.x == Interval(1, 2)
    assert a.y == Interval(3, 4)
    assert a.z == Interval(5, 6)

    a = AABB([1, 2, 3], [4, 5, 6])
    assert a.x == Interval(1, 4)
    assert a.y == Interval(2, 5)
    assert a.z == Interval(3, 6)

    a = AABB([-2, 0, -1], [-1, 1, -2])
    b = AABB([1, 0, -1], [2, 1, -2])
    c = AABB(a, b)
    assert c == AABB([-2, 0, -1], [2, 1, -2])

def test_aabb_hit():
    box = AABB([-10, -10, -1], [20, 20, -2])
    assert box.hit(Ray([0, 0, 0], [-1, -1, -1]), Interval(0, math.inf))
    assert not box.hit(Ray([0, 0, 0], [1, 1, 1]), Interval(0, math.inf))

def test_sphere_bbox():
    sphere = Sphere([2, 3, 4], 1, None)
    assert sphere.bounding_box() == AABB(Interval(1, 3), Interval(2, 4), Interval(3, 5))
