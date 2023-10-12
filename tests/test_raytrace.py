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
