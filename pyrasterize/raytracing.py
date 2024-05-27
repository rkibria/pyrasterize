#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ray tracing functions
"""

from __future__ import annotations

import math

def make_interval(a = float('inf'), b = float('-inf')) -> list:
    """0 = min, 1 = max"""
    return [a, b]

def make_interval_from_intervals(a : list, b : list) -> list:
    return [min(a[0], b[0]), max(a[1], b[1])]

def interval_size(iv : list) -> float:
    return iv[1] - iv[0]

def interval_expand(iv : list, delta : float) -> list:
    padding = delta / 2
    return [iv[0] - padding, iv[1] + padding]

def interval_contains(iv : list, x : float) -> bool:
    return iv[0] <= x and x <= iv[1]

def interval_surrounds(iv : list, x : float) -> bool:
    return iv[0] < x and x < iv[1]

def interval_clamp(iv : list, x : float) -> float:
    if x < iv[0]:
        return iv[0]
    elif x > iv[1]:
        return iv[1]
    else:
        return x

INTERVAL_EMPTY = [float('inf'), float('-inf')]
INTERVAL_UNIVERSE = [float('-inf'), float('inf')]

class AABB:
    """The default AABB is empty, since intervals are empty by default"""

    def __init__(self, ix = None, iy = None, iz = None) -> None:
        if ix is None:
            self.x = make_interval()
            self.y = make_interval()
            self.z = make_interval()
        elif isinstance(ix, list): # intervals ix, iy, iz
            if len(ix) == 2:
                self.x = ix[:]
                self.y = iy[:]
                self.z = iz[:]
            else: # ix = vec3, iy = vec3
                # Treat the two points a and b as extrema for the bounding box, so we don't require a
                # particular minimum/maximum coordinate order.
                a = ix
                b = iy
                self.x = make_interval(min(a[0],b[0]), max(a[0],b[0]))
                self.y = make_interval(min(a[1],b[1]), max(a[1],b[1]))
                self.z = make_interval(min(a[2],b[2]), max(a[2],b[2]))
        elif isinstance(ix, AABB):
            self.x = make_interval_from_intervals(ix.x, iy.x)
            self.y = make_interval_from_intervals(ix.y, iy.y)
            self.z = make_interval_from_intervals(ix.z, iy.z)
        self.pad_to_minimums()

    def __str__(self) -> str:
        return f"AABB({self.x}, {self.y}, {self.z})"

    def __eq__(self, __value: object) -> bool:
        return self.x == __value.x and self.y == __value.y and self.z == __value.z

    def axis(self, n : int):
        if n == 1:
            return self.y
        if n == 2:
            return self.z
        return self.x

    def pad_to_minimums(self):
        """Adjust the AABB so that no side is narrower than some delta, padding if necessary"""
        delta = 0.0001
        if interval_size(self.x) < delta:
            self.x = interval_expand(self.x, delta)
        if interval_size(self.y) < delta:
            self.y = interval_expand(self.y, delta)
        if interval_size(self.z) < delta:
            self.z = interval_expand(self.z, delta)

    def hit(self, r : Ray, ray_t : list) -> bool:
        for a,axis in zip([0, 1, 2], [self.x, self.y, self.z]):
            r_dir = r.direction[a]

            if r_dir:
                invD = 1 / r_dir
            else:
                invD = math.inf

            orig = r.origin[a]

            t0 = (axis[0] - orig) * invD
            t1 = (axis[1] - orig) * invD

            if invD < 0:
                t1, t0 = t0, t1

            if t0 > ray_t[0]:
                ray_t[0] = t0
            if t1 < ray_t[1]:
                ray_t[1] = t1
            
            if ray_t[1] <= ray_t[0]:
                return False
        return True

    @staticmethod
    def empty() -> AABB:
        return AABB(INTERVAL_EMPTY, INTERVAL_EMPTY, INTERVAL_EMPTY)

    @staticmethod
    def universe() -> AABB:
        return AABB(INTERVAL_UNIVERSE, INTERVAL_UNIVERSE, INTERVAL_UNIVERSE)

class Ray:
    def __init__(self, origin : list, direction : list) -> None:
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return [self.origin[0] + t * self.direction[0],
                self.origin[1] + t * self.direction[1],
                self.origin[2] + t * self.direction[2]]
