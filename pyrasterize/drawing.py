#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Drawing primitives
"""

def _draw_flat_top_triangle_2d(array, color, x1, y1, x2, y2, x3, y3):
    """
    Draw flat top triangle
    """
    if x2 < x1:
        x2, x1 = x1, x2
    
    height = y3 - y1

    dx_left  = (x3 - x1) / height
    dx_right = (x3 - x2) / height

    xs = x1
    xe = x2 + 0.5

    for y in range(int(y1), int(y3 + 1)):
        for x in range(int(xs), int(xe) + 1):
            array[x, y] = color
        xs += dx_left
        xe += dx_right


def _draw_flat_bottom_triangle_2d(array, color, x1, y1, x2, y2, x3, y3):
    """
    Draw flat bottom triangle
    """
    if x3 < x2:
        x3, x2 = x2, x3
    
    height = y3 - y1

    dx_left  = (x2 - x1) / height
    dx_right = (x3 - x1) / height

    xs = x1
    xe = x1

    for y in range(int(y1), int(y3 + 1)):
        for x in range(int(xs), int(xe) + 1):
            array[x, y] = color
        xs += dx_left
        xe += dx_right


def draw_triangle_2d(array, color, x1, y1, x2, y2, x3, y3):
    """
    Draw a triangle in 2d coordinates by decomposing it into separate flat-top/flat-bottom triangles.
    Writes output pixels with array[x,y]=color.
    Positive y is down.
    """
    if (x1 == x2 and x2 == x3) or (y1 == y2 and y2 == y3):
        return

    if y2 < y1:
        x2, x1 = x1, x2
        y2, y1 = y1, y2

    if y3 < y1:
        x3, x1 = x1, x3
        y3, y1 = y1, y3

    if y3 < y2:
        x3, x2 = x2, x3
        y3, y2 = y2, y3

    if y1 == y2:
        _draw_flat_top_triangle_2d(array, color, x1, y1, x2, y2, x3, y3)
    else:
        if y2 == y3:
            _draw_flat_bottom_triangle_2d(array, color, x1, y1, x2, y2, x3, y3)
        else:
            new_x = x1 + int(0.5 + (y2 - y1) * (x3 - x1) / (y3 - y1))
            _draw_flat_bottom_triangle_2d(array, color, x1, y1, new_x, y2, x2, y2)
            _draw_flat_top_triangle_2d(array, color, x2, y2, new_x, y2, x3, y3)
