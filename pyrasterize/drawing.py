#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Drawing primitives
"""

def bresenham(x0, y0, x1, y1):
    """
    From https://github.com/encukou/bresenham
    Yield integer coordinates on the line from (x0, y0) to (x1, y1).

    Input coordinates should be integers.

    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def triangle(x1, y1, x2, y2, x3, y3):
    """
    Generator for points of a triangle in 2d coordinates.
    Positive y is down.
    """
    def flat_top_triangle(x1, y1, x2, y2, x3, y3):
        if x2 < x1:
            x2, x1 = x1, x2
        # x1,y1 is upper left point
        # x2,y2 is upper right point (and y1 == y2)
        # x3,y3 is bottom point
        lb = bresenham(x1, y1, x3, y3)
        rb = bresenham(x2, y2, x3, y3)
        cur_y = None
        while True:
            lx,ly = next(lb, (None, None))
            if lx is None:
                break
            if cur_y is None:
                cur_y = ly
            else:
                if ly == cur_y:
                    continue
                else:
                    cur_y = ly
            while True:
                rx,ry = next(rb, (None, None))
                if ry == cur_y:
                    for x in range(lx, rx+1):
                        yield x,cur_y
                    break

    def flat_bottom_triangle(x1, y1, x2, y2, x3, y3):
        if x3 < x2:
            x3, x2 = x2, x3
        # x1,y1 is top point
        # x2,y2 is lower left point
        # x3,y3 is lower right point (and y2 == y3)
        lb = bresenham(x1, y1, x2, y2)
        rb = bresenham(x1, y1, x3, y3)
        cur_y = None
        while True:
            lx,ly = next(lb, (None, None))
            if lx is None:
                break
            if cur_y is None:
                cur_y = ly
            else:
                if ly == cur_y:
                    continue
                else:
                    cur_y = ly
            while True:
                rx,ry = next(rb, (None, None))
                if ry == cur_y:
                    for x in range(lx, rx+1):
                        yield x,cur_y
                    break

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
        yield from flat_top_triangle(x1, y1, x2, y2, x3, y3)
    else:
        if y2 == y3:
            yield from flat_bottom_triangle(x1, y1, x2, y2, x3, y3)
        else:
            new_x = x1 + int((y2 - y1) * (x3 - x1) / (y3 - y1))
            yield from flat_bottom_triangle(x1, y1, new_x, y2, x2, y2)
            yield from flat_top_triangle(x2, y2, new_x, y2, x3, y3)
