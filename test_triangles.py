BLACK_COLOR = (0, 0, 0)
TRIANGLE_COLOR = (100, 200, 50)

class MockPixelArray:
    def __init__(self, uninitialized_value = BLACK_COLOR):
        self.data = {}
        self.uninitialized_value = uninitialized_value

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key] if key in self.data else self.uninitialized_value

    def get_text(self, x0, y0, x1, y1):
        """
        Return rectangular area as text.
        To match expected easily the First line is empty
        and last line ends with newline.
        """
        lines = []
        for y in range(y0, y1 + 1):
            line = []
            for x in range(x0, x1 + 1):
                line.append("x" if self[x, y] == TRIANGLE_COLOR else ".")
            lines.append("".join(line))
        text = "\n".join(lines)
        return "\n" + text + "\n"


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
    pass

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


# Positive y is down. peak, lower right, lower left
TRIANGLE_FLAT_BOTTOM = [25, 20, 27, 22, 23, 22]

EXPECTED_FLAT_BOTTOM_TRIANGLE= """
.......
...x...
..xxx..
.xxxxx.
.......
"""

def test_draw_flat_bottom_triangle():
    """A flat bottom triangle is drawn"""
    array = MockPixelArray()
    draw_triangle_2d(array, TRIANGLE_COLOR, *TRIANGLE_FLAT_BOTTOM)
    actual = array.get_text(22, 19, 28, 23)
    assert(EXPECTED_FLAT_BOTTOM_TRIANGLE == actual)


TRIANGLE_FLAT_TOP = [23, 20, 27, 20, 25, 22]

EXPECTED_FLAT_TOP_TRIANGLE= """
.......
.xxxxx.
..xxx..
...x...
.......
"""

def test_draw_flat_top_triangle():
    """A flat top triangle is drawn"""
    array = MockPixelArray()
    draw_triangle_2d(array, TRIANGLE_COLOR, *TRIANGLE_FLAT_TOP)
    actual = array.get_text(22, 19, 28, 23)
    assert(EXPECTED_FLAT_TOP_TRIANGLE == actual)


if __name__ == '__main__':
    test_draw_flat_bottom_triangle()
