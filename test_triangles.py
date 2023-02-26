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

def draw_triangle_2d(array, color, x0, y0, x1, y1, x2, y2):
    """
    Draw a triangle in 2d coordinates.
    The array object is accessed with a[x,y]=... syntax
    """
    # P2 is LEFT of peak, P1 is RIGHT of peak
    dxy_left = (x2 - x0) / (y2 - y0)
    dxy_right = (x1 - x0) / (y1 - y0)
    xs = x0
    xe = x0
    for y in range(int(y0), int(y1) + 1):
        # Draw line
        for x in range(int(xs), int(xe) + 1):
            array[x, y] = color
        # Go down scanline
        xs += dxy_left
        xe += dxy_right

# Positive y is down. peak, lower right, lower left
TRIANGLE_FLAT_BOTTOM = [25, 20, 27, 22, 23, 22]

EXPECTED_FLAT_TOP_TRIANGLE= """
.......
...x...
..xxx..
.xxxxx.
.......
"""

def test_draw_flat_bottom_triangle():
    """A flat top triangle is drawn"""
    array = MockPixelArray()
    draw_triangle_2d(array, TRIANGLE_COLOR, *TRIANGLE_FLAT_BOTTOM)
    actual = array.get_text(22, 19, 28, 23)
    assert(EXPECTED_FLAT_TOP_TRIANGLE == actual)

if __name__ == '__main__':
    test_draw_flat_bottom_triangle()
