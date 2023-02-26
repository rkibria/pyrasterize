TRIANGLE_COLOR = (100, 200, 50)

# Positive y is down
TRIANGLE_FLAT_BOTTOM = [25, 20, 23, 22, 27, 22]

class MockPixelArray:
    def __init__(self, uninitialized_value = (0, 0, 0)):
        self.data = {}
        self.uninitialized_value = uninitialized_value

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key] if key in self.data else self.uninitialized_value

def draw_triangle_2d(array, color_rgb, x1, y1, x2, y2, x3, y3):
    """
    Draw a triangle in 2d coordinates.
    The array object is accessed with a[x,y]=... syntax
    """
    pass

def test_flat_bottom_peak_draws(mocker):
    """The peak point of a flat top triangle is drawn"""
    array = MockPixelArray()
    draw_triangle_2d(array, TRIANGLE_COLOR, *TRIANGLE_FLAT_BOTTOM)
    assert(array[TRIANGLE_FLAT_BOTTOM[0], TRIANGLE_FLAT_BOTTOM[1]] == TRIANGLE_COLOR)
