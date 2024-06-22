import pytest

from pyrasterize import vecmat
from pyrasterize import rasterizer

def test_persp_m():
    fov = 90
    d = vecmat.get_view_plane_from_fov(fov)
    persp_m = vecmat.get_persp_m4(d, 1)
    assert rasterizer.project_to_clip_space([1, 1, -1, 1], persp_m)[:2] == pytest.approx([1, 1])
    assert rasterizer.project_to_clip_space([2, 2, -2, 1], persp_m)[:2] == pytest.approx([1, 1])
    assert rasterizer.project_to_clip_space([-1, -1, -1, 1], persp_m)[:2] == pytest.approx([-1, -1])
    assert rasterizer.project_to_clip_space([-2, -2, -2, 1], persp_m)[:2] == pytest.approx([-1, -1])
