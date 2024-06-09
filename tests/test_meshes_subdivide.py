import pytest

from pyrasterize import meshes

def test_subdivide_1_triangle_mesh():
    mesh = meshes.get_test_triangle_mesh()
    new_mesh = meshes.subdivide_triangles(mesh, 1)
    print(new_mesh)
    assert len(new_mesh["tris"]) == 2
    assert len(new_mesh["verts"]) == 4
