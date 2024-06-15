import pytest

from pyrasterize import meshes

def test_subdivide_1_triangle_mesh():
    mesh = meshes.get_test_triangle_mesh()
    new_mesh = meshes.subdivide_triangles(mesh)
    assert len(new_mesh["tris"]) == 2
    assert len(new_mesh["verts"]) == 4
    assert len(new_mesh["verts"]) == len(new_mesh["uv"])

def test_subdivide_1_triangle_mesh_twice():
    mesh = meshes.get_test_triangle_mesh()
    new_mesh = meshes.subdivide_triangles(mesh)
    new_mesh = meshes.subdivide_triangles(new_mesh)
    assert len(new_mesh["tris"]) == 4
    assert len(new_mesh["verts"]) == 8
    assert len(new_mesh["verts"]) == len(new_mesh["uv"])
