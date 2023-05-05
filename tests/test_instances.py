from pyrasterize import rasterizer
from pyrasterize import meshes

def test_instance_with_null_model():
    """Instance with no model is allowed"""
    instance = rasterizer.get_model_instance(None)
    assert("enabled" in instance)
    assert("model" in instance)
    assert("preproc_m4" in instance)
    assert("xform_m4" in instance)
    assert("children" in instance)
    model = instance["model"]
    assert(model is None)

def test_instance_contains_expected_entries():
    """Sanity check instance"""
    instance = rasterizer.get_model_instance(meshes.get_test_triangle_mesh())
    assert("enabled" in instance)
    assert("model" in instance)
    assert("preproc_m4" in instance)
    assert("xform_m4" in instance)
    assert("children" in instance)
    model = instance["model"]
    assert("verts" in model)
    assert("tris" in model)
    assert("colors" in model)

def test_instance_creates_normals():
    """Making an instance adds triangle and vertex normals to the model if they don't exist"""
    instance = rasterizer.get_model_instance(meshes.get_test_triangle_mesh())
    model = instance["model"]
    assert("normals" in model)
    normals = model["normals"]
    assert(len(normals) == 1)
    assert(normals[0] == [0, 0, 1, 0])
    assert("vert_normals" in model)
    vert_normals = model["vert_normals"]
    assert(len(vert_normals) == 3)
    assert(vert_normals[0] == [0,0,1,0])
    assert(vert_normals[1] == [0,0,1,0])
    assert(vert_normals[2] == [0,0,1,0])

if __name__ == "__main__":
    pass
