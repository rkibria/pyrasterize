from pyrasterize import vecmat

def test_mouse_pos_to_ray_origin():
    """Ray in screen center goes straight forward"""
    ray = vecmat.mouse_pos_to_ray([400, 300], [800, 600])
    assert ray[0:2] == [0, 0]

def test_ray_intersects_sphere():
    """Test: intersect"""
    r_orig = [0, 0, 0]
    r_dir = vecmat.mouse_pos_to_ray([400, 300], [800, 600])
    sph_orig = [0, 0, -10]
    sph_r = 1
    t = vecmat.ray_sphere_intersect(r_orig, r_dir, sph_orig, sph_r)
    assert t == 9

def test_ray_intersects_sphere_glance():
    """Test: intersect"""
    r_orig = [0.99999, 0, 0]
    r_dir = vecmat.mouse_pos_to_ray([400, 300], [800, 600])
    sph_orig = [0, 0, -10]
    sph_r = 1
    t = vecmat.ray_sphere_intersect(r_orig, r_dir, sph_orig, sph_r)
    assert abs(t - 10) < 0.1

def test_ray_doesnt_intersect_sphere():
    """Test: intersect"""
    r_orig = [0, 0, 0]
    r_dir = vecmat.mouse_pos_to_ray([400, 300], [800, 600])
    sph_orig = [0, 0, 10]
    sph_r = 1
    t = vecmat.ray_sphere_intersect(r_orig, r_dir, sph_orig, sph_r)
    assert t is None
