from pyrasterize import vecmat

def ray_sphere_intersect(r_orig, r_dir, sph_orig, sph_r, t_min=0.001, t_max=10**6):
    """Return t if ray intersects sphere or None"""
    oc = [r_orig[0] - sph_orig[0], r_orig[1] - sph_orig[1], r_orig[2] - sph_orig[2]]
    a = r_dir[0] * r_dir[0] + r_dir[1] * r_dir[1] + r_dir[2] * r_dir[2]
    b = oc[0] * r_dir[0] + oc[1] * r_dir[1] + oc[2] * r_dir[2]
    c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - sph_r * sph_r
    discriminant = b * b - a * c
    if discriminant > 0:
        sqrt_discriminant = discriminant ** 0.5
        temp_1 = (-b - sqrt_discriminant) / a
        if temp_1 < t_max and temp_1 > t_min:
            return temp_1
        temp_2 = (-b + sqrt_discriminant) / a
        if temp_2 < t_max and temp_2 > t_min:
            return temp_2
    return None

def mouse_pos_to_ray(pos, scr_size):
    """Get ray vec3 into scene from mouse position"""
    ndc_x = 2 * pos[0] / scr_size[0] - 1
    ndc_y = 1 - (2 * pos[1]) / scr_size[1]
    return vecmat.norm_vec3([ndc_x, ndc_y, -1])



def test_mouse_pos_to_ray_origin():
    """Ray in screen center goes straight forward"""
    ray = mouse_pos_to_ray([400, 300], [800, 600])
    assert ray[0:2] == [0, 0]

def test_ray_intersects_sphere():
    """Test: intersect"""
    r_orig = [0, 0, 0]
    r_dir = mouse_pos_to_ray([400, 300], [800, 600])
    sph_orig = [0, 0, -10]
    sph_r = 1
    t = ray_sphere_intersect(r_orig, r_dir, sph_orig, sph_r)
    assert t == 9

def test_ray_doesnt_intersect_sphere():
    """Test: intersect"""
    r_orig = [0, 0, 0]
    r_dir = mouse_pos_to_ray([400, 300], [800, 600])
    sph_orig = [0, 0, 10]
    sph_r = 1
    t = ray_sphere_intersect(r_orig, r_dir, sph_orig, sph_r)
    assert t is None
