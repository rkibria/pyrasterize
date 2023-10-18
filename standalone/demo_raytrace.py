"""
Ray tracing demo with BVH structure

Based on https://raytracing.github.io/books/RayTracingInOneWeekend.html
"""

from __future__ import annotations

import random
from timeit import default_timer as timer
import copy
import math
import functools

import pygame
import pygame.gfxdraw

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 320, 240
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)
RGB_BLACK = (0, 0, 0)

# MATHS CODE

def deg_to_rad(degrees):
    """Return degrees converted to radians"""
    return degrees * (math.pi / 180)

def dot_product_vec3(a, b):
    """Return dot product of vec3"""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def mag_vec3(v_3):
    """Return magnitude of vec3"""
    return (v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2]) ** 0.5

def mag_sq_vec3(v_3):
    """Return squared magnitude of vec3"""
    return (v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2])

def random_in_unit_sphere_vec3():
    while True:
        p = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        if mag_sq_vec3(p) >= 1:
            continue
        return p

def norm_vec3(v_3):
    """Return normalized vec3"""
    mag = v_3[0]*v_3[0] + v_3[1]*v_3[1] + v_3[2]*v_3[2]
    if mag == 0:
        return [0, 0, 0]
    mag = mag ** -0.5 # 1.0 / math.sqrt(mag)
    return [v_3[0] * mag, v_3[1] * mag, v_3[2] * mag]

def random_unit_vector_vec3():
    return norm_vec3(random_in_unit_sphere_vec3())

def near_zero_vec3(v : list):
    s = 1e-8
    return abs(v[0]) < s and abs(v[1]) < s and abs(v[2]) < s

def reflect_vec3(v : list, n : list):
    dot_vn = dot_product_vec3(v, n)
    return [v[i] - 2 * dot_vn * n[i] for i in range(3)]

def refract_vec3(uv : list, n : list, etai_over_etat : float):
    minus_uv = [-uv[i] for i in range(3)]
    cos_theta = min(dot_product_vec3(minus_uv, n), 1.0)
    r_out_perp = [etai_over_etat * (uv[i] + cos_theta * n[i]) for i in range(3)]
    r_out_perp_length_squared = mag_sq_vec3(r_out_perp)
    k = -((abs(1 - r_out_perp_length_squared)) ** 0.5)
    r_out_parallel = [n[i] * k for i in range(3)]
    return [r_out_perp[i] + r_out_parallel[i] for i in range(3)]

def ray_sphere_intersect(r_orig3, r_dir3, sph_orig3, sph_r, t_min=0.001, t_max=10**6):
    """Return ray direction multi t if ray intersects sphere or None"""
    oc = [r_orig3[0] - sph_orig3[0], r_orig3[1] - sph_orig3[1], r_orig3[2] - sph_orig3[2]]
    a = r_dir3[0] * r_dir3[0] + r_dir3[1] * r_dir3[1] + r_dir3[2] * r_dir3[2]
    b = oc[0] * r_dir3[0] + oc[1] * r_dir3[1] + oc[2] * r_dir3[2]
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

def cross_vec3(a, b):
    """Return vec3 result of cross product of 2 vec3's"""
    return [a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]]

def random_in_unit_disk_vec3():
    while True:
        p = [random.uniform(-1, 1), random.uniform(-1, 1), 0]
        if mag_sq_vec3(p) >= 1:
            continue
        return p

# RAY TRACER CODE

# Interval is faster as simple lists instead of a class because we create so many of them

def make_interval(a = float('inf'), b = float('-inf')) -> list:
    """0 = min, 1 = max"""
    return [a, b]

def make_interval_from_intervals(a : list, b : list) -> list:
    return [min(a[0], b[0]), max(a[1], b[1])]

def interval_size(iv : list) -> float:
    return iv[1] - iv[0]

def interval_expand(iv : list, delta : float) -> list:
    padding = delta / 2
    return [iv[0] - padding, iv[1] + padding]

def interval_contains(iv : list, x : float) -> bool:
    return iv[0] <= x and x <= iv[1]

def interval_surrounds(iv : list, x : float) -> bool:
    return iv[0] < x and x < iv[1]

def interval_clamp(iv : list, x : float) -> float:
    if x < iv[0]:
        return iv[0]
    elif x > iv[1]:
        return iv[1]
    else:
        return x

INTERVAL_EMPTY = [float('inf'), float('-inf')]
INTERVAL_UNIVERSE = [float('-inf'), float('inf')]

class AABB:
    """The default AABB is empty, since intervals are empty by default"""

    def __init__(self, ix = None, iy = None, iz = None) -> None:
        if ix is None:
            self.x = make_interval()
            self.y = make_interval()
            self.z = make_interval()
        elif isinstance(ix, list): # intervals ix, iy, iz
            if len(ix) == 2:
                self.x = ix[:]
                self.y = iy[:]
                self.z = iz[:]
            else: # ix = vec3, iy = vec3
                # Treat the two points a and b as extrema for the bounding box, so we don't require a
                # particular minimum/maximum coordinate order.
                a = ix
                b = iy
                self.x = make_interval(min(a[0],b[0]), max(a[0],b[0]))
                self.y = make_interval(min(a[1],b[1]), max(a[1],b[1]))
                self.z = make_interval(min(a[2],b[2]), max(a[2],b[2]))
        elif isinstance(ix, AABB):
            self.x = make_interval_from_intervals(ix.x, iy.x)
            self.y = make_interval_from_intervals(ix.y, iy.y)
            self.z = make_interval_from_intervals(ix.z, iy.z)
        self.pad_to_minimums()

    def __str__(self) -> str:
        return f"AABB({self.x}, {self.y}, {self.z})"

    def __eq__(self, __value: object) -> bool:
        return self.x == __value.x and self.y == __value.y and self.z == __value.z

    def axis(self, n : int):
        if n == 1:
            return self.y
        if n == 2:
            return self.z
        return self.x

    def pad_to_minimums(self):
        """Adjust the AABB so that no side is narrower than some delta, padding if necessary"""
        delta = 0.0001
        if interval_size(self.x) < delta:
            self.x = interval_expand(self.x, delta)
        if interval_size(self.y) < delta:
            self.y = interval_expand(self.y, delta)
        if interval_size(self.z) < delta:
            self.z = interval_expand(self.z, delta)

    def hit(self, r : Ray, ray_t : list) -> bool:
        for a,axis in zip([0, 1, 2], [self.x, self.y, self.z]):
            r_dir = r.direction[a]

            if r_dir:
                invD = 1 / r_dir
            else:
                invD = math.inf

            orig = r.origin[a]

            t0 = (axis[0] - orig) * invD
            t1 = (axis[1] - orig) * invD

            if invD < 0:
                t1, t0 = t0, t1

            if t0 > ray_t[0]:
                ray_t[0] = t0
            if t1 < ray_t[1]:
                ray_t[1] = t1
            
            if ray_t[1] <= ray_t[0]:
                return False
        return True

    def longest_axis(self) -> int:
        """Returns the index of the longest axis of the bounding box"""
        if interval_size(self.x) > interval_size(self.y):
            return 0 if interval_size(self.x) > interval_size(self.z) else 2
        else:
            return 1 if interval_size(self.y) > interval_size(self.z) else 2

    @staticmethod
    def empty() -> AABB:
        return AABB(INTERVAL_EMPTY, INTERVAL_EMPTY, INTERVAL_EMPTY)

    @staticmethod
    def universe() -> AABB:
        return AABB(INTERVAL_UNIVERSE, INTERVAL_UNIVERSE, INTERVAL_UNIVERSE)

class Ray:
    def __init__(self, origin : list, direction : list) -> None:
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return [self.origin[0] + t * self.direction[0],
                self.origin[1] + t * self.direction[1],
                self.origin[2] + t * self.direction[2]]

class HitRecord:
    def __init__(self) -> None:
        self.hit_point = [0, 0, 0]
        self.normal = [0, 0, 0]
        self.t = 0.0
        self.front_face = False
        self.material = None

    def copy(self, rec : HitRecord):
        self.hit_point = copy.copy(rec.hit_point)
        self.normal = copy.copy(rec.normal)
        self.t = rec.t
        self.front_face = rec.front_face
        self.material = rec.material

    def set_face_normal(self, r : Ray, outward_normal):
        self.front_face = dot_product_vec3(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else [-outward_normal[i] for i in range(3)]

class Hittable:
    def __init__(self) -> None:
        pass

    def hit(self, r : Ray, ray_t : list, rec: HitRecord) -> bool:
        return False

    def bounding_box(self) -> AABB:
        return None

class HittableList:
    def __init__(self) -> None:
        self.objects : list(Hittable) = []
        self.bbox = AABB()

    def bounding_box(self) -> AABB:
        return self.bbox

    def add(self, object : Hittable) -> None:
        self.objects.append(object)
        self.bbox = AABB(self.bbox, object.bounding_box())

    def hit(self, r : Ray, ray_t : list, rec: HitRecord) -> bool:
        temp_rec = HitRecord()
        hit_anything = False
        closest_so_far = ray_t[1]

        for object in self.objects:
            if object.hit(r, make_interval(ray_t[0], closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.copy(temp_rec)

        return hit_anything

class BvhNode(Hittable):
    @staticmethod
    def box_compare(a : Hittable, b : Hittable, axis_index : int) -> bool:
        return a.bounding_box().axis(axis_index)[0] < b.bounding_box().axis(axis_index)[0]

    @staticmethod
    def box_x_compare(a : Hittable, b : Hittable) -> bool:
        return BvhNode.box_compare(a, b, 0)

    @staticmethod
    def box_y_compare(a : Hittable, b : Hittable) -> bool:
        return BvhNode.box_compare(a, b, 1)

    @staticmethod
    def box_z_compare(a : Hittable, b : Hittable) -> bool:
        return BvhNode.box_compare(a, b, 2)

    def __str__(self) -> str:
        s = f"BvhNode({self.bbox})["
        s += str(self.left) if self.left is not None else "-"
        s += ","
        s += str(self.right) if self.right is not None else "-"
        s += "]"
        return s

    def __init__(self, src_objects_ : None, start : int = 0, end : int = 0) -> None:
        self.left = None
        self.right = None
        self.bbox = AABB.empty()

        if src_objects_ is None:
            raise RuntimeError("Need argument")
        elif isinstance(src_objects_, HittableList):
            src_objects = src_objects_.objects
            start = 0
            end = len(src_objects)
        else:
            src_objects = src_objects_

        objects: list(Hittable) = copy.copy(src_objects)

        for object_index in range(start, end):
            self.bbox = AABB(self.bbox, objects[object_index].bounding_box())

        axis = self.bbox.longest_axis()

        comparator = None
        if axis == 0:
            comparator = BvhNode.box_x_compare
        elif axis == 1:
            comparator = BvhNode.box_y_compare
        else:
            comparator = BvhNode.box_z_compare

        object_span = end - start

        if object_span == 1:
            self.left = self.right = objects[start]
        elif object_span == 2:
            if comparator(objects[start], objects[start+1]):
                self.left = objects[start]
                self.right = objects[start + 1]
            else:
                self.left = objects[start + 1]
                self.right = objects[start]
        else:
            objects[start:end] = sorted(objects[start:end], key=functools.cmp_to_key(comparator))
            mid = start + object_span // 2
            self.left = BvhNode(objects, start, mid)
            self.right = BvhNode(objects, mid, end)

    def bounding_box(self) -> AABB:
        return self.bbox

    def hit(self, r : Ray, ray_t : list, rec: HitRecord) -> bool:
        if not self.bbox.hit(r, ray_t[:]):
            return False
        
        hit_left = False
        if self.left:
            hit_left = self.left.hit(r, ray_t[:], rec)
        
        hit_right = False
        if self.right:
            hit_right = self.right.hit(r, make_interval(ray_t[0], rec.t if hit_left else ray_t[1]), rec)
        return hit_left or hit_right

class Material:
    def __init__(self) -> None:
        pass

    # Return (is_scattered : bool, attenuation : vec3, scattered : Ray)
    def scatter(self, r_in : Ray, rec : HitRecord):
        pass

class Lambertian(Material):
    def __init__(self, albedo : list) -> None:
        super().__init__()
        self.albedo = albedo

    # Return (is_scattered : bool, attenuation : vec3, scattered : Ray)
    def scatter(self, r_in : Ray, rec : HitRecord):
        rand_v = random_unit_vector_vec3()
        scatter_direction = [rec.normal[i] + rand_v[i] for i in range(3)]
        # Catch degenerate scatter direction
        if near_zero_vec3(scatter_direction):
            scatter_direction = rec.normal

        scattered = Ray(rec.hit_point, scatter_direction)
        return (True, self.albedo, scattered)

class Metal(Material):
    def __init__(self, albedo : list, fuzz : float) -> None:
        super().__init__()
        self.albedo = albedo
        self.fuzz = fuzz if fuzz < 1 else 1

    # Return (is_scattered : bool, attenuation : vec3, scattered : Ray)
    def scatter(self, r_in : Ray, rec : HitRecord):
        norm_r_in_dir = norm_vec3(r_in.direction)
        reflected = reflect_vec3(norm_r_in_dir, rec.normal)
        rand_v = random_in_unit_sphere_vec3()
        reflected = [reflected[i] + self.fuzz * rand_v[i] for i in range(3)]
        scattered = Ray(rec.hit_point, reflected)
        return (dot_product_vec3(scattered.direction, rec.normal) > 0, self.albedo, scattered)

class Dielectric(Material):
    def __init__(self, index_of_refraction : float) -> None:
        super().__init__()
        self.index_of_refraction = index_of_refraction

    @staticmethod
    def reflectance(cosine : float, ref_idx : float) -> float:
        # Use Schlick's approximation for reflectance
        r0 = (1-ref_idx) / (1+ref_idx)
        r0 *= r0
        return r0 + (1-r0) * ((1 - cosine) ** 5)

    # Return (is_scattered : bool, attenuation : vec3, scattered : Ray)
    def scatter(self, r_in : Ray, rec : HitRecord):
        refraction_ratio = 1.0 / self.index_of_refraction if rec.front_face else self.index_of_refraction
        unit_direction = norm_vec3(r_in.direction)

        minus_unit_direction = [-unit_direction[i] for i in range(3)]
        cos_theta = min(dot_product_vec3(minus_unit_direction, rec.normal), 1.0)
        sin_theta = (1.0 - cos_theta ** 2) ** 0.5

        cannot_refract = refraction_ratio * sin_theta > 1.0

        if cannot_refract:
            direction = reflect_vec3(unit_direction, rec.normal)
        else:
            direction = refract_vec3(unit_direction, rec.normal, refraction_ratio)

        scattered = Ray(rec.hit_point, direction)
        return (True, [1.0, 1.0, 1.0], scattered)

class Sphere(Hittable):
    def __init__(self, center : list, radius : float, material : Material) -> None:
        super().__init__()
        self.center = center
        self.radius = radius
        self.material = material
        self.bbox = AABB([center[i] - radius for i in range(3)], [center[i] + radius for i in range(3)])

    def __str__(self) -> str:
        return f"Sphere({self.center},{self.radius},{self.bbox})"

    def bounding_box(self) -> AABB:
        return self.bbox

    def hit(self, r : Ray, ray_t : list, rec: HitRecord) -> bool:
        t = ray_sphere_intersect(r.origin, r.direction, self.center, self.radius, ray_t[0], ray_t[1])
        if t:
            hit_point = r.at(t)
            outward_normal = [(hit_point[i] - self.center[i]) / self.radius for i in range(3)]
            rec.hit_point = hit_point
            rec.set_face_normal(r, outward_normal)
            rec.t = t
            rec.material = self.material
            return True
        else:
            return False

class Camera:
    def __init__(self,
                 lookfrom : list,
                 lookat : list,
                 vup : list,
                 vfov : float, # vertical field-of-view in degrees
                 aspect_ratio : float,
                 aperture : float,
                 focus_dist : float) -> None:
        theta = deg_to_rad(vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self.w = norm_vec3([lookfrom[i] - lookat[i] for i in range(3)])
        self.u = norm_vec3(cross_vec3(vup, self.w))
        self.v = cross_vec3(self.w, self.u)

        self.origin = lookfrom
        self.horizontal = [focus_dist * viewport_width * self.u[i] for i in range(3)]
        self.vertical = [focus_dist * viewport_height * self.v[i] for i in range(3)]
        self.lower_left_corner = [self.origin[i] - self.horizontal[i]/2 - self.vertical[i]/2 - focus_dist * self.w[i] for i in range(3)]

        self.lens_radius = aperture / 2

    def get_ray(self, s : float, t : float):
        rd = random_in_unit_disk_vec3()
        rd = [self.lens_radius * rd[i] for i in range(3)]
        offset = [self.u[i] * rd[0] + self.v[i] * rd[1] for i in range(3)]

        origin = [self.origin[i] + offset[i] for i in range(3)]
        direction = [self.lower_left_corner[i] + s * self.horizontal[i] + t * self.vertical[i] - self.origin[i] - offset[i] for i in range(3)]
        return Ray(origin, direction)

def ray_color(r : Ray, world : Hittable, depth: int):
    if depth <= 0:
        return [0, 0, 0]

    rec = HitRecord()
    if world.hit(r, make_interval(0.001, float("inf")), rec):
        is_scattered, attenuation, scattered = rec.material.scatter(r, rec)
        if is_scattered:
            rec_color = ray_color(scattered, world, depth - 1)
            return [attenuation[i] * rec_color[i] for i in range(3)]
        return [0, 0, 0]

    unit_direction = norm_vec3(r.direction)
    t = 0.5 * (unit_direction[1] + 1)
    mt = 1.0 - t
    return [mt * 1.0 + t * 0.5, mt * 1.0 + t * 0.7, mt * 1.0 + t * 1.0]

def raytrace(surface):
    pixel_data = [[[0 for _ in range(3)] for _ in range(SCR_WIDTH)] for _ in range(SCR_HEIGHT)]

    def add_color(x, y, v):
        pixel = pixel_data[y][x]
        pixel[0] += v[0]
        pixel[1] += v[1]
        pixel[2] += v[2]

    world = HittableList()

    material_ground = Lambertian([0.5, 0.5, 0.5])
    world.add(Sphere([0, -1000, 0], 1000, material_ground))

    for a in range(-3, 3):
        for b in range(-3, 3):
            choose_mat = random.random()
            center = (a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())
            if mag_vec3([center[0] - 4, center[1] - 0.2, center[2] - 0]) > 0.9:
                if choose_mat < 0.8:
                    albedo_1 = [random.random(), random.random(), random.random()]
                    albedo_2 = [random.random(), random.random(), random.random()]
                    albedo = [albedo_1[i] * albedo_2[i] for i in range(3)]
                    sphere_material = Lambertian(albedo)
                    world.add(Sphere(center, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    albedo = [random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1)]
                    fuzz = random.uniform(0.5, 1)
                    sphere_material = Metal(albedo, fuzz)
                    world.add(Sphere(center, 0.2, sphere_material))
                else:
                    sphere_material = Dielectric(1.5)
                    world.add(Sphere(center, 0.2, sphere_material))

    material1 = Dielectric(1.5)
    world.add(Sphere([0, 1, 0], 1, material1))

    material2 = Lambertian([0.4, 0.2, 0.1])
    world.add(Sphere([-4, 1, 0], 1, material2))

    material3 = Metal([0.7, 0.6, 0.5], 0.0)
    world.add(Sphere([4, 1, 0], 1, material3))

    bvh_world = BvhNode(world)
    scene = HittableList()
    scene.add(bvh_world)
    # scene = world # comment out above and uncomment this to not use BVH

    max_depth = 50
    samples_per_pixel = 10

    aspect_ratio = SCR_WIDTH / float(SCR_HEIGHT)
    lookfrom = [13, 2, 3]
    lookat = [0, 0, 0]
    vup = [0, 1, 0]
    dist_to_focus = 10.0
    aperture = 0.1
    cam = Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus)

    for y in range(SCR_HEIGHT):
        if y % 10 == 0:
            print(f"Rendering line {y}...")
        for x in range(SCR_WIDTH):
            for _ in range(samples_per_pixel):
                u = (x + random.random()) / float(SCR_WIDTH - 1)
                v = (y + random.random()) / float(SCR_HEIGHT - 1)
                r = cam.get_ray(u, v)
                pixel_color = ray_color(r, scene, max_depth)
                add_color(x, y, pixel_color)

    scale = 1.0 / samples_per_pixel
    for y in range(SCR_HEIGHT):
        for x in range(SCR_WIDTH):
            pixel = pixel_data[y][x]
            gamma_pixel = [(scale * pixel[i]) ** 0.5 for i in range(3)]
            color = [int(255 * gamma_pixel[i]) for i in range(3)]
            surface.set_at((x, SCR_HEIGHT - 1 - y), color)

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE)
    pygame.display.set_caption("Ray tracing demo")
    clock = pygame.time.Clock()

    offscreen = pygame.Surface(SCR_SIZE)
    offscreen.fill(RGB_BLACK)

    start = timer()
    raytrace(offscreen)
    end = timer()
    print(f"Render time {end - start} s")

    frame = 0
    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
        screen.fill(RGB_BLACK)
        screen.blit(offscreen, (0,0))
        pygame.display.flip()
        frame += 1

if __name__ == '__main__':
    main_function()
