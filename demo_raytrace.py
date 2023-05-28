"""
https://raytracing.github.io/books/RayTracingInOneWeekend.html
"""

import copy
import random
import math

import pygame
import pygame.gfxdraw

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 320, 240
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

CAMERA = { "pos": [0,0,3], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}

def random_in_unit_sphere_vec3():
    while True:
        p = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        if vecmat.mag_sq_vec3(p) >= 1:
            continue
        return p

def random_unit_vector_vec3():
    return vecmat.norm_vec3(random_in_unit_sphere_vec3())

def random_in_hemisphere(normal : list):
    in_unit_sphere = random_in_unit_sphere_vec3()
    if vecmat.dot_product_vec3(in_unit_sphere, normal) > 0:
        return in_unit_sphere
    else:
        return [-in_unit_sphere[i] for i in range(3)]

def random_in_unit_disk_vec3():
    while True:
        p = [random.uniform(-1, 1), random.uniform(-1, 1), 0]
        if vecmat.mag_sq_vec3(p) >= 1:
            continue
        return p

def near_zero_vec3(v : list):
    s = 1e-8
    return abs(v[0]) < s and abs(v[1]) < s and abs(v[2]) < s

def reflect_vec3(v : list, n : list):
    dot_vn = vecmat.dot_product_vec3(v, n)
    return [v[i] - 2 * dot_vn * n[i] for i in range(3)]

def refract_vec3(uv : list, n : list, etai_over_etat : float):
    minus_uv = [-uv[i] for i in range(3)]
    cos_theta = min(vecmat.dot_product_vec3(minus_uv, n), 1.0)
    r_out_perp = [etai_over_etat * (uv[i] + cos_theta * n[i]) for i in range(3)]
    r_out_perp_length_squared = vecmat.mag_sq_vec3(r_out_perp)
    k = -((abs(1 - r_out_perp_length_squared)) ** 0.5)
    r_out_parallel = [n[i] * k for i in range(3)]
    return [r_out_perp[i] + r_out_parallel[i] for i in range(3)]

class Ray:
    def __init__(self, origin, direction) -> None:
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

    def copy(self, rec : "HitRecord"):
        self.hit_point = copy.copy(rec.hit_point)
        self.normal = copy.copy(rec.normal)
        self.t = rec.t
        self.front_face = rec.front_face
        self.material = rec.material

    def set_face_normal(self, r : Ray, outward_normal):
        self.front_face = vecmat.dot_product_vec3(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else [-outward_normal[i] for i in range(3)]

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
        norm_r_in_dir = vecmat.norm_vec3(r_in.direction)
        reflected = reflect_vec3(norm_r_in_dir, rec.normal)
        rand_v = random_in_unit_sphere_vec3()
        reflected = [reflected[i] + self.fuzz * rand_v[i] for i in range(3)]
        scattered = Ray(rec.hit_point, reflected)
        return (vecmat.dot_product_vec3(scattered.direction, rec.normal) > 0, self.albedo, scattered)

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
        unit_direction = vecmat.norm_vec3(r_in.direction)

        minus_unit_direction = [-unit_direction[i] for i in range(3)]
        cos_theta = min(vecmat.dot_product_vec3(minus_unit_direction, rec.normal), 1.0)
        sin_theta = (1.0 - cos_theta ** 2) ** 0.5

        cannot_refract = refraction_ratio * sin_theta > 1.0

        if cannot_refract:
            direction = reflect_vec3(unit_direction, rec.normal)
        else:
            direction = refract_vec3(unit_direction, rec.normal, refraction_ratio)

        scattered = Ray(rec.hit_point, direction)
        return (True, [1.0, 1.0, 1.0], scattered)

class Hittable:
    def __init__(self) -> None:
        pass

    def hit(self, r : Ray, t_min : float, t_max : float, rec: HitRecord) -> bool:
        return False

class HittableList:
    def __init__(self) -> None:
        self.objects : list(Hittable) = []

    def add(self, object : Hittable) -> None:
        self.objects.append(object)

    def hit(self, r : Ray, t_min : float, t_max : float, rec: HitRecord) -> bool:
        temp_rec = HitRecord()
        hit_anything = False
        closest_so_far = t_max

        for object in self.objects:
            if object.hit(r, t_min, closest_so_far, temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.copy(temp_rec)

        return hit_anything

class Sphere(Hittable):
    def __init__(self, center : list, radius : float, material : Material) -> None:
        super().__init__()
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, r : Ray, t_min : float, t_max : float, rec: HitRecord) -> bool:
        t = vecmat.ray_sphere_intersect(r.origin, r.direction, self.center, self.radius, t_min, t_max)
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
        theta = vecmat.deg_to_rad(vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        self.w = vecmat.norm_vec3([lookfrom[i] - lookat[i] for i in range(3)])
        self.u = vecmat.norm_vec3(vecmat.cross_vec3(vup, self.w))
        self.v = vecmat.cross_vec3(self.w, self.u)

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
    if world.hit(r, 0.001, float("inf"), rec):
        is_scattered, attenuation, scattered = rec.material.scatter(r, rec)
        if is_scattered:
            rec_color = ray_color(scattered, world, depth - 1)
            return [attenuation[i] * rec_color[i] for i in range(3)]
        return [0, 0, 0]

    unit_direction = vecmat.norm_vec3(r.direction)
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

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = (a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())
            if vecmat.mag_vec3([center[0] - 4, center[1] - 0.2, center[2] - 0]) > 0.9:
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

    max_depth = 50
    samples_per_pixel = 2

    aspect_ratio = SCR_WIDTH / float(SCR_HEIGHT)
    lookfrom = [13, 2, 3]
    lookat = [0, 0, 0]
    vup = [0, 1, 0]
    dist_to_focus = 10.0
    aperture = 0.1
    cam = Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus)

    for y in range(SCR_HEIGHT):
        print(f"y = {y}")
        for x in range(SCR_WIDTH):
            for _ in range(samples_per_pixel):
                u = (x + random.random()) / float(SCR_WIDTH - 1)
                v = (y + random.random()) / float(SCR_HEIGHT - 1)
                r = cam.get_ray(u, v)
                pixel_color = ray_color(r, world, max_depth)
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

    screen = pygame.display.set_mode(SCR_SIZE) # , flags=pygame.SCALED
    pygame.display.set_caption("pyrasterize demo")
    clock = pygame.time.Clock()

    # scene_graph = {"root": rasterizer.get_model_instance(None)}
    # scene_graph["root"]["children"]["cube"] = rasterizer.get_model_instance(meshes.get_cube_mesh(), vecmat.get_scal_m4(0.1, 0.1, 0.1))
    # scene_graph["root"]["children"]["cube"]["wireframe"] = True
    # scene_graph["root"]["children"]["cube"]["noCulling"] = True

    # font = pygame.font.Font(None, 30)

    offscreen = pygame.Surface(SCR_SIZE)

    offscreen.fill(RGB_BLACK)
    raytrace(offscreen)

    frame = 0
    done = False
    # persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
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

        # rasterizer.render(screen, SCR_AREA, scene_graph,
        #     vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()
