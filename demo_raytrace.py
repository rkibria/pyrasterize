"""
Template for demos
"""

import copy
import random

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

def near_zero_vec3(v : list):
    s = 1e-8
    return abs(v[0]) < s and abs(v[1]) < s and abs(v[2]) < s

def reflect_vec3(v : list, n : list):
    dot_vn = vecmat.dot_product_vec3(v, n)
    return [v[i] - 2 * dot_vn * n[i] for i in range(3)]

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
    def __init__(self, albedo : list) -> None:
        super().__init__()
        self.albedo = albedo

    # Return (is_scattered : bool, attenuation : vec3, scattered : Ray)
    def scatter(self, r_in : Ray, rec : HitRecord):
        norm_r_in_dir = vecmat.norm_vec3(r_in.direction)
        reflected = reflect_vec3(norm_r_in_dir, rec.normal)
        scattered = Ray(rec.hit_point, reflected)
        return (vecmat.dot_product_vec3(scattered.direction, rec.normal) > 0, self.albedo, scattered)

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
    def __init__(self) -> None:
        aspect_ratio = SCR_WIDTH / float(SCR_HEIGHT)
        viewport_height = 2.0
        viewport_width = aspect_ratio * viewport_height
        focal_length = 1.0

        self.origin = (0, 0, 0)
        self.horizontal = (viewport_width, 0, 0)
        self.vertical = (0, viewport_height, 0)
        self.lower_left_corner = [
            self.origin[0] - self.horizontal[0]/2 - self.vertical[0]/2,
            self.origin[1] - self.horizontal[1]/2 - self.vertical[1]/2,
            self.origin[2] - self.horizontal[2]/2 - self.vertical[2]/2 - focal_length]

    def get_ray(self, u : float, v : float):
        direction = [
            self.lower_left_corner[i] + u * self.horizontal[i] + v * self.vertical[i] - self.origin[i] for i in range(3)
        ]
        return Ray(self.origin, direction)


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

    aspect_ratio = SCR_WIDTH / float(SCR_HEIGHT)
    viewport_height = 2.0
    viewport_width = aspect_ratio * viewport_height
    focal_length = 1.0

    material_ground = Lambertian([0.8, 0.8, 0.0])
    material_center = Lambertian([0.7, 0.3, 0.3])
    material_left = Metal([0.8, 0.8, 0.8])
    material_right = Metal([0.8, 0.6, 0.2])

    world = HittableList()
    world.add(Sphere([0, -100.5, -1], 100, material_ground))
    world.add(Sphere([0, 0, -1], 0.5, material_center))
    world.add(Sphere([-1, 0, -1], 0.5, material_left))
    world.add(Sphere([1, 0, -1], 0.5, material_right))

    max_depth = 50
    samples_per_pixel = 2

    cam = Camera()
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
            pixel = [(scale * pixel[i]) ** 0.5 for i in range(3)]
            color = [int(255 * pixel[i]) for i in range(3)]
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
