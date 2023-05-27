"""
Template for demos
"""

import copy
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

    def copy(self, rec : "HitRecord"):
        self.hit_point = copy.copy(rec.hit_point)
        self.normal = copy.copy(rec.normal)
        self.t = rec.t
        self.front_face = rec.front_face

    def set_face_normal(self, r : Ray, outward_normal):
        self.front_face = vecmat.dot_product_vec3(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else [-outward_normal[i] for i in range(3)]

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
            if object.hit(r, t_min, t_max, temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.copy(temp_rec)

        return hit_anything

class Sphere(Hittable):
    def __init__(self, center, radius) -> None:
        super().__init__()
        self.center = center
        self.radius = radius

    def hit(self, r : Ray, t_min : float, t_max : float, rec: HitRecord) -> bool:
        t = vecmat.ray_sphere_intersect(r.origin, r.direction, self.center, self.radius, t_min, t_max)
        if t:
            hit_point = r.at(t)
            outward_normal = [hit_point[i] - self.center[i] for i in range(3)]
            rec.hit_point = hit_point
            rec.set_face_normal(r, outward_normal)
            rec.t = t
            return True
        else:
            return False

def ray_color(r : Ray):
    sph_origin = [0, 0, -1]
    sph_radius = 0.5
    t = vecmat.ray_sphere_intersect(r.origin, r.direction, sph_origin, sph_radius)
    if t:
        hit_point = r.at(t)
        normal = [hit_point[i] - sph_origin[i] for i in range(3)]
        return [0.5 * (normal[i] + 1) for i in range(3)]

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

    origin = (0, 0, 0)
    horizontal = (viewport_width, 0, 0)
    vertical = (0, viewport_height, 0)
    lower_left_corner = [
        origin[0] - horizontal[0]/2 - vertical[0]/2,
        origin[1] - horizontal[1]/2 - vertical[1]/2,
        origin[2] - horizontal[2]/2 - vertical[2]/2 - focal_length]
    for y in range(SCR_HEIGHT):
        for x in range(SCR_WIDTH):
            u = x / float(SCR_WIDTH - 1)
            v = y / float(SCR_HEIGHT - 1)
            direction = [
                lower_left_corner[i] + u * horizontal[i] + v * vertical[i] - origin[i] for i in range(3)
            ]
            r = Ray(origin, direction)
            pixel_color = ray_color(r)
            add_color(x, y, pixel_color)

    for y in range(SCR_HEIGHT):
        for x in range(SCR_WIDTH):
            pixel = pixel_data[y][x]
            color = [int(255 * pixel[i]) for i in range(3)]
            surface.set_at((x,y), color)

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(SCR_SIZE, flags=pygame.SCALED)
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
