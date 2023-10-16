"""
https://raytracing.github.io/books/RayTracingInOneWeekend.html
"""

import random
from timeit import default_timer as timer

import pygame
import pygame.gfxdraw

from pyrasterize import vecmat
from pyrasterize.raytracer import *

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 320, 240
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)

CAMERA = { "pos": [0,0,3], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}


def ray_color(r : Ray, world : Hittable, depth: int):
    if depth <= 0:
        return [0, 0, 0]

    rec = HitRecord()
    if world.hit(r, Interval(0.001, float("inf")), rec):
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

    for a in range(-3, 3):
        for b in range(-3, 3):
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

    # zz = 10
    # for i in range(zz):
    #     center = [(-zz/2 + i), 2.2, 0]
    #     print(f"center {center}")
    #     # albedo_1 = [random.random(), random.random(), random.random()]
    #     # albedo_2 = [random.random(), random.random(), random.random()]
    #     # albedo = [albedo_1[i] * albedo_2[i] for i in range(3)]
    #     albedo = [0.9, 1.0 / zz * (i + 1), 0.0]
    #     sphere_material = Lambertian(albedo)
    #     world.add(Sphere(center, 0.2, sphere_material))

    bvh_world = BvhNode(world)
    scene = HittableList()
    scene.add(bvh_world)
    # scene = world #########

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
        if y % 10 == 0:
            print(f"y = {y}")
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

    screen = pygame.display.set_mode(SCR_SIZE) # , flags=pygame.SCALED
    pygame.display.set_caption("pyrasterize demo")
    clock = pygame.time.Clock()

    # font = pygame.font.Font(None, 30)

    offscreen = pygame.Surface(SCR_SIZE)
    offscreen.fill(RGB_BLACK)

    start = timer()
    raytrace(offscreen)
    end = timer()
    print(f"Render time {end - start}")

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
