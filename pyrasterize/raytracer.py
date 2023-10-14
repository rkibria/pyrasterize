#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Raytracer engine
"""

from . import vecmat

import copy
import math

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
        rand_v = vecmat.random_unit_vector_vec3()
        scatter_direction = [rec.normal[i] + rand_v[i] for i in range(3)]
        # Catch degenerate scatter direction
        if vecmat.near_zero_vec3(scatter_direction):
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
        reflected = vecmat.reflect_vec3(norm_r_in_dir, rec.normal)
        rand_v = vecmat.random_in_unit_sphere_vec3()
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
            direction = vecmat.reflect_vec3(unit_direction, rec.normal)
        else:
            direction = vecmat.refract_vec3(unit_direction, rec.normal, refraction_ratio)

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
        rd = vecmat.random_in_unit_disk_vec3()
        rd = [self.lens_radius * rd[i] for i in range(3)]
        offset = [self.u[i] * rd[0] + self.v[i] * rd[1] for i in range(3)]

        origin = [self.origin[i] + offset[i] for i in range(3)]
        direction = [self.lower_left_corner[i] + s * self.horizontal[i] + t * self.vertical[i] - self.origin[i] - offset[i] for i in range(3)]
        return Ray(origin, direction)

class Interval:
    """Default interval is empty"""

    def __init__(self,
                 min: float = float('inf'),
                 max: float = float('-inf')
                 ) -> None:
        self.min = min
        self.max = max

    def __str__(self) -> str:
        return f"Interval({self.min}, {self.max})"

    def __eq__(self, __value: object) -> bool:
        return self.min == __value.min and self.max == __value.max

    def size(self):
        return self.max - self.min

    def expand(self, delta : float):
        padding = delta / 2
        return Interval(self.min - padding, self.max + padding)

class AABB:
    """The default AABB is empty, since intervals are empty by default"""

    def __init__(self, ix = None, iy = None, iz = None) -> None:
        if ix is None:
            self.x = Interval()
            self.y = Interval()
            self.z = Interval()
        elif isinstance(ix, Interval):
            self.x = ix
            self.y = iy
            self.z = iz
        elif isinstance(ix, list): # ix = vec3, iy = vec3
            # Treat the two points a and b as extrema for the bounding box, so we don't require a
            # particular minimum/maximum coordinate order.
            a = ix
            b = iy
            self.x = Interval(min(a[0],b[0]), max(a[0],b[0]))
            self.y = Interval(min(a[1],b[1]), max(a[1],b[1]))
            self.z = Interval(min(a[2],b[2]), max(a[2],b[2]))
        self.pad_to_minimums()

    def pad_to_minimums(self):
        """Adjust the AABB so that no side is narrower than some delta, padding if necessary"""
        delta = 0.0001
        if self.x.size() < delta:
            self.x = self.x.expand(delta)
        if self.y.size() < delta:
            self.y = self.y.expand(delta)
        if self.z.size() < delta:
            self.z = self.z.expand(delta)

    def hit(self, r : Ray, ray_t : Interval) -> bool:
        for a,axis in zip([0, 1, 2], [self.x, self.y, self.z]):
            r_dir = r.direction[a]

            if r_dir:
                invD = 1 / r_dir
            else:
                invD = math.inf

            orig = r.origin[a]

            t0 = (axis.min - orig) * invD
            t1 = (axis.max - orig) * invD

            if invD < 0:
                t1, t0 = t0, t1

            if t0 > ray_t.min:
                ray_t.min = t0
            if t1 < ray_t.max:
                ray_t.max = t1
            
            if ray_t.max <= ray_t.min:
                return False
        return True
