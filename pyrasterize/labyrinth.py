"""
A flat area consisting of quadratic tiles that are either floor or wall
"""

import math

from . import rasterizer
from . import vecmat

class Labyrinth:
    def __init__(self, camera : dict, tile_size : float, ceil_height : float) -> None:
        self.camera = camera
        self.tile_size = tile_size
        self.ceil_height = ceil_height

    def set_area(self, tiles : list, size : tuple):
        """
        """
        self.tiles = tiles
        self.rows,self.cols = size

    def create_floor_and_ceiling(self,
                                 scene_graph_root_instance,
                                 floor_model,
                                 ceil_model,
                                 floor_preproc_m4=vecmat.get_unit_m4(),
                                 ceil_preproc_m4=vecmat.get_unit_m4()):
        """
        Creates instances named tile_[row]_[col] under the given scene graph root,
        a floor and ceiling each. Ceiling model is translated up the given amount.
        Floor and ceiling models length along x and z must be (after applying
        preprocess matrix) = tile size and its middle is assumed to be at model space origin.
        """
        ceil_preproc_m4 = vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, self.ceil_height, 0),
                                               ceil_preproc_m4)

        for row in range(self.rows):
            row_tiles = self.tiles[row]
            for col in range(self.cols):
                tile_char = row_tiles[col]
                if tile_char != "#":
                    tile_name = f"tile_{row}_{col}"
                    scene_graph_root_instance["children"][tile_name] = rasterizer.get_model_instance(None)
                    tile_inst = scene_graph_root_instance["children"][tile_name]
                    tile_transl = vecmat.get_transl_m4(self.tile_size / 2 + self.tile_size * col,
                                                       0,
                                                       -self.tile_size / 2 + -self.tile_size * (self.rows - 1 - row))
                    tile_inst["children"]["floor"] = rasterizer.get_model_instance(floor_model,
                        preproc_m4=floor_preproc_m4, xform_m4=tile_transl, create_bbox=False)
                    tile_inst["children"]["ceiling"] = rasterizer.get_model_instance(ceil_model,
                        preproc_m4=ceil_preproc_m4, xform_m4=tile_transl, create_bbox=False)

                    tile_inst["children"]["floor"]["ignore_lighting"] = True
                    tile_inst["children"]["ceiling"]["ignore_lighting"] = True

    def create_walls(self,
                     scene_graph_root_instance,
                     wall_model,
                     wall_preproc_m4):
        """
        Creates instances 
        """
        wall_inst = rasterizer.get_model_instance(wall_model, preproc_m4=wall_preproc_m4, create_bbox=False)
        # Wall meshes are culled if not facing the camera.
        wall_inst["instance_normal"] = [0, 0, 1]
        wall_inst["use_minimum_z_order"] = True
        wall_inst["ignore_lighting"] = True

        for row in range(self.rows):
            row_tiles = self.tiles[row]
            for col in range(self.cols):
                tile_name = f"tile_{row}_{col}"
                scene_graph_root_instance["children"][tile_name] = rasterizer.get_model_instance(None,
                    xform_m4=vecmat.get_transl_m4(self.tile_size * col, 0,
                                                  -self.tile_size * (self.rows - 1 - row)))
                tile_inst = scene_graph_root_instance["children"][tile_name]

                wall_n = False
                wall_s = False
                wall_w = False
                wall_e = False

                tile = row_tiles[col]
                if tile == "#":
                    if row != 0 and self.tiles[row - 1][col] != "#":
                        wall_n = True
                    if row != self.rows -1 and self.tiles[row + 1][col] != "#":
                        wall_s = True
                    if col != 0 and self.tiles[row][col - 1] != "#":
                        wall_w = True
                    if col != self.cols - 1 and self.tiles[row][col + 1] != "#":
                        wall_e = True

                half_tile = self.tile_size / 2
                half_ceil = self.ceil_height / 2

                if wall_n:
                    tile_inst["children"]["wall_n"] = rasterizer.get_model_instance(None, None,
                        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(half_tile,
                                                                  half_ceil,
                                                                  -self.tile_size),
                        vecmat.get_rot_y_m4(vecmat.deg_to_rad(180))),
                        {"wall": wall_inst})
                if wall_s:
                    tile_inst["children"]["wall_s"] = rasterizer.get_model_instance(None, None,
                        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(half_tile,
                                                                  half_ceil,
                                                                  0),
                        vecmat.get_rot_y_m4(vecmat.deg_to_rad(0))),
                        {"wall": wall_inst})
                if wall_w:
                    tile_inst["children"]["wall_w"] = rasterizer.get_model_instance(None, None,
                        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0,
                                                                  half_ceil,
                                                                  -half_tile),
                        vecmat.get_rot_y_m4(vecmat.deg_to_rad(-90))),
                        {"wall": wall_inst})
                if wall_e:
                    tile_inst["children"]["wall_e"] = rasterizer.get_model_instance(None, None,
                        vecmat.mat4_mat4_mul(vecmat.get_transl_m4(self.tile_size,
                                                                  half_ceil,
                                                                  -half_tile),
                        vecmat.get_rot_y_m4(vecmat.deg_to_rad(90))),
                        {"wall": wall_inst})

    def update_viewable_area(self, view_max, root_instances):
        """
        """
        def enable_tile(row, col, enable):
            tile_name = f"tile_{row}_{col}"
            for root_instance in root_instances:
                children = root_instance["children"]
                if tile_name in children:
                    root_instance["children"][tile_name]["enabled"] = enable

        # Turn off everything
        for row in range(self.rows):
            for col in range(self.cols):
                enable_tile(row, col, False)

        def pos_to_cell(z, x):
            return [self.rows - 1 + int(z / self.tile_size), int(x / self.tile_size)]

        cam_rot_y = self.camera["rot"][1]
        cam_v_forward = [-math.cos(cam_rot_y), -math.sin(cam_rot_y)]

        step = self.tile_size / 4.0
        enables = set()
        for delta_angle in range(-60, 60, 2):
            delta_rad = vecmat.deg_to_rad(delta_angle)
            cos = math.cos(delta_rad)
            sin = math.sin(delta_rad)
            rot_forward = [cos * cam_v_forward[0] - sin * cam_v_forward[1], sin * cam_v_forward[0] + cos * cam_v_forward[1]]
            pos_zx = [self.camera["pos"][2], self.camera["pos"][0]]
            for _ in range(int(view_max / step)):
                pos_zx[0] += rot_forward[0] * step
                pos_zx[1] += rot_forward[1] * step
                row,col = pos_to_cell(pos_zx[0], pos_zx[1])
                if row < 0:
                    break
                if col < 0:
                    break
                if row >= self.rows:
                    break
                if col >= self.cols:
                    break
                if self.tiles[row][col] == "#":
                    enables.add((row, col))
                    break
                enables.add((row, col))

        for row,col in enables:
            enable_tile(row, col, True)

    def get_tile_pos(self, x, z):
        """
        Lower left corner of the map is at 0,0
        (the tile in the last row and first column)
        """
        row = self.rows - 1 + int(z / self.tile_size)
        col = int(x / self.tile_size)
        return row, col

    def tile_to_world_pos(self, row, col):
        x = col * self.tile_size
        z = (self.rows - 1 - row) * -self.tile_size
        return x,z

    def is_position_reachable(self, x, y, z):
        """
        Is this position in open air (i.e. not inside a wall)
        """
        if y < 0 or y > self.ceil_height:
            return False

        row,col = self.get_tile_pos(x, z)

        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False

        if self.tiles[row][col] == "#":
            return False

        return True

    def is_position_walkable(self, x, y, z, char_radius):
        if not self.is_position_reachable(x, y, z):
            return False

        # We are in a free cell, don't let char get closer than their radius to walls
        row,col = self.get_tile_pos(x, z)
        tile_x,tile_z = self.tile_to_world_pos(row, col)

        # Check if we are too close to any surrounding walls
        tiles = self.tiles
        # NW
        if (tiles[row - 1][col - 1] == "#"):
            if x < tile_x + char_radius and z < tile_z - self.tile_size + char_radius:
                return False
        # N
        if (tiles[row - 1][col] == "#"):
            if z < tile_z - self.tile_size + char_radius:
                return False
        # NE
        if (tiles[row - 1][col + 1] == "#"):
            if x > tile_x + self.tile_size - char_radius and z < tile_z - self.tile_size + char_radius:
                return False
        # E
        if (tiles[row][col + 1] == "#"):
            if x > (tile_x + self.tile_size - char_radius):
                return False
        # SE
        if (tiles[row + 1][col + 1] == "#"):
            if x > tile_x + self.tile_size - char_radius and z > tile_z - char_radius:
                return False
        # S
        if (tiles[row + 1][col] == "#"):
            if z > tile_z - char_radius:
                return False
        # SW
        if (tiles[row + 1][col - 1] == "#"):
            if x < tile_x + char_radius and z > tile_z - char_radius:
                return False
        # W
        if (tiles[row][col - 1] == "#"):
            if x < tile_x + char_radius:
                return False

        return True
