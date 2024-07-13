"""
A flat area consisting of quadratic tiles that are either floor or wall
"""

from . import rasterizer
from . import vecmat

class Labyrinth:
    def __init__(self, tile_size : float, ceil_height : float) -> None:
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
