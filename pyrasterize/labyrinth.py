"""
A flat area consisting of quadratic tiles that are either floor or wall
"""

from . import rasterizer
from . import vecmat

class Labyrinth:
    def __init__(self, tile_size : float) -> None:
        self.tile_size = tile_size

    def set_area(self, tiles : list, size : tuple):
        """
        """
        self.tiles = tiles
        self.rows,self.cols = size

    def create_floor_and_ceiling(self,
                                 scene_graph_root_instance,
                                 ceil_height,
                                 floor_model,
                                 ceil_model,
                                 floor_preproc_m4=vecmat.get_unit_m4(),
                                 ceil_preproc_m4=vecmat.get_unit_m4()):
        """
        """
        ceil_preproc_m4 = vecmat.mat4_mat4_mul(vecmat.get_transl_m4(0, ceil_height, 0),
                                               ceil_preproc_m4)

        for row in range(self.rows):
            row_tiles = self.tiles[row]
            for col in range(self.cols):
                tile_char = row_tiles[col]
                if tile_char != "#":
                    tile_name = f"tile_{row}_{col}"
                    scene_graph_root_instance["children"][tile_name] = rasterizer.get_model_instance(None)
                    tile_inst = scene_graph_root_instance["children"][tile_name]
                    tile_inst["children"]["floor"] = rasterizer.get_model_instance(floor_model,
                        preproc_m4=floor_preproc_m4,
                        xform_m4=vecmat.get_transl_m4(self.tile_size / 2 + self.tile_size * col,
                                                      0,
                                                      -self.tile_size / 2 + -self.tile_size * (self.rows - 1 - row)),
                                                      create_bbox=False)

                    tile_inst["children"]["ceiling"] = rasterizer.get_model_instance(ceil_model,
                        preproc_m4=ceil_preproc_m4,
                        xform_m4=vecmat.get_transl_m4(self.tile_size / 2 + self.tile_size * col,
                                                      0,
                                                      -self.tile_size / 2 + -self.tile_size * (self.rows - 1 - row)),
                                                      create_bbox=False)
