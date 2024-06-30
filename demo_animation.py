"""
Animated geometry demo
"""

import pygame
import pygame.gfxdraw

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import model_file_io


def main_function():
    """Main"""
    pygame.init()

    # CONSTANTS
    SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 640, 480
    SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

    CAMERA = { "pos": [0, 1, 3], "rot": [0,0,0], "fov": 90, "ar": SCR_WIDTH/SCR_HEIGHT }
    render_settings = rasterizer.get_default_render_settings()
    screen = pygame.display.set_mode(SCR_SIZE, flags=pygame.SCALED)
    pygame.display.set_caption("pyrasterize demo")
    clock = pygame.time.Clock()

    scene_graph = {"root": rasterizer.get_model_instance(None)}

    scene_graph["root"]["children"]["my_animation"] = rasterizer.get_model_instance(model_file_io.load_animation({
        "idle": ("assets/dummy-idle.zip", (1, 60)),
        "walk": ("assets/dummy-walk.zip", (63, 123)),
        "run": ("assets/dummy-run.zip", (1, 35))
        }
        ))
    scene_graph["root"]["children"]["my_animation"]["animation"] = "walk"
    phases = ["idle", "walk", "run"]

    frame = 0
    done = False
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    while not done:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
        screen.fill(render_settings["fog_color"])

        scene_graph["root"]["xform_m4"] = vecmat.get_rot_y_m4(vecmat.deg_to_rad(frame * 0.5))

        rasterizer.render(screen, SCR_AREA, scene_graph,
            vecmat.get_simple_camera_m(CAMERA), persp_m, render_settings)

        pygame.display.flip()
        frame += 1

        if frame % 200 == 0:
            scene_graph["root"]["children"]["my_animation"]["animation"] = phases[(frame // 200) % 3]

if __name__ == '__main__':
    main_function()
