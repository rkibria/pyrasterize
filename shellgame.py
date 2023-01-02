"""
Simulates the classic shell game with 3d models
"""

import math
import random

import pygame
import pygame.gfxdraw
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import rasterizer
from pyrasterize import meshes

# GAME LOGIC
#
# SHELL GAME https://en.wikipedia.org/wiki/Shell_game
#
# Shell  Shell  Shell
#   0      1      2
#
# /---\  /---\  /---\
# |   |  |   |  |   |
# |   |  |   |  |   |
#
#   O <-- Pea location
#
# Game state: pea location = 0/1/2
#
# Possible shell swaps/game operations:
# 0-1, 1-2, 0-2
# - each move can be animated clockwise or counter-clockwise,
#   but has the same result.
# - a swap swaps the position of the pea if the pea is in
#   either one of the affected shell positions.
#

SWAP_01 = 0
SWAP_12 = 1
SWAP_02 = 2

# Key: old pea location, Values: new location depending on swap
SWAP_RESULT_TABLE = {
    0: {SWAP_01: 1, SWAP_12: 0, SWAP_02: 2},
    1: {SWAP_01: 0, SWAP_12: 2, SWAP_02: 1},
    2: {SWAP_01: 2, SWAP_12: 1, SWAP_02: 0},
}

def get_new_pea_loc(pea_loc, n_swap):
    """Returns new pea location depending on swap"""
    return SWAP_RESULT_TABLE[pea_loc][n_swap]

# CONSTANTS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 800, 600
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

RGB_BLACK = (0, 0, 0)
RGB_WHITE = (255, 255, 255)
RGB_GREEN = (0, 255, 0)

CAMERA = { "pos": [0, 4, 7],
    "rot": [vecmat.deg_to_rad(-20), 0, 0],
    "fov": 90,
    "ar": SCR_WIDTH/SCR_HEIGHT }
LIGHTING = {"lightDir" : (1, 1, 1), "ambient": 0.3, "diffuse": 0.7}
CUR_SELECTED = None
SHELL_DIST = 2.5
PEA_RADIUS = 0.5
START_PRESSED = False

def create_scene_graph():
    """Create the main scene graph"""
    scene_graph = { "root": rasterizer.get_model_instance(None) }
    for i in range(3):
        scene_graph["root"]["children"]["shell_" + str(i)] = rasterizer.get_model_instance(
            meshes.get_cylinder_mesh(2, 1, 50, (100, 100, 230), close_bottom=False),
            xform_m4=vecmat.get_transl_m4(-SHELL_DIST + i * SHELL_DIST, 0, 0))
        # scene_graph["root"]["children"][name]["wireframe"] = True
        # scene_graph["root"]["children"][name]["noCulling"] = True
    scene_graph["root"]["children"]["pea"] = rasterizer.get_model_instance(
        meshes.get_sphere_mesh(PEA_RADIUS, 6, 4, (200, 20, 20)),
        xform_m4=vecmat.get_transl_m4(-SHELL_DIST, 0, 0))
    scene_graph["root"]["children"]["pea"]["wireframe"] = True
    scene_graph["root"]["children"]["pea"]["noCulling"] = True

    return scene_graph

def set_pea_pos(scene_graph, n_shell, y=0):
    """Set pea position at shell"""
    inst = scene_graph["root"]["children"]["pea"]
    inst["xform_m4"] = vecmat.get_transl_m4(-SHELL_DIST + n_shell * SHELL_DIST, y, 0)

def set_shell_pos(scene_graph, n_shell, x, y, z):
    """Set cup n position"""
    inst = scene_graph["root"]["children"]["shell_" + str(n_shell)]
    inst["xform_m4"] = vecmat.get_transl_m4(x, y, z)

def rotate_shell_around_point(scene_graph, n_shell, px, pz, y, angle, radius):
    """Rotate a shell counter-clockwise by the angle around the point px,0,pz"""
    x = px + radius * math.cos(angle)
    z = pz + radius * math.sin(angle)
    set_shell_pos(scene_graph, n_shell, x, y, z)

def rotate_shell_01(scene_graph, angle):
    """Rotate shells 0 and 1 around their mid point"""
    rotate_shell_around_point(scene_graph, 0, -SHELL_DIST/2, 0, 0, angle, SHELL_DIST/2)
    rotate_shell_around_point(scene_graph, 1, -SHELL_DIST/2, 0, 0, angle + math.pi, SHELL_DIST/2)

def rotate_shell_12(scene_graph, angle):
    """Rotate shells 1 and 2 around their mid point"""
    rotate_shell_around_point(scene_graph, 1, SHELL_DIST/2, 0, 0, angle, SHELL_DIST/2)
    rotate_shell_around_point(scene_graph, 2, SHELL_DIST/2, 0, 0, angle + math.pi, SHELL_DIST/2)

def rotate_shell_02(scene_graph, angle):
    """Rotate shells 0 and 2 around their mid point"""
    rotate_shell_around_point(scene_graph, 0, 0, 0, 0, angle, SHELL_DIST)
    rotate_shell_around_point(scene_graph, 2, 0, 0, 0, angle + math.pi, SHELL_DIST)

def enable_pea(scene_graph, en):
    """Enable drawing of the pea"""
    scene_graph["root"]["children"]["pea"]["enabled"] = en

def reset_shell_positions(scene_graph):
    """Put all shells in their starting positions"""
    set_shell_pos(scene_graph, 0, -SHELL_DIST, 0, 0)
    set_shell_pos(scene_graph, 1, 0, 0, 0)
    set_shell_pos(scene_graph, 2, SHELL_DIST, 0, 0)

def run_swap(swap, scene_graph, angle):
    """Animate the current swap"""
    if swap == SWAP_01:
        rotate_shell_01(scene_graph, angle)
    elif swap == SWAP_02:
        rotate_shell_02(scene_graph, angle)
    elif swap == SWAP_12:
        rotate_shell_12(scene_graph, angle)

# Rotation speeds for small/big swaps depending on difficulty level
ROTATE_SPEEDS = {
    0: [5, 2],
    1: [15, 10],
    2: [35, 30]
}

GS_WAIT_FOR_START = 0 # Waiting for player to press start
GS_SHOW_PEA_START = 1 # Player started, showing pea start location
GS_SWAPPING = 2 # Swapping shells until done
GS_WAIT_FOR_CHOICE = 3 # Waiting for player to choose shell
GS_REVEAL = 4 # Lift shell the player chose

def create_game_state():
    """Generate new game state dict"""
    game_state = {
        "state": GS_WAIT_FOR_START,
        "swap_done": False,
        "cur_swap": 0,
        "swap_clockwise": 0,
        "cur_frame": 0,
        "rotate_speeds": ROTATE_SPEEDS[0],
        "remaining_swaps": 3,
        "pea_loc": 0
    }
    return game_state

# Possible new swaps depending on old swap, make a binary choice
NEW_SWAP_TABLE = {
    SWAP_01: [SWAP_02, SWAP_12],
    SWAP_02: [SWAP_01, SWAP_12],
    SWAP_12: [SWAP_01, SWAP_02],
}

def set_new_swap(game_state):
    """
    Set state to start of a new random swap
    Pea location is set to where it would be at END of this swap
    """
    game_state["cur_swap"] = NEW_SWAP_TABLE[game_state["cur_swap"]][random.randint(0, 1)]
    game_state["swap_clockwise"] = random.randint(0, 1)
    game_state["cur_frame"] = 0
    game_state["swap_done"] = False
    new_loc = get_new_pea_loc(game_state["pea_loc"], game_state["cur_swap"])
    print(f"pea from {game_state['pea_loc']} to {new_loc}")
    game_state["pea_loc"] = new_loc

def advance_game_state(scene_graph, game_state):
    """
    Animate the current state of the game
    Return True if reached last animation frame
    """
    if game_state["cur_frame"] == 0:
        game_state["cur_frame"] = 1
    degs_per_frame = game_state["rotate_speeds"][0 if game_state["cur_swap"] != SWAP_02 else 1]
    degs = min(180, game_state["cur_frame"] * degs_per_frame)
    angle = vecmat.deg_to_rad(degs)
    angle = angle if game_state["swap_clockwise"] == 0 else -angle
    run_swap(game_state["cur_swap"], scene_graph, angle)
    game_state["cur_frame"] += 1
    if degs >= 180:
        game_state["swap_done"] = True
    return game_state["swap_done"]

def run_game(scene_graph, game_state):
    """Run the game logic and animate scene graph"""
    # angle = 10 * vecmat.deg_to_rad(frame)
    # enable_pea(scene_graph, False)
    # set_shell_pos(scene_graph, 0, -SHELL_DIST, abs(math.sin(5 * vecmat.deg_to_rad(frame))) * 3, 0)

    global START_PRESSED
    if game_state["state"] == GS_WAIT_FOR_START:
        if START_PRESSED:
            game_state["state"] = GS_SWAPPING
    elif game_state["state"] == GS_SWAPPING:
        if advance_game_state(scene_graph, game_state):
            reset_shell_positions(scene_graph)
            set_pea_pos(scene_graph, game_state["pea_loc"])
            if game_state["remaining_swaps"] > 0:
                game_state["remaining_swaps"] -= 1
                set_new_swap(game_state)
                advance_game_state(scene_graph, game_state)

def draw_scene_graph(surface, _, scene_graph):
    """Draw and animate the scene graph"""
    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    rasterizer.render(surface, SCR_AREA, scene_graph,
        vecmat.get_simple_camera_m(CAMERA), persp_m, LIGHTING)

# MAIN

def on_left_down(pos, scene_graph):
    """Handle left button down"""
    global CUR_SELECTED
    selection = rasterizer.get_selection(SCR_AREA, pos, scene_graph,
        vecmat.get_simple_camera_m(CAMERA))
    if CUR_SELECTED is not None:
        CUR_SELECTED["wireframe"] = False
        CUR_SELECTED["noCulling"] = False
    if selection:
        CUR_SELECTED = selection
        CUR_SELECTED["wireframe"] = True
        CUR_SELECTED["noCulling"] = True
    else:
        CUR_SELECTED = None

    global START_PRESSED
    if pos[0] < 100 and pos[1] < 100:
        START_PRESSED = True

def main_function():
    """Main"""
    pygame.init()
    random.seed()

    screen = pygame.display.set_mode(SCR_SIZE)
    pygame.display.set_caption("PyRasterize")
    clock = pygame.time.Clock()

    pygame.mouse.set_cursor(*pygame.cursors.broken_x)

    scene_graph = create_scene_graph()
    game_state = create_game_state()
    set_new_swap(game_state)

    # font = pygame.font.Font(None, 30)

    frame = 0
    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_left_down(pygame.mouse.get_pos(), scene_graph)

        screen.fill(RGB_BLACK)

        run_game(scene_graph, game_state)
        draw_scene_graph(screen, frame, scene_graph)

        pygame.draw.rect(screen, (0, 200, 0), (0, 0, 100, 100))

        pygame.display.flip()
        frame += 1
        # if frame % 30 == 0:
        #     print(f"{clock.get_fps()} fps")

if __name__ == '__main__':
    main_function()

