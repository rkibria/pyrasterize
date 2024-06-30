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
from pyrasterize import selecting

# SHELL GAME https://en.wikipedia.org/wiki/Shell_game
#
# Shell  Shell  Shell
#   0      1      2
#
# /---\  /---\  /---\
# |   |  |   |  |   |
# |   |  |   |  |   |
#
#   O <-- Pea location = 0/1/2
#
# Possible shell swaps:
# 0-1, 1-2, 0-2
# - each move can be animated clockwise or counter-clockwise,
#   but has the same result.

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

# VIEW SETTINGS

SCR_SIZE = SCR_WIDTH, SCR_HEIGHT = 640, 480
SCR_AREA = (0, 0, SCR_WIDTH, SCR_HEIGHT)

CAMERA = { "pos": [0, 4, 7],
    "rot": [vecmat.deg_to_rad(-20), 0, 0],
    "fov": 90,
    "ar": SCR_WIDTH/SCR_HEIGHT }

render_settings = rasterizer.get_default_render_settings()

# 3D ANIMATION FUNCTIONS

SHELL_DIST = 2.5
SHELL_LENGTH = 2
PEA_RADIUS = 0.5

def create_scene_graph():
    """Create the main scene graph"""
    scene_graph = { "root": rasterizer.get_model_instance(None) }
    for i in range(3):
        name = "shell_" + str(i)
        scene_graph["root"]["children"][name] = rasterizer.get_model_instance(
            meshes.get_cylinder_mesh(SHELL_LENGTH, 1, 10, (100, 100, 230),
            close_bottom=False, top_offset=0.5),
            xform_m4=vecmat.get_transl_m4(-SHELL_DIST + i * SHELL_DIST, 0, 0))
        shell = scene_graph["root"]["children"][name]
        shell["bound_sph_r"] = 1
        shell["gouraud"] = True
        shell["gouraud_max_iterations"] = 3
    scene_graph["root"]["children"]["pea"] = rasterizer.get_model_instance(
        meshes.get_sphere_mesh(PEA_RADIUS, 16, 16, (200, 20, 20)),
        xform_m4=vecmat.get_transl_m4(-SHELL_DIST, 0, 0))
    pea = scene_graph["root"]["children"]["pea"]
    pea["gouraud"] = True
    return scene_graph

def set_pea_pos(scene_graph, n_shell):
    """Set pea position at shell"""
    inst = scene_graph["root"]["children"]["pea"]
    y = -SHELL_LENGTH/2 + PEA_RADIUS
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
        rotate_shell_around_point(scene_graph, 0, -SHELL_DIST/2, 0, 0, angle, SHELL_DIST/2)
        rotate_shell_around_point(scene_graph, 1, -SHELL_DIST/2, 0, 0, angle + math.pi, SHELL_DIST/2)
    elif swap == SWAP_02:
        rotate_shell_around_point(scene_graph, 0, 0, 0, 0, angle, SHELL_DIST)
        rotate_shell_around_point(scene_graph, 2, 0, 0, 0, angle + math.pi, SHELL_DIST)
    elif swap == SWAP_12:
        rotate_shell_around_point(scene_graph, 1, SHELL_DIST/2, 0, 0, angle, SHELL_DIST/2)
        rotate_shell_around_point(scene_graph, 2, SHELL_DIST/2, 0, 0, angle + math.pi, SHELL_DIST/2)

# GAME STATE AND GAME LOGIC HELPERS

GS_WAIT_FOR_START = 0  # Waiting for player to click to start game
GS_SHOW_PEA_START = 1  # Player started, showing pea start location
GS_SWAPPING = 2        # Swapping shells until done
GS_WAIT_FOR_CHOICE = 3 # Waiting for player to choose shell
GS_REVEAL = 4          # Lift shell the player chose
GS_GAME_OVER = 5       # Game over, restart if player clicks

def create_game_state():
    """Generate new game state dict"""
    game_state = {
        "state": GS_WAIT_FOR_START,
        "swap_done": False,
        "cur_swap": 0,
        "swap_clockwise": 0,
        "cur_frame": 0,
        "rotate_speeds": [15, 10], # narrow/wide swap rotation speed
        "remaining_swaps": 3,
        "pea_loc": random.randint(0, 2),
        "button_pressed": False,
        "selected_shell": None,
        "selected_shell_idx": None
    }
    return game_state

# Possible new swaps depending on old swap
NEW_SWAP_TABLE = {
    SWAP_01: [SWAP_02, SWAP_12],
    SWAP_02: [SWAP_01, SWAP_12],
    SWAP_12: [SWAP_01, SWAP_02],
}

def init_swap(game_state):
    """Set state to start of a new random swap"""
    game_state["cur_swap"] = NEW_SWAP_TABLE[game_state["cur_swap"]][random.randint(0, 1)]
    game_state["swap_clockwise"] = random.randint(0, 1)
    game_state["cur_frame"] = 0
    game_state["swap_done"] = False

def perform_swap(game_state):
    """Perform the swap indicated by current state"""
    game_state["pea_loc"] = get_new_pea_loc(game_state["pea_loc"], game_state["cur_swap"])

# DRAWING HELPERS

def animate_shell_swapping(scene_graph, game_state):
    """
    Animate the shell swapping
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

def draw_centered_text(surface, font_cache, string, pos):
    """Draw text centered at position"""
    if not string in font_cache:
        font_cache[string] = font_cache["_FONT_OBJ"].render(string, True, (255, 255, 255))
    img = font_cache[string]
    text_rect = img.get_rect(center=pos)
    surface.blit(img, text_rect)

# MAIN GAME LOGIC AND DRAWING

def run_game_state_machine(game_state, scene_graph):
    """Run the game logic and animate scene graph"""
    if game_state["state"] == GS_WAIT_FOR_START:
        enable_pea(scene_graph, False)
        set_pea_pos(scene_graph, game_state['pea_loc'])
        if game_state["button_pressed"]:
            enable_pea(scene_graph, True)
            game_state["button_pressed"] = False
            game_state["state"] = GS_SHOW_PEA_START
    elif game_state["state"] == GS_SHOW_PEA_START:
        angle = 5 * vecmat.deg_to_rad(game_state["cur_frame"] * 1.95)
        set_shell_pos(scene_graph, game_state['pea_loc'],
            -SHELL_DIST + game_state["pea_loc"] * SHELL_DIST,
            abs(math.sin(angle)) * 3, 0)
        game_state["cur_frame"] += 1
        if angle >= math.pi:
            game_state["cur_frame"] = 0
            enable_pea(scene_graph, False)
            game_state["state"] = GS_SWAPPING
    elif game_state["state"] == GS_SWAPPING:
        if animate_shell_swapping(scene_graph, game_state):
            perform_swap(game_state)
            set_pea_pos(scene_graph, game_state["pea_loc"])
            reset_shell_positions(scene_graph)
            if game_state["remaining_swaps"] > 0:
                game_state["remaining_swaps"] -= 1
                init_swap(game_state)
                animate_shell_swapping(scene_graph, game_state)
            else:
                game_state["state"] = GS_WAIT_FOR_CHOICE
    elif game_state["state"] == GS_WAIT_FOR_CHOICE:
        if game_state["selected_shell"] is not None:
            game_state["cur_frame"] = 0
            game_state["selected_shell_idx"] = {"shell_0": 0, "shell_1": 1, "shell_2": 2}[game_state["selected_shell"][0]]
            enable_pea(scene_graph, True)
            game_state["state"] = GS_REVEAL
    elif game_state["state"] == GS_REVEAL:
        angle = 5 * vecmat.deg_to_rad(game_state["cur_frame"] * 1.95)
        set_shell_pos(scene_graph, game_state["selected_shell_idx"],
            -SHELL_DIST + game_state["selected_shell_idx"] * SHELL_DIST,
            abs(math.sin(angle)) * 3, 0)
        if angle <= math.pi/2:
            game_state["cur_frame"] += 1
        else:
            game_state["state"] = GS_GAME_OVER
    elif game_state["state"] == GS_GAME_OVER:
        if game_state["button_pressed"]:
            reset_shell_positions(scene_graph)
            new_game_state = create_game_state()
            for k,v in new_game_state.items():
                game_state[k] = v
            init_swap(game_state)
            set_pea_pos(scene_graph, game_state["pea_loc"])

def draw_game_state(surface, font_cache, game_state, scene_graph):
    """Draw and animate the scene graph and anything else related to the game"""
    title_pos = (SCR_WIDTH/2, SCR_HEIGHT/6)
    if game_state["state"] == GS_WAIT_FOR_START:
        draw_centered_text(surface, font_cache, "Click left button to start", title_pos)
    elif game_state["state"] == GS_WAIT_FOR_CHOICE:
        draw_centered_text(surface, font_cache, "Click on the shell hiding the pea", title_pos)
    elif game_state["state"] == GS_GAME_OVER:
        if game_state["pea_loc"] == game_state["selected_shell_idx"]:
            draw_centered_text(surface, font_cache, "Correct! Click left button to play again", title_pos)
        else:
            draw_centered_text(surface, font_cache, "Sorry, wrong! Click left button to play again", title_pos)

    persp_m = vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"])
    rasterizer.render(surface, SCR_AREA, scene_graph,
        vecmat.get_simple_camera_m(CAMERA), persp_m, render_settings)

def on_left_down(pos, game_state, scene_graph):
    """Handle left button down"""
    if game_state["state"] == GS_WAIT_FOR_START or game_state["state"] == GS_GAME_OVER:
        game_state["button_pressed"] = True
    elif game_state["state"] == GS_WAIT_FOR_CHOICE:
        m = vecmat.mat4_mat4_mul(vecmat.get_persp_m4(vecmat.get_view_plane_from_fov(CAMERA["fov"]), CAMERA["ar"]),
                                vecmat.get_simple_camera_m(CAMERA))
        game_state["selected_shell"] = selecting.get_selection(SCR_AREA, pos, scene_graph, m)

# PYGAME MAIN

def main_function():
    """Main"""
    pygame.init()
    random.seed()

    screen = pygame.display.set_mode(SCR_SIZE, pygame.SCALED)
    pygame.display.set_caption("Shell Game")
    clock = pygame.time.Clock()

    pygame.mouse.set_cursor(*pygame.cursors.broken_x)

    scene_graph = create_scene_graph()
    game_state = create_game_state()
    init_swap(game_state)

    font = pygame.font.Font(None, 30)
    font_cache = {"_FONT_OBJ" : font}

    done = False
    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_left_down(pygame.mouse.get_pos(), game_state, scene_graph)

        screen.fill(render_settings["fog_color"])

        run_game_state_machine(game_state, scene_graph)
        draw_game_state(screen, font_cache, game_state, scene_graph)
        pygame.display.flip()

if __name__ == '__main__':
    main_function()
