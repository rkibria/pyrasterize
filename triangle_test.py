import pygame
import pygame.gfxdraw
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import drawing

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 160, 120
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)
RGB_BLACK = (0, 0, 0)

def main_function():
    """Main"""
    pygame.init()

    PYGAME_SCR_SIZE = (800, 600)
    screen = pygame.display.set_mode(PYGAME_SCR_SIZE)
    pygame.display.set_caption("triangle test")
    clock = pygame.time.Clock()

    pygame.mouse.set_cursor(*pygame.cursors.broken_x)

    font = pygame.font.Font(None, 30)
    TEXT_COLOR = (200, 200, 230)

    done = False
    offscreen = pygame.Surface(RASTER_SCR_SIZE)
    px_array = pygame.PixelArray(offscreen)

    x0 = 30
    y0 = 30
    x1 = 60
    y1 = 30
    x2 = 45
    y2 = 45

    while not done:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                # WASD: p0
                if event.key == pygame.K_w:
                    y0 -= 1
                elif event.key == pygame.K_s:
                    y0 += 1
                elif event.key == pygame.K_a:
                    x0 -= 1
                elif event.key == pygame.K_d:
                    x0 += 1
                # TFGH: p1
                elif event.key == pygame.K_t:
                    y1 -= 1
                elif event.key == pygame.K_g:
                    y1 += 1
                elif event.key == pygame.K_f:
                    x1 -= 1
                elif event.key == pygame.K_h:
                    x1 += 1

        offscreen.fill(RGB_BLACK)
        for x,y in drawing.get_triangle_2d_points(x0, y0, x1, y1, x2, y2):
            px_array[x, y] = (255, 255, 255)

        screen.blit(pygame.transform.scale(offscreen, PYGAME_SCR_SIZE), (0,0))
        pygame.display.flip()

if __name__ == '__main__':
    main_function()
