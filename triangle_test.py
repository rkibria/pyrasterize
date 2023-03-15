import pygame
import pygame.gfxdraw
import pygame.mouse
import pygame.cursors

from pyrasterize import vecmat
from pyrasterize import drawing

RASTER_SCR_SIZE = RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT = 1200, 1000 # 160, 120
RASTER_SCR_AREA = (0, 0, RASTER_SCR_WIDTH, RASTER_SCR_HEIGHT)
RGB_BLACK = (0, 0, 0)
PYGAME_SCR_SIZE = (1200, 1000)

def main_function():
    """Main"""
    pygame.init()

    screen = pygame.display.set_mode(PYGAME_SCR_SIZE)
    pygame.display.set_caption("triangle test")
    clock = pygame.time.Clock()

    pygame.mouse.set_cursor(*pygame.cursors.broken_x)

    font = pygame.font.Font(None, 30)
    TEXT_COLOR = (200, 200, 230)

    done = False
    offscreen = pygame.Surface(RASTER_SCR_SIZE)
    px_array = pygame.PixelArray(offscreen)

    # x0 = 30
    # y0 = 30
    # x1 = 30
    # y1 = 60
    # x2 = 60
    # y2 = 30
    # x3 = 60
    # y3 = 60

    x0 = 867
    y0 = 837
    x1 = 988
    y1 = 688
    x2 = 858
    y2 = 699
    x3 = 1005
    y3 = 824

    frame = 0
    while not done:
        frame += 1
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

        offscreen.fill((255, 0, 0))
        cols = [(255, 0, 0), (0, 128, 0), (255, 255, 0), (0, 255, 255)]
        # if int(frame / 10) % 2 == 0:
        #     cols = [(255, 0, 0), (0, 128, 0), (255, 255, 0), (0, 255, 255)]
        # else:
        #     cols = [(255, 255, 0), (0, 255, 255), (255, 0, 0), (0, 128, 0),]
        t = 0
        # print("------")
        for x,y in drawing.triangle(x0, y0, x1, y1, x2, y2):
            f = cols[0] if t else cols[1]
            px_array[x, y] = f
            # print(x,y)
        for x,y in drawing.triangle(x0, y0, x3, y3, x1, y1):
            f = cols[2] if t else cols[3]
            px_array[x, y] = f

        # px_array[x0, y0] = (0, 0, 255)
        # px_array[x1, y1] = (0, 0, 255)
        # px_array[x2, y2] = (0, 0, 255)

        # px_array[x3, y3] = (0, 0, 255)

        # if int(frame / 10) % 2 == 0:
        #     for x,y in drawing.bresenham(x0, y0, x1, y1):
        #         px_array[x, y] = (255, 0, 255)
        #     for x,y in drawing.bresenham(x0, y0, x2, y2):
        #         px_array[x, y] = (255, 0, 255)
        #     for x,y in drawing.bresenham(x1, y1, x2, y2):
        #         px_array[x, y] = (255, 0, 255)

        screen.blit(pygame.transform.scale(offscreen, PYGAME_SCR_SIZE), (0,0))
        pygame.display.flip()

if __name__ == '__main__':
    main_function()
