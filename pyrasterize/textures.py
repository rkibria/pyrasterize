import pygame

def get_mip_textures(file_name) -> list:
    """
    Return texture data to assign to model['texture']

    List elements scanlines of color tuples [[line y=0 (r,g,b), ...], [ line y=1 (r,g,b), ...], ...]
    """
    img = pygame.image.load(file_name).convert_alpha()
    tex_data = [] # mipmap levels, 0 = original, 1 = original/2
    mip_level = 0
    while True:
        mip_scale = 2 ** mip_level
        mip_size = (img.get_width() // mip_scale, img.get_height() // mip_scale)
        mip_surface = pygame.Surface(mip_size)
        mip_surface.blit(pygame.transform.scale(img, mip_size), (0,0))
        mip_tex = []
        for y in range(mip_surface.get_height()):
            row = []
            for x in range(mip_surface.get_width()):
                rgb = mip_surface.get_at((x, y))[:3]
                row.append(rgb)
            mip_tex.append(row)
        mip_tex.reverse()
        tex_data.append(mip_tex)
        if mip_size[0] <= 1 or mip_size[1] <= 1:
            break
        mip_level += 1
    return tex_data
