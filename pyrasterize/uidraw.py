"""
"""

import pygame as pg

import xml.etree.ElementTree as ET

def load_texture_atlas(path):
    """
    Return (sheet path, atlas)
    """
    tree = ET.parse(path)
    root = tree.getroot()
    sheet_path = root.attrib["imagePath"]
    atlas = {}
    for child in root:
        entry = dict(child.attrib)
        name = entry["name"]
        del entry["name"]
        atlas[name] = entry
    return (sheet_path, atlas)

def atlas_to_ui_images(sheet_path, atlas):
    """
    Return ui images dict
    'element_name': {'x': 190, 'y': 0, 'width': 100, 'height': 100, 'img': <Surface(100x100x32 SW)>}
    """
    sprite_sheet = pg.image.load(sheet_path).convert_alpha()
    ui_images = {}
    for name,entry in atlas.items():
        if name.endswith(".png"):
            name = name[:-4]
        x = int(entry["x"])
        y = int(entry["y"])
        width = int(entry["width"])
        height = int(entry["height"])
        img = pg.Surface([width, height], pg.SRCALPHA)
        img.blit(sprite_sheet, (0,0), (x, y, width, height))
        ui_images[name] = {"x": x, "y": y, "width": width, "height": height, "img": img}
    return ui_images

def draw_horizontal_bar(surface : pg.Surface, ui_images, style, progress, x, y, w, h):
    """
    Draw a horizontal progress bar
    """
    left = ui_images[style + "_horizontalLeft"]
    left_width = left["width"]
    mid = ui_images[style + "_horizontalMid"]
    right = ui_images[style + "_horizontalRight"]
    right_width = right["width"]

    extent_pixels = int(w * progress)

    left_img = pg.transform.scale(left["img"], (left_width, h))
    surface.blit(left_img, (x, y), (0, 0, min(left_width, extent_pixels), h))
    if extent_pixels <= left_width:
        return

    mid_extent = extent_pixels - left_width
    mid_total = w - left_width - right_width
    if mid_extent > mid_total:
        mid_extent -= mid_extent - mid_total
    mid_img = pg.transform.scale(mid["img"], (mid_extent, h))
    surface.blit(mid_img, (x + left_width, y))

    right_start = left_width + mid_total
    if extent_pixels <= right_start:
        return

    right_extent = extent_pixels - right_start
    right_img = pg.transform.scale(right["img"], (right_width, h))
    surface.blit(right_img, (x + right_start, y), (0, 0, right_extent, h))

def draw_window(surface, ui_images, style, x, y, w, h):
    """
    Draw a window
    """
    corner_ul = ui_images[style + "_window_corner_ul"]
    corner_ur = ui_images[style + "_window_corner_ur"]
    corner_ll = ui_images[style + "_window_corner_ll"]
    corner_lr = ui_images[style + "_window_corner_lr"]

    surface.blit(corner_ul["img"], (x, y))
    surface.blit(corner_ur["img"], (x + w - corner_ur["width"], y))
    surface.blit(corner_ll["img"], (x, y + h - corner_ll["height"]))
    surface.blit(corner_lr["img"], (x + w - corner_ur["width"], y + h - corner_ll["height"]))

    bar_top = ui_images[style + "_window_bar_top"]
    bar_bottom = ui_images[style + "_window_bar_bottom"]
    bar_left = ui_images[style + "_window_bar_left"]
    bar_right = ui_images[style + "_window_bar_right"]

    surface.blit(pg.transform.scale(bar_top["img"], (w - corner_ul["width"] - corner_ur["width"], bar_top["height"])),
                 (x + corner_ul["width"], y))
    surface.blit(pg.transform.scale(bar_bottom["img"], (w - corner_ll["width"] - corner_lr["width"], bar_bottom["height"])),
                 (x + corner_ll["width"], y + h - corner_ll["height"]))
    surface.blit(pg.transform.scale(bar_left["img"], (bar_left["width"], h - corner_ul["height"] - corner_ll["height"])),
                 (x, y + corner_ul["height"]))
    surface.blit(pg.transform.scale(bar_right["img"], (bar_right["width"], h - corner_ur["height"] - corner_lr["height"])),
                 (x + w - corner_ur["width"], y + corner_ur["height"]))

    window_fill = ui_images[style + "_window_fill"]
    surface.blit(pg.transform.scale(window_fill["img"], (w - corner_ul["width"] - corner_ur["width"], h - corner_ul["height"] - corner_lr["height"])),
                 (x + corner_ul["width"], y + corner_ul["height"]))

def wrap_text_iter(text, size, font_height, get_text_width_func):
    """
    Yields (text, (x,y)) pairs to draw
    """
    y = 0
    max_width,max_height = size
    for line in text.splitlines():
        if y >= max_height:
            break
        i = 1
        while i < len(line):
            width = get_text_width_func(line[:i])
            if width > max_width:
                prev_i = i
                i = line.rfind(" ", 0, i)
                if i == -1:
                    yield (line[:prev_i], (0, y))
                    line = line[prev_i:]
                    i = 1
                else:
                    yield (line[:i], (0, y))
                    i += 1
                    line = line[i:]
                    i = 1
                y += font_height
                continue
            i += 1
        yield (line, (0, y))
        y += font_height
    return
    yield

def draw_text(surface, text, color, rect, font, aa=False,line_spacing=-2):
    """
    """
    font_height = font.size("Tg")[1]
    for line,pos in wrap_text_iter(text, rect.size, font_height, lambda line: font.size(line)[0]):
        image = font.render(line, aa, color)
        surface.blit(image, pos)
