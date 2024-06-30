import pygame as pg

import os

from . import uidraw

class WindowManager:
    def __init__(self, screen_size):
        """
        """
        self.screen_size = screen_size
        atlas_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "uipack_rpg_sheet.xml")
        sheet_path,atlas = uidraw.load_texture_atlas(atlas_xml_path)
        sheet_path = os.path.join(os.path.dirname(atlas_xml_path), sheet_path)
        self.ui_images = uidraw.atlas_to_ui_images(sheet_path, atlas)

        # Sorted from front to back (i.e. reverse draw order)
        self.widgets = []

        self.cursor_enable = True
        self.cursor_pos = pg.Vector2(0, 0)
        self.cursor_size = pg.Vector2(10, 10)
        self.cursor_width = 2
        self.rgb_cursor = (150, 150, 255, 255)
        self.cursor_surface = pg.Surface((2 * self.cursor_size.x, 2 * self.cursor_size.y))
        pg.draw.rect(self.cursor_surface, self.rgb_cursor, (self.cursor_size.x - self.cursor_width, 0, self.cursor_width * 2, self.cursor_size.y * 2))
        pg.draw.rect(self.cursor_surface, self.rgb_cursor, (0, self.cursor_size.y - self.cursor_width, self.cursor_size.y * 2, self.cursor_width * 2))
        pg.draw.rect(self.cursor_surface, (0, 0, 0),
                     (self.cursor_size.x - 2 * self.cursor_width,
                      self.cursor_size.y - 2 * self.cursor_width, self.cursor_width * 4, self.cursor_width * 4))

    def clear_widgets(self):
        self.widgets.clear()

    def add_widget(self, widget, to_back=True):
        """
        """
        if to_back:
            self.widgets.append(widget)
        else:
            self.widgets.insert(0, widget)

    def set_cursor_pos(self, pos):
        self.cursor_pos = pos

    def on_mouse_button_down(self, event):
        pos = pg.Vector2(event.pos)
        for widget in self.widgets:
            done = widget.on_mouse_button_down(self, pos)
            if done:
                break

    def draw_cursor(self, surface):
        surface.blit(self.cursor_surface, self.cursor_pos - self.cursor_size, special_flags=pg.BLEND_RGBA_ADD)

    def draw(self, surface, dest=pg.Vector2(0,0)):
        """
        """
        for widget in self.widgets:
            widget.draw(self, surface, dest)

        if self.cursor_enable:
            self.draw_cursor(surface)
