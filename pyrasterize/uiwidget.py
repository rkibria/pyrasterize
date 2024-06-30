import pygame as pg
from pygame import Vector2

from . import uidraw

class Widget:
    """
    """
    def __init__(self, name, pos=Vector2(), size=Vector2()):
        """
        """
        self.name = name
        self.pos = Vector2(pos)
        self.size = Vector2(size)
        self.rect = pg.Rect(pos, size)
        self.children = []

        # Special effects
        self.cooldown_counter = [0, 0] # total, current frames
        self.cooldown_overlay = None

        # Events
        self.on_mouse_button_down_cb = None # (widget, wnd_mgr, event)

    def __str__(self):
        """
        """
        return f"widget({self.name}, {self.pos}, {self.size})"

    def draw(self, wnd_mgr, surface, dest=Vector2()):
        """
        """
        for child in self.children:
            child.draw(wnd_mgr, surface, dest + self.pos)
        if self.cooldown_counter[0]:
            if self.cooldown_counter[1] < self.cooldown_counter[0]:
                percent = self.cooldown_counter[1] / self.cooldown_counter[0]
                height = int((1 - percent) * self.size.y)
                overlay_resized = pg.transform.scale(self.cooldown_overlay, (self.size.x, height))
                alpha = max(100, int((1 - percent) * 255))
                overlay_resized.set_alpha(alpha)
                d = (0, int(self.size.y + 1 - height))
                surface.blit(overlay_resized, dest + self.pos + d)
                self.cooldown_counter[1] += 1
            else:
                self.cooldown_counter[0] = 0
                self.cooldown_counter[1] = 0
                self.cooldown_overlay = None

    def add_child(self, child):
        """
        """
        self.children.append(child)

    def on_mouse_button_down(self, wnd_mgr, pos : pg.Vector2):
        rel_pos = pos - self.pos
        if self.rect.collidepoint(rel_pos):
            if callable(self.on_mouse_button_down_cb):
                done = self.on_mouse_button_down_cb(self, wnd_mgr, rel_pos)
                if done:
                    return True
        for child in self.children:
            done = child.on_mouse_button_down(wnd_mgr, rel_pos)
            if done:
                return True
        return False

    def start_cooldown(self, num_frames, color):
        """
        Show a cooldown indicator on the area of the widget.
        """
        self.cooldown_counter[0] = num_frames
        self.cooldown_counter[1] = 0
        self.cooldown_overlay = pg.Surface(self.size)
        self.cooldown_overlay.fill(color)

    def get_cooldown(self):
        """
        Return time to end of cooldown (0 means ready)
        """
        if self.cooldown_counter[0]:
            return self.cooldown_counter[0] - self.cooldown_counter[1]
        else:
            return 0

class Image(Widget):
    """
    """
    def __init__(self, name, img, pos=Vector2(), size=Vector2()):
        super().__init__(name, pos, size)
        self.img = img

    def __str__(self):
        """
        """
        return f"image({self.name}, {self.pos}, {self.size})"

    def draw(self, wnd_mgr, surface, dest=Vector2()):
        """
        """
        surface.blit(pg.transform.scale(self.img, self.size), dest + self.pos)
        super().draw(wnd_mgr, surface, dest)


class Label(Widget):
    """
    """
    def __init__(self, name, text,
                 font_size, font_color=(0,0,0), font_name=None, antialias=True, under_color=None,
                 pos=Vector2(), size=Vector2()):
        super().__init__(name, pos, size)
        self.set_text(text, font_size, font_color, under_color, font_name, antialias)

    def set_text(self, text, font_size, font_color=(0,0,0), under_color=None, font_name=None, antialias=True):
        font = pg.font.Font(font_name, font_size)
        if under_color is not None:
            self.under_img = font.render(text, antialias, under_color)
        else:
            self.under_img = None
        self.img = font.render(text, antialias, font_color)
        self.size = self.img.get_size()

    def __str__(self):
        """
        """
        return f"label({self.name}, {self.pos}, {self.size})"

    def draw(self, wnd_mgr, surface, dest=Vector2()):
        """
        """
        if self.under_img is not None:
            surface.blit(pg.transform.scale(self.under_img, self.size), dest + self.pos + pg.Vector2(1, 1))
        surface.blit(pg.transform.scale(self.img, self.size), dest + self.pos)
        super().draw(wnd_mgr, surface, dest)


class TextBox(Widget):
    """
    """
    def __init__(self, name, text, font_size, font_color=(0,0,0), font_name=None, aa=False, pos=Vector2(), size=Vector2()):
        """
        """
        super().__init__(name, pos, size)
        self.set_text(text, font_size, font_color, font_name, aa)

    def set_text(self, text, font_size, font_color=(0,0,0), font_name=None, aa=False):
        """
        """
        self.font_size = font_size
        self.font_color = font_color
        self.font_name = font_name
        self.aa = aa
        self.update_text(text)

    def update_text(self, text):
        """
        """
        font = pg.font.Font(self.font_name, self.font_size)
        self.img = pg.Surface(self.size).convert_alpha()
        self.img.fill((0,0,0,0))
        uidraw.draw_text(self.img, text, self.font_color, pg.Rect((0, 0), self.size), font, self.aa)

    def __str__(self):
        """
        """
        return f"label({self.name}, {self.pos}, {self.size})"

    def draw(self, wnd_mgr, surface, dest=Vector2()):
        """
        """
        surface.blit(self.img, dest + self.pos)
        super().draw(wnd_mgr, surface, dest)


class Window(Widget):
    """
    """
    def __init__(self, name, wnd_mgr, style, pos=Vector2(), size=Vector2()):
        """
        """
        super().__init__(name, pos, size)
        self.wnd_mgr = wnd_mgr
        self.style = style

    def __str__(self):
        """
        """
        return f"window({self.name}, {self.pos}, {self.size})"

    def draw(self, wnd_mgr, surface, dest=Vector2()):
        """
        """
        uidraw.draw_window(surface, self.wnd_mgr.ui_images, self.style, dest.x + self.pos.x, dest.y + self.pos.y, self.size.x, self.size.y)
        super().draw(wnd_mgr, surface, dest)


class HorizontalBar(Widget):
    """
    Horizontal progress bar. Set .progress to value between 0 and 1
    """
    def __init__(self, name, wnd_mgr, style, pos=Vector2(), size=Vector2()):
        """
        """
        super().__init__(name, pos, size)
        self.wnd_mgr = wnd_mgr
        self.style = style
        self.progress = 0.0

    def __str__(self):
        """
        """
        return f"horizontal_bar({self.name}, {self.pos}, {self.size})"

    def draw(self, wnd_mgr, surface, dest=Vector2()):
        """
        """
        uidraw.draw_horizontal_bar(surface, self.wnd_mgr.ui_images, self.style,
                                   self.progress,
                                   dest.x + self.pos.x, dest.y + self.pos.y,
                                   self.size.x, self.size.y)
        super().draw(wnd_mgr, surface, dest)

class HorizontalSlider(Widget):
    """
    Horizontal progress bar. Set .progress to value between 0 and 1
    """
    def __init__(self, name, wnd_mgr, bar_style, circle_style, pos=Vector2(), size=Vector2()):
        """
        """
        super().__init__(name, pos, size)
        self.wnd_mgr = wnd_mgr
        self.bar_style = bar_style
        self.circle_style = circle_style
        self.progress = 0.0
        self.on_change_cb = None # Signature (progress)

    def __str__(self):
        """
        """
        return f"horizontal_slider({self.name}, {self.pos}, {self.size})"

    def draw(self, wnd_mgr, surface, dest=Vector2()):
        """
        """
        uidraw.draw_horizontal_bar(surface, self.wnd_mgr.ui_images, self.bar_style,
                                   1.0,
                                   dest.x + self.pos.x, dest.y + self.pos.y,
                                   self.size.x, self.size.y)
        circle = self.wnd_mgr.ui_images["iconCircle_" + self.circle_style]
        circle_size = self.size.y
        surface.blit(pg.transform.scale(circle["img"],
                                        (circle_size, circle_size)),
                     (dest.x + self.pos.x + int(self.size.x * self.progress - circle_size / 2),
                      dest.y + self.pos.y))
        super().draw(wnd_mgr, surface, dest)

    def on_mouse_button_down(self, wnd_mgr, pos : pg.Vector2):
        if self.rect.collidepoint(pos):
            dx = pos.x - self.pos.x
            if dx >= 0 and dx < self.size.x:
                self.progress = dx / self.size.x
                if callable(self.on_change_cb):
                    self.on_change_cb(self.progress)
                return True
        return False
