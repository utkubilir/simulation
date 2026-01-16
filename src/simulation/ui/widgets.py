
import pygame
import math
from typing import Tuple, Optional
from .theme import Colors, Fonts

class Widget:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        
    def render(self, surface: pygame.Surface):
        pass

class Panel(Widget):
    """Semi-transparent glass panel with border."""
    def __init__(self, x: int, y: int, width: int, height: int, title: Optional[str] = None, font: Optional[pygame.font.Font] = None):
        super().__init__(x, y, width, height)
        self.title = title
        self.font = font
        
    def render(self, surface: pygame.Surface):
        # Draw semi-transparent background
        s = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        s.fill((*Colors.PANEL_BG, 180)) # Alpha 180
        surface.blit(s, (self.rect.x, self.rect.y))
        
        # Draw border
        pygame.draw.rect(surface, Colors.PRIMARY_DIM, self.rect, 1)
        
        # Draw corner accents (Sci-fi look)
        corner_len = 15
        # Top-Left
        pygame.draw.line(surface, Colors.PRIMARY, (self.rect.left, self.rect.top), (self.rect.left + corner_len, self.rect.top), 2)
        pygame.draw.line(surface, Colors.PRIMARY, (self.rect.left, self.rect.top), (self.rect.left, self.rect.top + corner_len), 2)
        # Bottom-Right
        pygame.draw.line(surface, Colors.PRIMARY, (self.rect.right-1, self.rect.bottom-1), (self.rect.right - corner_len, self.rect.bottom-1), 2)
        pygame.draw.line(surface, Colors.PRIMARY, (self.rect.right-1, self.rect.bottom-1), (self.rect.right-1, self.rect.bottom - corner_len), 2)

        # Title
        if self.title and self.font:
            text = self.font.render(self.title, True, Colors.PRIMARY)
            surface.blit(text, (self.rect.x + 10, self.rect.y + 5))

class VerticalTape(Widget):
    """Vertical scrolling tape for Altitude/Speed."""
    def __init__(self, x: int, y: int, width: int, height: int, min_val: float, max_val: float, current_val: float, step: int = 10, label: str = ""):
        super().__init__(x, y, width, height)
        self.min = min_val
        self.max = max_val
        self.val = current_val
        self.step = step
        self.label = label
        
    def update(self, val: float):
        self.val = val
        
    def render(self, surface: pygame.Surface, font: pygame.font.Font):
        # Background
        s = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        s.fill((*Colors.PANEL_BG, 100))
        surface.blit(s, self.rect)
        pygame.draw.rect(surface, Colors.PRIMARY_DIM, self.rect, 1)
        
        # Center line
        center_y = self.rect.centery
        pygame.draw.line(surface, Colors.PRIMARY, (self.rect.left, center_y), (self.rect.right, center_y), 2)
        
        # Label (Top)
        lbl_surf = font.render(self.label, True, Colors.PRIMARY)
        surface.blit(lbl_surf, (self.rect.centerx - lbl_surf.get_width()//2, self.rect.top - 20))
        
        # Value readout (Center box)
        val_str = f"{int(self.val)}"
        val_surf = font.render(val_str, True, Colors.TEXT_MAIN)
        
        # Ticks
        # Pixel per unit
        px_per_unit = 20 # 10 units = 20 pixels spacing? Let's say 40px = step
        pixels_per_step = 40
        pixels_per_unit = pixels_per_step / self.step
        
        # Determine range visible
        # We need to draw ticks relative to center_y
        
        offset_val = self.val % self.step
        offset_px = offset_val * pixels_per_unit
        
        # Find nearest step below
        start_step = (int(self.val) // self.step) * self.step
        
        # Draw steps up and down
        num_ticks = (self.rect.height // 2) // pixels_per_step + 2
        
        for i in range(-num_ticks, num_ticks + 1):
            tick_val = start_step + (i * self.step)
            y_pos = center_y - (tick_val - self.val) * pixels_per_unit
            
            if self.rect.top <= y_pos <= self.rect.bottom:
                # Major tick
                pygame.draw.line(surface, Colors.TEXT_DIM, (self.rect.right - 10, y_pos), (self.rect.right, y_pos), 2)
                
                # Text
                t = font.render(str(tick_val), True, Colors.TEXT_DIM)
                surface.blit(t, (self.rect.right - 15 - t.get_width(), y_pos - t.get_height()//2))

class ProgressBar(Widget):
    """Horizontal progress bar."""
    def __init__(self, x: int, y: int, width: int, height: int, label: str, color: Tuple[int,int,int]):
        super().__init__(x, y, width, height)
        self.label = label
        self.color = color
        self.value = 1.0 # 0.0 to 1.0
        
    def render(self, surface: pygame.Surface, font: pygame.font.Font):
        # Label
        lbl = font.render(self.label, True, Colors.TEXT_MAIN)
        surface.blit(lbl, (self.rect.x, self.rect.y - 15))
        
        # Background bar
        pygame.draw.rect(surface, (50, 50, 50), self.rect)
        
        # Fill bar
        fill_width = int(self.rect.width * self.value)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
        pygame.draw.rect(surface, self.color, fill_rect)
        
        # Border
        pygame.draw.rect(surface, Colors.TEXT_DIM, self.rect, 1)

class Radar(Widget):
    """Circular Radar."""
    def __init__(self, x: int, y: int, radius: int):
        super().__init__(x, y, radius*2, radius*2)
        self.radius = radius
        self.center = (x + radius, y + radius)
        
    def render(self, surface: pygame.Surface, targets: list, player_heading: float, fov: float):
        # Background
        pygame.draw.circle(surface, (*Colors.PANEL_BG, 150), self.center, self.radius)
        pygame.draw.circle(surface, Colors.PRIMARY_DIM, self.center, self.radius, 1)
        
        # Range rings
        pygame.draw.circle(surface, Colors.PRIMARY_DIM, self.center, int(self.radius * 0.5), 1)
        pygame.draw.circle(surface, Colors.PRIMARY_DIM, self.center, int(self.radius * 0.25), 1)
        
        # Cross lines
        pygame.draw.line(surface, Colors.PRIMARY_DIM, (self.center[0] - self.radius, self.center[1]), (self.center[0] + self.radius, self.center[1]), 1)
        pygame.draw.line(surface, Colors.PRIMARY_DIM, (self.center[0], self.center[1] - self.radius), (self.center[0], self.center[1] + self.radius), 1)
        
        # Player (Center)
        # Draw small triangle
        # TODO: Rotate triangle based on heading if North-Up, but usually Radar is Heading-Up
        # Let's assume Heading-Up (Player facing UP)
        pygame.draw.polygon(surface, Colors.SUCCESS, [
            (self.center[0], self.center[1] - 5),
            (self.center[0] - 4, self.center[1] + 4),
            (self.center[0] + 4, self.center[1] + 4)
        ])
        
        # FOV Lines (V shape) up
        # FOV is total angle (e.g. 60 deg)
        half_fov = math.radians(fov / 2)
        len_fov = self.radius
        
        # Left line
        lx = self.center[0] + len_fov * math.sin(-half_fov)
        ly = self.center[1] - len_fov * math.cos(-half_fov)
        pygame.draw.line(surface, (*Colors.PRIMARY, 100), self.center, (lx, ly), 1)
        
        # Right line
        rx = self.center[0] + len_fov * math.sin(half_fov)
        ry = self.center[1] - len_fov * math.cos(half_fov)
        pygame.draw.line(surface, (*Colors.PRIMARY, 100), self.center, (rx, ry), 1)
        
        # Targets
        # They need to be transformed from World -> Body (relative) -> Radar Screen
        # This logic usually happens outside, but let's assume `targets` is list of (rel_x, rel_y, type)
        # rel_y is forward distance, rel_x is side distance
        
        # We need max range scaling. Say 1000m = Radius
        MAX_RANGE = 1000.0
        scale = self.radius / MAX_RANGE
        
        for t in targets:
            # t: {'rel_pos': [x, y], 'type': 'enemy', 'locked': bool}
            rx, ry = t['rel_pos']
            
            # Rotate if needed. If input is relative to body (X=Forward, Y=Right in Body frame? Or X=East? )
            # Usually: body_x = forward, body_y = right.
            # Screen y is Up (-y), Screen x is Right (+x).
            # So forward (rx) -> screen -y
            # right (ry) -> screen +x
            
            sx = self.center[0] + ry * scale 
            sy = self.center[1] - rx * scale
            
            # Clip to radar
            dist = math.sqrt((sx - self.center[0])**2 + (sy - self.center[1])**2)
            if dist > self.radius:
                # Clamp to edge
                angle = math.atan2(sy - self.center[1], sx - self.center[0])
                sx = self.center[0] + self.radius * math.cos(angle)
                sy = self.center[1] + self.radius * math.sin(angle)
            
            col = Colors.DANGER if t.get('type') == 'enemy' else Colors.SUCCESS
            if t.get('locked'):
                col = Colors.DANGER
                pygame.draw.circle(surface, (255, 255, 255), (int(sx), int(sy)), 4, 1) # Halo
                
            pygame.draw.circle(surface, col, (int(sx), int(sy)), 3)

