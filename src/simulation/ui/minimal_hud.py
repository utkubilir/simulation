"""
Minimalist HUD - Sade ve temiz arayüz
"""

import pygame
import math
from typing import Dict, Optional


class MinimalHUD:
    """
    Minimalist Head-Up Display.
    
    Sadece kritik bilgiler:
    - Hız, irtifa, heading
    - Kilit durumu
    - Puan
    """
    
    # Renkler
    COLORS = {
        'text': (255, 255, 255),
        'text_dim': (150, 150, 150),
        'primary': (0, 200, 255),
        'danger': (255, 50, 50),
        'success': (50, 255, 100),
        'warning': (255, 200, 50),
        'bg_overlay': (0, 0, 0, 120),
    }
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fonts = {}
        self._init_fonts()
        
        # Lock animation
        self.lock_flash_timer = 0
        self.successful_locks = 0
        self.last_lock_time = 0
        
    def _init_fonts(self):
        pygame.font.init()
        self.fonts['large'] = pygame.font.SysFont("Consolas", 32, bold=True)
        self.fonts['medium'] = pygame.font.SysFont("Consolas", 24)
        self.fonts['small'] = pygame.font.SysFont("Consolas", 16)
        
    def update(self, uav_state: Dict, world_state: Dict, lock_state: Dict):
        """Update HUD state."""
        # Track successful locks
        if lock_state and lock_state.get('is_valid') and lock_state.get('is_locked'):
            current_time = world_state.get('time', 0)
            if current_time - self.last_lock_time > 1.0:  # New lock
                self.successful_locks += 1
                self.lock_flash_timer = 30  # Flash frames
                self.last_lock_time = current_time
                
        if self.lock_flash_timer > 0:
            self.lock_flash_timer -= 1
        
    def render(self, surface: pygame.Surface, uav_state: Dict, world_state: Dict, lock_state: Dict):
        """Render minimalist HUD."""
        
        # === Center Crosshair ===
        self._render_crosshair(surface, lock_state)
        
        # === Bottom Left: Flight Data ===
        self._render_flight_data(surface, uav_state)
        
        # === Top Right: Score ===
        self._render_score(surface, lock_state)
        
        # === Lock Success Flash ===
        if self.lock_flash_timer > 0:
            self._render_lock_success(surface)
            
    def _render_crosshair(self, surface: pygame.Surface, lock_state: Dict):
        """Minimal crosshair with lock progress."""
        cx, cy = self.width // 2, self.height // 2
        
        # Determine color based on lock state
        if lock_state and lock_state.get('is_valid'):
            color = self.COLORS['success']
        elif lock_state and lock_state.get('is_locked'):
            color = self.COLORS['warning']
        else:
            color = self.COLORS['primary']
            
        # Simple crosshair lines
        size = 25
        gap = 8
        
        # Horizontal
        pygame.draw.line(surface, color, (cx - size, cy), (cx - gap, cy), 2)
        pygame.draw.line(surface, color, (cx + gap, cy), (cx + size, cy), 2)
        # Vertical
        pygame.draw.line(surface, color, (cx, cy - size), (cx, cy - gap), 2)
        pygame.draw.line(surface, color, (cx, cy + gap), (cx, cy + size), 2)
        
        # Lock progress arc
        if lock_state and lock_state.get('is_locked'):
            progress = lock_state.get('progress', 0)
            if progress > 0:
                rect = pygame.Rect(cx - 35, cy - 35, 70, 70)
                end_angle = progress * 2 * math.pi
                pygame.draw.arc(surface, color, rect, -math.pi/2, -math.pi/2 + end_angle, 3)
                
    def _render_flight_data(self, surface: pygame.Surface, uav_state: Dict):
        """Compact flight data display."""
        if not uav_state:
            return
            
        x, y = 20, self.height - 90
        
        # Background
        bg = pygame.Surface((160, 75), pygame.SRCALPHA)
        bg.fill(self.COLORS['bg_overlay'])
        surface.blit(bg, (x - 10, y - 5))
        
        speed = uav_state.get('speed', 0)
        altitude = uav_state.get('altitude', 0)
        heading = uav_state.get('heading', 0)
        
        # Speed
        spd_txt = self.fonts['medium'].render(f"{speed:.0f} m/s", True, self.COLORS['text'])
        surface.blit(spd_txt, (x, y))
        
        # Altitude
        alt_txt = self.fonts['medium'].render(f"{altitude:.0f} m", True, self.COLORS['text'])
        surface.blit(alt_txt, (x, y + 25))
        
        # Heading
        hdg_txt = self.fonts['small'].render(f"HDG {heading:.0f}°", True, self.COLORS['text_dim'])
        surface.blit(hdg_txt, (x, y + 52))
        
    def _render_score(self, surface: pygame.Surface, lock_state: Dict):
        """Score and lock count in top right."""
        x = self.width - 20
        y = 20
        
        # Background
        bg = pygame.Surface((140, 60), pygame.SRCALPHA)
        bg.fill(self.COLORS['bg_overlay'])
        surface.blit(bg, (x - 130, y - 5))
        
        # Score
        score = lock_state.get('score', 0) if lock_state else 0
        score_txt = self.fonts['large'].render(f"{score}", True, self.COLORS['success'])
        surface.blit(score_txt, (x - score_txt.get_width(), y))
        
        # Lock count
        lock_txt = self.fonts['small'].render(f"LOCKS: {self.successful_locks}", True, self.COLORS['text_dim'])
        surface.blit(lock_txt, (x - lock_txt.get_width(), y + 35))
        
    def _render_lock_success(self, surface: pygame.Surface):
        """Flash effect on successful lock."""
        alpha = int(255 * (self.lock_flash_timer / 30))
        
        # Green flash overlay
        flash = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        flash.fill((50, 255, 100, alpha // 4))
        surface.blit(flash, (0, 0))
        
        # "LOCKED" text
        if self.lock_flash_timer > 15:
            txt = self.fonts['large'].render("✓ LOCKED", True, self.COLORS['success'])
            x = self.width // 2 - txt.get_width() // 2
            y = self.height // 2 - 80
            surface.blit(txt, (x, y))
            
    def reset(self):
        """Reset HUD state."""
        self.successful_locks = 0
        self.lock_flash_timer = 0
        self.last_lock_time = 0
