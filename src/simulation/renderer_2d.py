"""
Renderer2D - Top-down 2D Map View for vNext

Features:
- Arena with boundaries and safe zone
- UAV icons with heading arrows
- Zoomable and pannable
- HUD overlay with telemetry
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional
from .ui.hud import HUDOverlay
from .ui.ui_mode import UIMode


class Renderer2D:
    """
    Top-down 2D map view renderer.
    
    Shows:
    - Arena boundaries
    - Player UAV (blue) with heading arrow
    - Enemy UAVs (red) with heading arrows and IDs
    - Optional detection radius rings
    - HUD with telemetry
    """
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_SAFE_ZONE = (30, 40, 35)
    COLOR_PLAYER = (80, 180, 255)
    COLOR_ENEMY = (255, 100, 100)
    COLOR_LOCKED = (255, 200, 50)
    COLOR_TEXT = (200, 200, 200)
    COLOR_HUD_BG = (30, 35, 40, 200)
    
    def __init__(self, width: int = 1280, height: int = 720, radar_heading_mode: str = "heading_up"):
        self.width = width
        self.height = height
        self.radar_heading_mode = radar_heading_mode
        
        # Map viewport
        self.map_rect = pygame.Rect(0, 0, width - 300, height)  # Leave room for camera inset
        
        # World bounds
        self.world_size = (2000, 2000)
        self.safe_zone = (200, 200, 1800, 1800)
        
        # View controls
        self.camera_x = 1000
        self.camera_y = 1000
        self.zoom = 0.5  # pixels per meter
        self.min_zoom = 0.1
        self.max_zoom = 2.0
        
        # State
        self.screen = None
        self.font = None
        self.font_small = None
        self.clock = None
        
        # New HUD Overlay
        self.hud_overlay = None
        
        # Detection visualization
        self.show_detection_radius = False
        self.detection_radius = 500
        
    def init(self):
        """Initialize pygame"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("TEKNOFEST Savaşan İHA Sim vNext")
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 36)  # Large font for headers
        self.clock = pygame.time.Clock()
        
        # Init HUD Overlay
        self.hud_overlay = HUDOverlay(self.width, self.height, radar_heading_mode=self.radar_heading_mode)
        
    def close(self):
        """Cleanup pygame"""
        pygame.quit()
        
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        # Center on camera position
        rel_x = world_x - self.camera_x
        rel_y = world_y - self.camera_y
        
        # Apply zoom
        screen_x = self.map_rect.centerx + rel_x * self.zoom
        screen_y = self.map_rect.centery - rel_y * self.zoom  # Y inverted
        
        return int(screen_x), int(screen_y)
        
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        rel_x = (screen_x - self.map_rect.centerx) / self.zoom
        rel_y = -(screen_y - self.map_rect.centery) / self.zoom
        
        world_x = self.camera_x + rel_x
        world_y = self.camera_y + rel_y
        
        return world_x, world_y
        
    def set_camera(self, x: float, y: float):
        """Set camera center position"""
        self.camera_x = x
        self.camera_y = y
        
    def handle_events(self) -> List[Dict]:
        """Handle pygame events"""
        events = []
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events.append({'type': 'quit'})
                
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                events.append({'type': 'keydown', 'key': key_name})
                
            elif event.type == pygame.KEYUP:
                key_name = pygame.key.name(event.key)
                events.append({'type': 'keyup', 'key': key_name})
                
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                zoom_factor = 1.1 if event.y > 0 else 0.9
                self.zoom *= zoom_factor
                self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom))
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:  # Middle click for pan
                    self._pan_start = event.pos
                    
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[1]:  # Middle button held
                    dx = event.pos[0] - getattr(self, '_pan_start', event.pos)[0]
                    dy = event.pos[1] - getattr(self, '_pan_start', event.pos)[1]
                    self.camera_x -= dx / self.zoom
                    self.camera_y += dy / self.zoom
                    self._pan_start = event.pos
                    
        return events
        
    def render(self, world_state: Dict, lock_state: Dict = None, 
               sim_time: float = 0.0, scenario: str = "", seed: int = 0, ui_mode: UIMode = UIMode.COMPETITION):
        """Render the map view"""
        self.screen.fill(self.COLOR_BG)
        
        # Draw grid
        self._draw_grid()
        
        # Draw safe zone
        self._draw_safe_zone()
        
        # Draw arena boundary
        self._draw_boundary()
        
        # Draw UAVs
        self._draw_uavs(world_state, lock_state)
        
        # Draw HUD
        # Draw HUD (New Overlay)
        if self.hud_overlay:
            # Construct uav_state for HUD if we have a player
            player = self._get_player(world_state)
            p_state = player.get('state', {}) if player else {}
            # Flatten or adapt p_state? 
            # HUDOverlay expects dict with: altitude, speed, throttle, battery, position, heading...
            # The 'player' dict from world_state usually has 'position', 'heading' at top level
            # and may not have speed/throttle directly if not in 'state' sub-dict.
            # Let's verify what HUDOverlay expects versus what we have.
            
            # construct HUD-friendly dict
            hud_uav = {}
            if player:
                pos = player.get('position', [0,0,0])
                vel = player.get('velocity', [0,0,0])
                hud_uav = {
                    'id': player.get('id'),
                    'position': pos,
                    'heading': player.get('heading', 0),
                    'altitude': pos[2],
                    'speed': np.linalg.norm(vel),
                    'throttle': player.get('throttle', 0.5), # Default/Estimated
                    'battery': player.get('battery', 1.0)
                }
            
            self.hud_overlay.update(hud_uav, world_state, lock_state)
            
            # Get FPS from clock
            fps = self.clock.get_fps() if self.clock else 0.0
            
            # Update flash timer
            dt = self.clock.get_time() / 1000.0 if self.clock else 0.016
            self.hud_overlay.update_flash(dt)
            
            self.hud_overlay.render(self.screen, hud_uav, world_state, lock_state, ui_mode, sim_time, fps)
            
        # Legacy HUD disabled - using HUDOverlay
        pass
        
    def _draw_grid(self):
        """Draw background grid"""
        grid_spacing = 100  # meters
        
        # Calculate visible grid lines
        left, top = self.screen_to_world(0, 0)
        right, bottom = self.screen_to_world(self.map_rect.width, self.map_rect.height)
        
        # Vertical lines
        x = int(left / grid_spacing) * grid_spacing
        while x < right:
            sx, _ = self.world_to_screen(x, 0)
            if 0 <= sx <= self.map_rect.width:
                pygame.draw.line(self.screen, self.COLOR_GRID, 
                               (sx, 0), (sx, self.map_rect.height), 1)
            x += grid_spacing
            
        # Horizontal lines
        y = int(bottom / grid_spacing) * grid_spacing
        while y < top:
            _, sy = self.world_to_screen(0, y)
            if 0 <= sy <= self.map_rect.height:
                pygame.draw.line(self.screen, self.COLOR_GRID,
                               (0, sy), (self.map_rect.width, sy), 1)
            y += grid_spacing
            
    def _draw_safe_zone(self):
        """Draw safe zone rectangle"""
        x1, y1, x2, y2 = self.safe_zone
        sx1, sy1 = self.world_to_screen(x1, y2)  # Top-left
        sx2, sy2 = self.world_to_screen(x2, y1)  # Bottom-right
        
        rect = pygame.Rect(sx1, sy1, sx2 - sx1, sy2 - sy1)
        pygame.draw.rect(self.screen, self.COLOR_SAFE_ZONE, rect, 0)
        pygame.draw.rect(self.screen, (60, 80, 70), rect, 2)
        
    def _draw_boundary(self):
        """Draw world boundary"""
        x1, y1 = self.world_to_screen(0, self.world_size[1])
        x2, y2 = self.world_to_screen(self.world_size[0], 0)
        
        rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
        pygame.draw.rect(self.screen, (80, 80, 80), rect, 3)
        
    def _draw_uavs(self, world_state: Dict, lock_state: Dict):
        """Draw all UAVs"""
        uavs = world_state.get('uavs', {})
        locked_id = lock_state.get('target_id') if lock_state else None
        
        for uav_id, uav in uavs.items():
            pos = uav.get('position', [0, 0, 0])
            heading = uav.get('heading', 0)
            team = uav.get('team', 'blue')
            is_player = uav.get('is_player', False)
            
            # Determine color
            if is_player:
                color = (50, 255, 50) # Solid Green for Player
            else:
                color = self.COLOR_ENEMY # Red for enemy
                
            # If locked, draw Halo
            if uav_id == locked_id:
                # Yellow Halo
                self._draw_lock_halo(pos[0], pos[1])
                # Locked target stays red, but has halo
                
            self._draw_uav(pos[0], pos[1], heading, color, uav_id, is_player)
            
            # Draw detection radius for player
            if is_player and self.show_detection_radius:
                self._draw_detection_radius(pos[0], pos[1])
                
    def _draw_lock_halo(self, x: float, y: float):
        """Draw pulsing halo around locked target"""
        sx, sy = self.world_to_screen(x, y)
        # Simple pulse or static ring
        pygame.draw.circle(self.screen, (255, 200, 0), (sx, sy), 30, 2)
        pygame.draw.circle(self.screen, (255, 200, 0, 100), (sx, sy), 40, 1)
                
    def _draw_uav(self, x: float, y: float, heading: float, 
                  color: Tuple[int, int, int], uav_id: str, is_player: bool):
        """Draw a single UAV"""
        sx, sy = self.world_to_screen(x, y)
        
        # UAV size based on zoom
        size = max(8, int(15 * self.zoom))
        
        # Draw body
        pygame.draw.circle(self.screen, color, (sx, sy), size)
        pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), size, 2)
        
        # Draw heading arrow
        arrow_len = size * 2
        heading_rad = np.radians(heading)
        ax = sx + arrow_len * np.cos(heading_rad)
        ay = sy - arrow_len * np.sin(heading_rad)  # Y inverted
        pygame.draw.line(self.screen, color, (sx, sy), (int(ax), int(ay)), 3)
        
        # Draw ID label (not for player)
        if not is_player:
            label = self.font_small.render(uav_id[-4:], True, color)
            self.screen.blit(label, (sx - label.get_width()//2, sy + size + 5))
            
    def _draw_detection_radius(self, x: float, y: float):
        """Draw detection radius ring"""
        sx, sy = self.world_to_screen(x, y)
        radius = int(self.detection_radius * self.zoom)
        pygame.draw.circle(self.screen, (60, 80, 60), (sx, sy), radius, 1)
        
    def _draw_hud(self, world_state: Dict, lock_state: Dict,
                  sim_time: float, scenario: str, seed: int):
        """Draw HUD overlay"""
        # Top bar
        self._draw_top_hud(sim_time, scenario, seed)
        
        # Bottom bar - lock status
        self._draw_lock_hud(lock_state)
        
        # Player telemetry
        player = self._get_player(world_state)
        if player:
            self._draw_telemetry_hud(player)
            
    def _draw_top_hud(self, sim_time: float, scenario: str, seed: int):
        """Draw top HUD bar"""
        # Background
        bar_height = 40
        surface = pygame.Surface((self.map_rect.width, bar_height), pygame.SRCALPHA)
        surface.fill(self.COLOR_HUD_BG)
        self.screen.blit(surface, (0, 0))
        
        # Text
        time_text = f"Time: {sim_time:.1f}s"
        scenario_text = f"Scenario: {scenario}"
        seed_text = f"Seed: {seed}"
        
        self._draw_text(time_text, 10, 10)
        self._draw_text(scenario_text, 200, 10)
        self._draw_text(seed_text, 450, 10)
        
    def _draw_lock_hud(self, lock_state: Dict):
        """Draw lock status bar at bottom"""
        if not lock_state:
            return
            
        bar_height = 50
        y = self.map_rect.height - bar_height
        
        # Background
        surface = pygame.Surface((self.map_rect.width, bar_height), pygame.SRCALPHA)
        surface.fill(self.COLOR_HUD_BG)
        self.screen.blit(surface, (0, y))
        
        # Lock state
        state = lock_state.get('state', 'idle').upper()
        target_id = lock_state.get('target_id', '--')
        progress = lock_state.get('progress', 0)
        lock_time = lock_state.get('lock_time', 0)
        score = lock_state.get('score', {})
        
        # State color
        state_colors = {
            'IDLE': (150, 150, 150),
            'LOCKING': (255, 200, 50),
            'SUCCESS': (100, 255, 100)
        }
        state_color = state_colors.get(state, self.COLOR_TEXT)
        
        # Draw state
        state_text = f"Lock: {state}"
        self._draw_text(state_text, 10, y + 15, state_color)
        
        # Draw target
        target_text = f"Target: {target_id if target_id else '--'}"
        self._draw_text(target_text, 180, y + 15)
        
        # Draw progress bar
        bar_x = 350
        bar_width = 150
        bar_rect = pygame.Rect(bar_x, y + 15, bar_width, 20)
        pygame.draw.rect(self.screen, (50, 50, 50), bar_rect)
        
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            fill_color = (100, 255, 100) if progress >= 1.0 else (255, 200, 50)
            fill_rect = pygame.Rect(bar_x, y + 15, fill_width, 20)
            pygame.draw.rect(self.screen, fill_color, fill_rect)
            
        pygame.draw.rect(self.screen, (100, 100, 100), bar_rect, 2)
        
        # Draw lock time
        time_text = f"{lock_time:.1f}s / 4.0s"
        self._draw_text(time_text, 520, y + 15)
        
        # Draw score
        total_score = score.get('total_score', 0)
        correct = score.get('correct_locks', 0)
        score_text = f"Score: {total_score} ({correct} locks)"
        self._draw_text(score_text, 700, y + 15)
        
    def _draw_telemetry_hud(self, player: Dict):
        """Draw player telemetry"""
        x = 10
        y = 60
        
        pos = player.get('position', [0, 0, 0])
        heading = player.get('heading', 0)
        speed = np.linalg.norm(player.get('velocity', [0, 0, 0]))
        altitude = pos[2] if len(pos) > 2 else 0
        
        # Background
        surface = pygame.Surface((180, 80), pygame.SRCALPHA)
        surface.fill(self.COLOR_HUD_BG)
        self.screen.blit(surface, (x, y))
        
        self._draw_text(f"Speed: {speed:.1f} m/s", x + 10, y + 10, size='small')
        self._draw_text(f"Alt: {altitude:.0f} m", x + 10, y + 30, size='small')
        self._draw_text(f"Hdg: {heading:.0f}°", x + 10, y + 50, size='small')
        
    def _draw_text(self, text: str, x: int, y: int, 
                   color: Tuple[int, int, int] = None, size: str = 'normal'):
        """Draw text"""
        color = color or self.COLOR_TEXT
        font = self.font if size == 'normal' else self.font_small
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))
        
    def _get_player(self, world_state: Dict) -> Optional[Dict]:
        """Get player UAV from world state"""
        uavs = world_state.get('uavs', {})
        for uav_id, uav in uavs.items():
            if uav.get('is_player', False):
                return uav
        return None
