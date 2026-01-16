
import pygame
import math
import numpy as np
from typing import Dict, List, Any
from .theme import Colors, Layout, Fonts
from .widgets import Panel, VerticalTape, ProgressBar, Radar
from .ui_mode import UIMode

class HUDOverlay:
    """
    Main Head-Up Display controller implementation.
    Manages layout and rendering of all UI components.
    """
    
    def __init__(self, width: int, height: int, radar_heading_mode: str = "heading_up"):
        self.width = width
        self.height = height
        self.radar_heading_mode = radar_heading_mode
        self.fonts = {}
        self._init_fonts()
        self._init_widgets()
        
        # Flash animation state
        self._flash_timer = 0.0
        self._flash_color = None  # (r, g, b) or None
        self._flash_duration = 0.3  # seconds
        
    def _init_fonts(self):
        # Use default sys font for portability, but style it
        self.fonts['header'] = pygame.font.SysFont("Verdana", Fonts.HEADER_SIZE, bold=True)
        self.fonts['normal'] = pygame.font.SysFont("Consolas", Fonts.NORMAL_SIZE) # Monospace for data
        self.fonts['small'] = pygame.font.SysFont("Consolas", Fonts.SMALL_SIZE)
        self.fonts['large'] = pygame.font.SysFont("Verdana", Fonts.HUGE_SIZE, bold=True)
        
    def _init_widgets(self):
        # 1. Left Panel (Telemetry)
        self.panel_left = Panel(Layout.MARGIN, Layout.TOP_BAR_HEIGHT + Layout.PADDING, 
                                Layout.SIDE_PANEL_WIDTH, self.height - Layout.TOP_BAR_HEIGHT - Layout.PADDING*2,
                                title="TELEMETRY", font=self.fonts['header'])
                                
        # Tapes inside left panel
        tape_h = 300
        tape_y = self.panel_left.rect.top + 60
        
        self.alt_tape = VerticalTape(self.panel_left.rect.x + 20, tape_y, 60, tape_h, 
                                     0, 2000, 100, step=50, label="ALT")
                                     
        self.spd_tape = VerticalTape(self.panel_left.rect.x + 100, tape_y, 60, tape_h,
                                     0, 200, 20, step=10, label="SPD")
                                     
        # Fuel/Throttle
        bar_w = 200
        self.throttle_bar = ProgressBar(self.panel_left.rect.x + 25, tape_y + tape_h + 40, 
                                        bar_w, 20, "THROTTLE", Colors.PRIMARY)
        self.battery_bar = ProgressBar(self.panel_left.rect.x + 25, tape_y + tape_h + 90,
                                       bar_w, 20, "BATTERY", Colors.SUCCESS)
                                       
        # 2. Right Panel (Tactical)
        p_right_x = self.width - Layout.SIDE_PANEL_WIDTH - Layout.MARGIN
        self.panel_right = Panel(p_right_x, Layout.TOP_BAR_HEIGHT + Layout.PADDING,
                                 Layout.SIDE_PANEL_WIDTH, self.height - Layout.TOP_BAR_HEIGHT - Layout.PADDING*2,
                                 title="TACTICAL", font=self.fonts['header'])
                                 
        # Radar inside right panel
        r_radius = (Layout.SIDE_PANEL_WIDTH - 40) // 2
        r_x = p_right_x + 20
        r_y = self.panel_right.rect.top + 50
        self.radar = Radar(r_x, r_y, r_radius)
        
        # Target List
        self.list_y = r_y + r_radius*2 + 20
        
        # 3. Top Panel (Global Status)
        # Using simple draw calls in render, no Widget class needed for static text bar
        
    def update(self, uav_state: Dict, world_state: Dict, lock_state: Dict):
        """Update widget states."""
        # Update Telemetry
        if uav_state:
            self.alt_tape.update(uav_state.get('altitude', 0))
            self.spd_tape.update(uav_state.get('speed', 0))
            self.throttle_bar.value = uav_state.get('throttle', 0)
            self.battery_bar.value = uav_state.get('battery', 1.0)

    def set_radar_heading_mode(self, mode: str):
        """Set radar heading mode ('heading_up' or 'north_up')."""
        self.radar_heading_mode = mode
    
    def _extract_score(self, lock_state: Dict) -> int:
        """Extract score value consistently regardless of format.
        
        Handles both:
        - lock_state['score'] = int (direct value)
        - lock_state['score'] = {'total_score': int, ...} (dict format)
        
        Returns:
            int: Score value, defaults to 0
        """
        if not lock_state:
            return 0
        score = lock_state.get('score', 0)
        if isinstance(score, dict):
            return int(score.get('total_score', 0))
        return int(score) if score else 0
            
    def render(self, surface: pygame.Surface, uav_state: Dict, world_state: Dict, lock_state: Dict, 
               ui_mode: UIMode = UIMode.COMPETITION, sim_time: float = 0.0, fps: float = 0.0):
        """Render entire HUD.
        
        Args:
            surface: Pygame surface to render on
            uav_state: Player UAV state dict
            world_state: World state dict
            lock_state: Lock-on state dict
            ui_mode: Current UI mode
            sim_time: Simulation time in seconds
            fps: Current frames per second
        """
        
        # 1. Widgets
        # 1. Widgets
        
        if ui_mode == UIMode.COMPETITION:
            # Minimal Telemetry - Text Only
            # Draw simple box with ALT / SPD
            
            # Position manually or use existing panel rect
            x, y = self.panel_left.rect.x, self.panel_left.rect.y
            
            # Draw numeric telemetry
            spd = uav_state.get('speed', 0)
            alt = uav_state.get('altitude', 0)
            batt = uav_state.get('battery', 1.0)
            
            # Label Color
            lbl_col = Colors.TEXT_DIM
            val_col = Colors.TEXT_MAIN
            
            # Speed
            surface.blit(self.fonts['small'].render("SPD (m/s)", True, lbl_col), (x + 20, y + 60))
            surface.blit(self.fonts['large'].render(f"{spd:.1f}", True, val_col), (x + 20, y + 80))
            
            # Alt
            surface.blit(self.fonts['small'].render("ALT (m)", True, lbl_col), (x + 20, y + 140))
            surface.blit(self.fonts['large'].render(f"{alt:.0f}", True, val_col), (x + 20, y + 160))
            
            # Battery simple bar
            self.battery_bar.render(surface, self.fonts['small'])
            
        else:
            # Full DEBUG / TRAINING Telemetry
            self.panel_left.render(surface)
            self.alt_tape.render(surface, self.fonts['normal'])
            self.spd_tape.render(surface, self.fonts['normal'])
            self.throttle_bar.render(surface, self.fonts['small'])
            self.battery_bar.render(surface, self.fonts['small'])
        
        self.panel_right.render(surface)
        
        # Prepare radar targets
        # Need relative conversion from world_state uavs
        radar_targets = []
        player_pos = np.array(uav_state.get('position', [0,0,0]))
        player_heading = uav_state.get('heading', 0)  # degrees
        heading_up = self.radar_heading_mode != "north_up"
        
        # Rotation matrix for Heading Up
        # Global: X=East, Y=North. 
        # Body: X=Forward, Y=Right.
        # Heading is angle from East (CCW). 
        # Actually in simulation: 0=East, 90=North. So standard math.
        # To convert Global (dx, dy) to Body (bx, by):
        # Rotate by -heading.
        heading_rad = math.radians(player_heading)
        c = math.cos(-heading_rad)
        s = math.sin(-heading_rad)
        
        for uid, uav in world_state.get('uavs', {}).items():
            if uid == uav_state.get('id'): continue
            
            pos = np.array(uav.get('position', [0,0,0]))
            rel = pos - player_pos
            
            if heading_up:
                # Rotate to body frame (heading-up radar)
                bx = rel[0]*c - rel[1]*s
                by = rel[0]*s + rel[1]*c
                rel_pos = [bx, -by]
            else:
                # North-up radar: forward is North (+Y), right is East (+X)
                rel_pos = [rel[1], rel[0]]
            
            # bx is 'East-aligned' rotated?
            # If heading=0 (East): c=1, s=0. bx=dx (East), by=dy (North).
            # Body X is Forward. East is Forward. So Correct.
            # Radar expects forward as 'up' (screen -y) or similar.
            # My Radar widget logic: passed (bx, by) -> 'rel_pos'
            # Widget logic: sx = center + ry*scale (Right), sy = center - rx*scale (Forward) 
            # If bx is forward, by is left? 
            # Standard: if heading=0, X is forward. Y is Left (North is Left of East).
            # Wait. East=0, North=90. Y is +90 deg from X. So Y is Left.
            # Check coords: X=10, Y=0 (Ahead). bx=10. sy = -10 (Up on screen). Correct.
            # X=0, Y=10 (Left). by=10. sx = center + 10. (Right on screen).
            # If Y is 'North' (Left of East), and screen Right is 'Right', we need -by.
            
            radar_targets.append({
                'rel_pos': rel_pos,
                'type': 'enemy' if 'target' in uid else 'friend',
                'locked': (lock_state.get('target_id') == uid)
            })
            
        self.radar.render(surface, radar_targets, player_heading, 60, heading_up=heading_up)
        
        # Target List
        y_off = self.list_y
        lbl = self.fonts['small'].render("TARGET LIST", True, Colors.PRIMARY)
        surface.blit(lbl, (self.panel_right.rect.x + 20, y_off))
        y_off += 20
        
        for t in radar_targets[:5]: # Show top 5
            # Dist
            d = math.sqrt(t['rel_pos'][0]**2 + t['rel_pos'][1]**2)
            c = Colors.DANGER if t['locked'] else Colors.TEXT_MAIN
            conn_sym = "[LOCKED]" if t['locked'] else "TRACK"
            txt = self.fonts['small'].render(f"HK: {d:.0f}m {conn_sym}", True, c)
            surface.blit(txt, (self.panel_right.rect.x + 20, y_off))
            y_off += 15
        
        # 2. Top Bar
        top_rect = pygame.Rect(0, 0, self.width, Layout.TOP_BAR_HEIGHT)
        s = pygame.Surface((self.width, Layout.TOP_BAR_HEIGHT), pygame.SRCALPHA)
        s.fill((*Colors.BACKGROUND, 240))
        surface.blit(s, (0,0))
        pygame.draw.line(surface, Colors.PRIMARY, (0, Layout.TOP_BAR_HEIGHT), (self.width, Layout.TOP_BAR_HEIGHT), 2)
        
        # Score (Center) - use helper for consistent type handling
        score = self._extract_score(lock_state)
        
        score_text = self.fonts['large'].render(f"{score:04d}", True, Colors.SUCCESS)
        surface.blit(score_text, (self.width//2 - score_text.get_width()//2, 5))
        
        # FPS / Info (Left) - now with sim_time and FPS
        mode_str = ui_mode.name
        time_str = f"{sim_time:.1f}s" if sim_time > 0 else "0.0s"
        fps_str = f"{fps:.0f}" if fps > 0 else "--"
        info_txt = self.fonts['small'].render(f"T: {time_str} | FPS: {fps_str} | {mode_str}", True, Colors.TEXT_DIM)
        surface.blit(info_txt, (20, 20))
        
        # 3. Main HUD (Center Overlay)
        center_x, center_y = self.width // 2, self.height // 2
        
        # Crosshair
        ch_size = 20
        ch_col = Colors.DANGER if lock_state.get('is_locked') else Colors.PRIMARY
        
        # Center gap
        pygame.draw.line(surface, ch_col, (center_x - ch_size, center_y), (center_x - 5, center_y), 2)
        pygame.draw.line(surface, ch_col, (center_x + 5, center_y), (center_x + ch_size, center_y), 2)
        pygame.draw.line(surface, ch_col, (center_x, center_y - ch_size), (center_x, center_y - 5), 2)
        pygame.draw.line(surface, ch_col, (center_x, center_y + 5), (center_x, center_y + ch_size), 2)
        
        # Lock Indicator
        if lock_state.get('is_locked'):
            # Draw box around target
            # Need screen coords of target. This is handled by renderer.render_detections usually.
            # But we can draw a 'Locking' circle around crosshair
            
            prog = lock_state.get('progress', 0.0) # 0 to 1
            if prog > 0:
                rect = pygame.Rect(0, 0, 60, 60)
                rect.center = (center_x, center_y)
                pygame.draw.arc(surface, Colors.DANGER, rect, 0, math.pi * 2 * prog, 3)
                
            if lock_state.get('is_valid'):
                lbl = self.fonts['header'].render("SHOOT", True, Colors.SUCCESS)
                surface.blit(lbl, (center_x - lbl.get_width()//2, center_y - 80))
            else:
                lbl = self.fonts['header'].render("LOCKING", True, Colors.WARNING)
                surface.blit(lbl, (center_x - lbl.get_width()//2, center_y - 80))
        
        # 4. Flash overlay (for lock success/lost feedback)
        self._render_flash(surface)
    
    def flash_success(self):
        """Trigger green flash for successful lock."""
        self._flash_color = Colors.SUCCESS
        self._flash_timer = self._flash_duration
        
    def flash_lost(self):
        """Trigger red flash for lost lock."""
        self._flash_color = Colors.DANGER
        self._flash_timer = self._flash_duration
        
    def update_flash(self, dt: float):
        """Update flash timer. Call each frame with delta time."""
        if self._flash_timer > 0:
            self._flash_timer -= dt
            if self._flash_timer <= 0:
                self._flash_timer = 0
                self._flash_color = None
    
    def _render_flash(self, surface: pygame.Surface):
        """Render flash overlay if active."""
        if self._flash_timer > 0 and self._flash_color:
            # Calculate alpha based on remaining time (fade out)
            alpha = int(80 * (self._flash_timer / self._flash_duration))
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((*self._flash_color, alpha))
            surface.blit(overlay, (0, 0))
