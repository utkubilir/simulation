import pygame
import math
import numpy as np
from src.simulation.ui.theme import Colors, Fonts
from src.simulation.ui.widgets import Radar, VerticalTape, ProgressBar

class HUD:
    """
    Head-Up Display / Primary Flight Display (PFD)
    
    Modernized to use 'Tactical Glass' theme and widgets.
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        self.font = None
        self.font_large = None
        self.font_small = None
        
        # Horizon scaling
        self.pixels_per_degree = height / 60.0 # 60 degrees visible vertically
        
        # Initialize Widgets
        # Speed Tape (Left)
        tape_w, tape_h = 60, 300
        self.spd_tape = VerticalTape(
            x=50,
            y=self.center_y - tape_h//2,
            width=tape_w,
            height=tape_h,
            min_val=0, max_val=200, current_val=0, step=10, label="SPD"
        )

        # Altitude Tape (Right)
        self.alt_tape = VerticalTape(
            x=width - 50 - tape_w,
            y=self.center_y - tape_h//2,
            width=tape_w,
            height=tape_h,
            min_val=0, max_val=2000, current_val=0, step=50, label="ALT"
        )

        # Throttle Bar (Bottom Left)
        self.throttle_bar = ProgressBar(
            x=50,
            y=height - 50,
            width=200,
            height=20,
            label="THROTTLE",
            color=Colors.PRIMARY
        )

        # Radar (Bottom Right)
        self.radar = Radar(
            x=width - 220,
            y=height - 220,
            radius=100
        )

    def init(self):
        """Initialize fonts (must be called after pygame.init)"""
        if not pygame.font.get_init():
             pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 18)
        
    def render(self, surface: pygame.Surface, uav_state: dict, 
               detections: list = None, lock_state: dict = None,
               ui_mode = None, world_state: dict = None,
               camera_config: dict = None):
        """
        Render HUD elements.
        
        Args:
            uav_state: dict with position, velocity, heading, attitude (pitch/roll)
            detections: list of detection dicts (optional)
            lock_state: dict of lock status (optional)
            ui_mode: UI mode enum (optional, for mode-specific rendering)
            world_state: World state dict (optional, for Radar)
            camera_config: dict with 'fov' and 'resolution' (optional)
        """
        # Extract state
        orientation = uav_state.get('orientation', [0.0, 0.0, 0.0])
        if len(orientation) >= 2:
            roll = orientation[0]
            pitch = orientation[1]
        else:
            roll = uav_state.get('roll', 0.0)
            pitch = uav_state.get('pitch', 0.0)
            
        heading = uav_state.get('heading', 0.0) # degrees
        
        vel = np.linalg.norm(uav_state.get('velocity', np.zeros(3)))
        pos = uav_state.get('position', np.zeros(3))
        alt = pos[2]
        
        # Calculate dynamic FOV scale if config provided
        if camera_config:
            fov = camera_config.get('fov', 60.0)
            # Update pixels per degree based on current FOV and height
            self.pixels_per_degree = self.height / fov
        
        # Update Widgets
        self.spd_tape.update(vel)
        self.alt_tape.update(alt)
        self.throttle_bar.value = uav_state.get('throttle', 0.5)

        # 1. Artificial Horizon (Lines Only)
        self._draw_horizon_line(surface, pitch, roll)
        
        # 2. Pitch Ladder
        self._draw_pitch_ladder(surface, pitch, roll)
        
        # 3. Bank Indicator
        self._draw_bank_indicator(surface, roll)
        
        # 4. AR Detections
        if detections:
            self._draw_detections(surface, detections, lock_state, camera_config)
            
        # 5. Crosshair (Center)
        self._draw_crosshair(surface, lock_state)

        # 6. Widgets
        self.spd_tape.render(surface, self.font)
        self.alt_tape.render(surface, self.font)
        self.throttle_bar.render(surface, self.font_small)
        
        # 7. Radar
        if world_state:
            self._render_radar(surface, uav_state, world_state, lock_state)

        # 8. Heading Strip
        self._draw_heading_strip(surface, heading)

    def _render_radar(self, surface, uav_state, world_state, lock_state):
        radar_targets = []
        player_pos = np.array(uav_state.get('position', [0,0,0]))
        player_heading = uav_state.get('heading', 0)

        # HUD Radar is usually "Heading Up" (Forward is Up)
        heading_rad = math.radians(player_heading)
        c = math.cos(-heading_rad)
        s = math.sin(-heading_rad)

        current_id = uav_state.get('id')

        for uid, uav in world_state.get('uavs', {}).items():
            if uid == current_id: continue

            pos = np.array(uav.get('position', [0,0,0]))
            rel = pos - player_pos

            # Rotate to body frame
            bx = rel[0]*c - rel[1]*s
            by = rel[0]*s + rel[1]*c

            radar_targets.append({
                'rel_pos': [bx, -by],
                'type': 'enemy' if 'target' in uid else 'friend',
                'locked': (lock_state.get('target_id') == uid) if lock_state else False
            })

        self.radar.render(surface, radar_targets, player_heading, 60, heading_up=True)

    def _draw_detections(self, surface, detections, lock_state, camera_config=None):
        """Draw Augmented Reality bounding boxes"""
        locked_id = lock_state.get('target_id') if lock_state else None
        
        # Dynamic camera scale
        cam_w, cam_h = 640, 480
        if camera_config:
            res = camera_config.get('resolution', (640, 480))
            cam_w, cam_h = res[0], res[1]
            
        scale_x = self.width / cam_w
        scale_y = self.height / cam_h
        
        for det in detections:
            bbox = det.get('bbox')
            if not bbox: continue
            
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            w = x2 - x1
            h = y2 - y1
            
            # Determine Color
            is_locked = (det.get('world_id') == locked_id)
            color = Colors.DANGER if is_locked else Colors.WARNING
            thickness = 3 if is_locked else 1
            
            # Draw Bracket Corners (Sci-Fi style)
            l = min(w, h) // 4
            # Top-Left
            pygame.draw.line(surface, color, (x1, y1), (x1 + l, y1), thickness)
            pygame.draw.line(surface, color, (x1, y1), (x1, y1 + l), thickness)
            # Top-Right
            pygame.draw.line(surface, color, (x2, y1), (x2 - l, y1), thickness)
            pygame.draw.line(surface, color, (x2, y1), (x2, y1 + l), thickness)
            # Bottom-Left
            pygame.draw.line(surface, color, (x1, y2), (x1 + l, y2), thickness)
            pygame.draw.line(surface, color, (x1, y2), (x1, y2 - l), thickness)
            # Bottom-Right
            pygame.draw.line(surface, color, (x2, y2), (x2 - l, y2), thickness)
            pygame.draw.line(surface, color, (x2, y2), (x2, y2 - l), thickness)
            
            # Label
            conf = det.get('confidence', 0)
            dist = det.get('distance', 0)
            s_txt = f"TGT {conf:.0%} [{dist:.0f}m]"
            txt = self.font.render(s_txt, True, color)
            surface.blit(txt, (x1, y1 - 20))

    def _draw_crosshair(self, surface, lock_state):
        """Draw central crosshair and lock status"""
        cx, cy = self.center_x, self.center_y

        is_locked = lock_state.get('is_locked') if lock_state else False
        color = Colors.DANGER if is_locked else Colors.HUD_LINE

        # Gap center
        gap = 10
        len_ = 20

        pygame.draw.line(surface, color, (cx - gap - len_, cy), (cx - gap, cy), 2)
        pygame.draw.line(surface, color, (cx + gap, cy), (cx + gap + len_, cy), 2)
        pygame.draw.line(surface, color, (cx, cy - gap - len_), (cx, cy - gap), 2)
        pygame.draw.line(surface, color, (cx, cy + gap), (cx, cy + gap + len_), 2)

        # Center dot
        pygame.draw.circle(surface, color, (cx, cy), 2)

        # Lock Arc/Text
        if lock_state and lock_state.get('target_id'):
            prog = lock_state.get('progress', 0)
            state = lock_state.get('state', 'IDLE')

            # Arc
            if prog > 0:
                rect = pygame.Rect(0, 0, 80, 80)
                rect.center = (cx, cy)
                arc_col = Colors.WARNING if prog < 1.0 else Colors.DANGER
                pygame.draw.arc(surface, arc_col, rect, 0, math.pi * 2 * prog, 3)

            # Text
            text = ""
            if state == 'LOCKING':
                text = "LOCKING..."
                col = Colors.WARNING
            elif state == 'SUCCESS':
                text = "SHOOT"
                col = Colors.SUCCESS

            if text:
                lbl = self.font_large.render(text, True, col)
                surface.blit(lbl, (cx - lbl.get_width()//2, cy - 100))

    def _draw_horizon_line(self, surface, pitch, roll):
        """Draw just the horizon line"""
        pitch_deg = math.degrees(pitch)
        pitch_offset = pitch_deg * self.pixels_per_degree
        
        diag = math.sqrt(self.width**2 + self.height**2)
        radius = diag
        
        sin_r = math.sin(roll)
        cos_r = math.cos(roll)
        
        cx = self.center_x + pitch_offset * sin_r
        cy = self.center_y + pitch_offset * cos_r
        
        dx = radius * cos_r
        dy = -radius * sin_r
        
        p1 = (cx - dx, cy - dy)
        p2 = (cx + dx, cy + dy)
        
        # Draw Line
        pygame.draw.line(surface, Colors.HUD_LINE, p1, p2, 2)

    def _draw_pitch_ladder(self, surface, pitch, roll):
        """Pitch ladder lines"""
        pitch_deg = math.degrees(pitch)
        
        # Draw range +/- 20 degrees
        start_p = math.floor((pitch_deg - 25) / 5) * 5
        end_p = math.ceil((pitch_deg + 25) / 5) * 5
        
        sin_r = math.sin(roll)
        cos_r = math.cos(roll)
        
        for p in range(int(start_p), int(end_p) + 5, 5):
            if p == 0: continue
            
            dist_angle = pitch_deg - p
            dist_px = dist_angle * self.pixels_per_degree
            
            # Rotate this offsets
            rcx = self.center_x + dist_px * sin_r
            rcy = self.center_y + dist_px * cos_r
            
            w = 80 if p % 10 == 0 else 40
            dx = cos_r * w / 2
            dy = -sin_r * w / 2
            
            start = (rcx - dx, rcy - dy)
            end = (rcx + dx, rcy + dy)
            
            if not (0 <= rcx <= self.width and 0 <= rcy <= self.height):
                continue
                
            pygame.draw.line(surface, Colors.HUD_LINE, start, end, 2)
            
            if p % 10 == 0:
                text = self.font.render(str(abs(p)), True, Colors.HUD_LINE)
                surface.blit(text, (end[0] + 5, end[1] - 8))

    def _draw_bank_indicator(self, surface, roll):
        """Top arc bank scale"""
        radius = min(self.width, self.height) // 2 - 40
        cx, cy = self.center_x, self.center_y
        
        # Scale range: +/- 60 deg
        for angle in [0, 10, 20, 30, 45, 60, -10, -20, -30, -45, -60]:
            rad = math.radians(angle - 90) - roll
            
            x1 = cx + radius * math.cos(rad)
            y1 = cy + radius * math.sin(rad)
            x2 = cx + (radius + 10) * math.cos(rad)
            y2 = cy + (radius + 10) * math.sin(rad)
            
            w = 3 if angle == 0 else 2
            pygame.draw.line(surface, Colors.HUD_LINE, (x1, y1), (x2, y2), w)
            
        # Fixed Triangle
        tri_pts = [
            (cx, cy - radius + 5),
            (cx - 6, cy - radius + 20),
            (cx + 6, cy - radius + 20)
        ]
        pygame.draw.polygon(surface, Colors.WARNING, tri_pts)

    def _draw_heading_strip(self, surface, heading):
        """Top heading strip"""
        w, h = 400, 40
        x = (self.width - w) // 2
        y = 10
        
        # Transparent BG
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((20, 20, 20, 150))
        surface.blit(bg, (x, y))
        
        pygame.draw.rect(surface, Colors.HUD_LINE, (x, y, w, h), 1)
        
        cx = x + w // 2
        pixels_per_deg = w / 60.0
        
        min_h = int(heading - 35)
        max_h = int(heading + 35)
        
        for h_val in range(min_h, max_h):
            if h_val % 10 == 0:
                h_norm = h_val % 360
                
                dx = (h_val - heading) * pixels_per_deg
                lx = cx + dx
                
                if x <= lx <= x+w:
                    pygame.draw.line(surface, Colors.HUD_LINE, (lx, y+h-10), (lx, y+h), 2)
                    
                    if h_norm % 90 == 0:
                        dirs = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
                        txt = dirs[h_norm]
                    else:
                        txt = str(h_norm // 10)
                        
                    lbl = self.font.render(txt, True, Colors.HUD_LINE)
                    surface.blit(lbl, (lx - 6, y+5))
            elif h_val % 5 == 0:
                dx = (h_val - heading) * pixels_per_deg
                lx = cx + dx
                if x <= lx <= x+w:
                    pygame.draw.line(surface, Colors.HUD_LINE, (lx, y+h-5), (lx, y+h), 1)
                    
        # Center Marker
        pygame.draw.polygon(surface, Colors.WARNING, [
            (cx, y+h+5), (cx-5, y+h+15), (cx+5, y+h+15)
        ])
        
        # Box
        lbl = self.font_large.render(f"{heading:.0f}°", True, Colors.WARNING)
        surface.blit(lbl, (cx - 20, y + h + 20))
