import pygame
import math
import numpy as np

class HUD:
    """
    Head-Up Display / Primary Flight Display (PFD)
    
    Provides:
    - Artificial Horizon (Pitch/Roll)
    - Bank Indicator
    - Speed & Altitude Tapes (Optional overlays)
    - Compass Strip (Optional)
    """
    
    # Colors
    COLOR_SKY = (0, 120, 200)       # Sky Blue
    COLOR_GROUND = (116, 92, 48)    # Earth Brown
    COLOR_LINE = (255, 255, 255)    # White
    COLOR_TEXT = (255, 255, 255)    # White
    COLOR_BG_TAPE = (40, 40, 40, 180) # Semi-transparent gray
    COLOR_REF = (255, 255, 0)       # Yellow Aircraft Reference
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        self.font = None
        self.font_large = None
        
        # Horizon scaling
        self.pixels_per_degree = height / 60.0 # 60 degrees visible vertically
        
    def init(self):
        """Initialize fonts (must be called after pygame.init)"""
        if not pygame.font.get_init():
             pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
    def render(self, surface: pygame.Surface, uav_state: dict, 
               detections: list = None, lock_state: dict = None,
               ui_mode = None):
        """
        Render HUD elements.
        
        Args:
            uav_state: dict with position, velocity, heading, attitude (pitch/roll)
            detections: list of detection dicts (optional)
            lock_state: dict of lock status (optional)
            ui_mode: UI mode enum (optional, for mode-specific rendering)
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
        
        # 1. Artificial Horizon
        self._draw_horizon(surface, pitch, roll)
        
        # 2. Pitch Ladder
        self._draw_pitch_ladder(surface, pitch, roll)
        
        # 3. Bank Indicator
        self._draw_bank_indicator(surface, roll)
        
        # 4. AR Detections
        if detections:
            self._draw_detections(surface, detections, lock_state)
            
        # 5. Aircraft Reference (Fixed)
        self._draw_aircraft_ref(surface)
        
        # 6. Speed Tape (Left)
        self._draw_tape(surface, 'speed', vel, 
                       x=0, y=self.height//2 - 150, w=70, h=300, 
                       step=10, pixel_step=5)
                       
        # 7. Altitude Tape (Right)
        self._draw_tape(surface, 'alt', alt, 
                       x=self.width-70, y=self.height//2 - 150, w=70, h=300, 
                       step=50, pixel_step=2)
                       
        # 8. Heading Strip (Top)
        self._draw_heading_strip(surface, heading)

    def _draw_detections(self, surface, detections, lock_state):
        """Draw Augmented Reality bounding boxes"""
        locked_id = lock_state.get('target_id') if lock_state else None
        
        # Assuming detections have 'bbox' [x1, y1, x2, y2] normalized? 
        # Or pixel coords for the camera frame?
        # SimulationDetector usually returns pixel coords for the generated frame (640x480).
        # We need to scale them to current HUD/screen size.
        
        # Default camera size
        cam_w, cam_h = 640, 480
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
            color = (255, 0, 0) if is_locked else (255, 255, 0)
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

    def _draw_horizon(self, surface, pitch, roll):
        """Rotating horizon background"""
        pitch_deg = math.degrees(pitch)
        pitch_offset = pitch_deg * self.pixels_per_degree
        
        # Huge diagonal size
        diag = math.sqrt(self.width**2 + self.height**2)
        radius = diag
        
        sin_r = math.sin(roll)
        cos_r = math.cos(roll)
        
        # Center of horizon line
        cx = self.center_x + pitch_offset * sin_r
        cy = self.center_y + pitch_offset * cos_r
        
        # Horizon Line vector
        dx = radius * cos_r
        dy = -radius * sin_r
        
        p1 = (cx - dx, cy - dy)
        p2 = (cx + dx, cy + dy)
        
        # Sky Polygon (perpendicular up)
        up_x = sin_r * radius
        up_y = cos_r * radius
        
        sky_poly = [
            p1, p2, 
            (p2[0] - up_x, p2[1] - up_y),
            (p1[0] - up_x, p1[1] - up_y)
        ]
        
        ground_poly = [
            p1, p2,
            (p2[0] + up_x, p2[1] + up_y),
            (p1[0] + up_x, p1[1] + up_y)
        ]
        
        pygame.draw.polygon(surface, self.COLOR_SKY, sky_poly)
        pygame.draw.polygon(surface, self.COLOR_GROUND, ground_poly)
        pygame.draw.line(surface, self.COLOR_LINE, p1, p2, 2)

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
            
            # Pixel distance from center
            # if p > pitch, line is ABOVE
            # screen Y is down positive.
            # dist_y = -(p - pitch) * scale
            
            dist_angle = pitch_deg - p # Positive if pitch > p (we are looking up, line is below)
            # Wait, if pitch=10, line 0 is below center. (10-0)=10 deg down.
            
            dist_px = dist_angle * self.pixels_per_degree
            
            # Rotate this offsets
            # center offset (0, dist_px) rotated by -roll
            rcx = self.center_x + dist_px * sin_r
            rcy = self.center_y + dist_px * cos_r
            
            w = 80 if p % 10 == 0 else 40
            dx = cos_r * w / 2
            dy = -sin_r * w / 2
            
            start = (rcx - dx, rcy - dy)
            end = (rcx + dx, rcy + dy)
            
            # Check bounds roughly
            if not (0 <= rcx <= self.width and 0 <= rcy <= self.height):
                continue
                
            pygame.draw.line(surface, self.COLOR_LINE, start, end, 2)
            
            if p % 10 == 0:
                text = self.font.render(str(abs(p)), True, self.COLOR_LINE)
                surface.blit(text, (end[0] + 5, end[1] - 8))

    def _draw_bank_indicator(self, surface, roll):
        """Top arc bank scale"""
        radius = min(self.width, self.height) // 2 - 40
        cx, cy = self.center_x, self.center_y
        
        # Scale range: +/- 60 deg
        for angle in [0, 10, 20, 30, 45, 60, -10, -20, -30, -45, -60]:
            # Scale moves with roll
            rad = math.radians(angle - 90) - roll
            
            x1 = cx + radius * math.cos(rad)
            y1 = cy + radius * math.sin(rad)
            x2 = cx + (radius + 10) * math.cos(rad)
            y2 = cy + (radius + 10) * math.sin(rad)
            
            w = 3 if angle == 0 else 2
            pygame.draw.line(surface, self.COLOR_LINE, (x1, y1), (x2, y2), w)
            
        # Fixed Triangle
        tri_pts = [
            (cx, cy - radius + 5),
            (cx - 6, cy - radius + 20),
            (cx + 6, cy - radius + 20)
        ]
        pygame.draw.polygon(surface, self.COLOR_REF, tri_pts)

    def _draw_aircraft_ref(self, surface):
        """Fixed aircraft symbol"""
        cx, cy = self.center_x, self.center_y
        
        # Left Wing
        pygame.draw.rect(surface, (0,0,0), (cx - 102, cy - 2, 74, 8)) # Outline
        pygame.draw.rect(surface, self.COLOR_REF, (cx - 100, cy, 70, 4))
        
        # Right Wing
        pygame.draw.rect(surface, (0,0,0), (cx + 28, cy - 2, 74, 8)) # Outline
        pygame.draw.rect(surface, self.COLOR_REF, (cx + 30, cy, 70, 4))
        
        # Center Dot
        pygame.draw.rect(surface, (0,0,0), (cx - 5, cy - 5, 10, 10))
        pygame.draw.rect(surface, self.COLOR_REF, (cx - 3, cy - 3, 6, 6))

    def _draw_tape(self, surface, type_, value, x, y, w, h, step, pixel_step):
        """Scrolling tape for Speed/Alt"""
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill(self.COLOR_BG_TAPE)
        surface.blit(bg, (x, y))
        
        # Border
        pygame.draw.rect(surface, self.COLOR_LINE, (x, y, w, h), 1)
        
        # Center line marker
        cy = y + h // 2
        
        # Range of values to draw
        # value is at cy
        # delta_y = (val - value) * pixel_step
        # visible if -h/2 < delta_y < h/2
        
        start_val = int(value - (h/2)/pixel_step)
        end_val = int(value + (h/2)/pixel_step)
        
        # Snap to step
        start_val = (start_val // step) * step
        
        for v in range(start_val, end_val + step, step):
            dy = (value - v) * pixel_step
            line_y = cy + dy
            
            if y <= line_y <= y+h:
                pygame.draw.line(surface, self.COLOR_LINE, (x, line_y), (x+15, line_y), 2)
                text = self.font.render(str(v), True, self.COLOR_LINE)
                surface.blit(text, (x + 20, line_y - 8))
                
        # Current Value Box
        box_h = 30
        pygame.draw.rect(surface, (0,0,0), (x, cy - box_h//2, w, box_h))
        pygame.draw.rect(surface, self.COLOR_LINE, (x, cy - box_h//2, w, box_h), 2)
        
        lbl = self.font_large.render(f"{value:.0f}", True, self.COLOR_REF)
        surface.blit(lbl, (x + 10, cy - 10))
        
        # Label
        tag = "SPD" if type_ == 'speed' else "ALT"
        lbl = self.font.render(tag, True, self.COLOR_TEXT)
        surface.blit(lbl, (x, y - 20))

    def _draw_heading_strip(self, surface, heading):
        """Top heading strip"""
        w, h = 400, 40
        x = (self.width - w) // 2
        y = 10
        
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill(self.COLOR_BG_TAPE)
        surface.blit(bg, (x, y))
        
        pygame.draw.rect(surface, self.COLOR_LINE, (x, y, w, h), 1)
        
        # Heading is 0..360
        # Visible range +/- 30 deg
        cx = x + w // 2
        pixels_per_deg = w / 60.0 # 60 degrees visible
        
        min_h = int(heading - 35)
        max_h = int(heading + 35)
        
        for h_val in range(min_h, max_h):
            if h_val % 10 == 0:
                h_norm = h_val % 360
                
                dx = (h_val - heading) * pixels_per_deg
                lx = cx + dx
                
                if x <= lx <= x+w:
                    pygame.draw.line(surface, self.COLOR_LINE, (lx, y+h-10), (lx, y+h), 2)
                    
                    if h_norm % 90 == 0:
                        dirs = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
                        txt = dirs[h_norm]
                    else:
                        txt = str(h_norm // 10)
                        
                    lbl = self.font.render(txt, True, self.COLOR_LINE)
                    surface.blit(lbl, (lx - 6, y+5))
            elif h_val % 5 == 0:
                dx = (h_val - heading) * pixels_per_deg
                lx = cx + dx
                if x <= lx <= x+w:
                    pygame.draw.line(surface, self.COLOR_LINE, (lx, y+h-5), (lx, y+h), 1)
                    
        # Center Marker
        pygame.draw.polygon(surface, self.COLOR_REF, [
            (cx, y+h+5), (cx-5, y+h+15), (cx+5, y+h+15)
        ])
        
        # Box
        lbl = self.font_large.render(f"{heading:.0f}°", True, self.COLOR_REF)
        surface.blit(lbl, (cx - 20, y + h + 20))

