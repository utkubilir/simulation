"""
Camera Inset Renderer - Bottom-right camera view for vNext

Features:
- Synthetic camera view from UAV perspective
- Detection bounding boxes with confidence
- Crosshair overlay
- Lock timer bar with 4-second progress
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.vision.validator import GeometryValidator
from .ui.ui_mode import UIMode


class CameraInsetRenderer:
    """
    Camera inset view (bottom-right corner).
    
    Shows:
    - Rendered camera view (synthetic or projected)
    - Detection bounding boxes (Thickness safe-guarded <= 3px)
    - Crosshair at center
    - Lock timer bar (0-4s)
    - Detailed HUD (State, Reason, Size%, dx/dy)
    """
    
    # Colors
    COLOR_CROSSHAIR = (0, 255, 0)
    COLOR_DETECTION = (255, 200, 0)
    COLOR_LOCKED = (255, 0, 0) # Strictly RED for locked
    COLOR_TIMER_BG = (50, 50, 50)
    COLOR_TIMER_FILL = (255, 200, 50)
    COLOR_TIMER_SUCCESS = (100, 255, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_ERROR = (255, 50, 50)
    
    def __init__(self, width: int = 320, height: int = 240):
        self.width = width
        self.height = height
        self.surface = None
        self.font = None
        self.small_font = None
        
    def init(self, parent_screen: pygame.Surface, x: int, y: int):
        """Initialize the inset renderer"""
        self.surface = parent_screen.subsurface(pygame.Rect(x, y, self.width, self.height))
        self.x = x
        self.y = y
        self.parent_screen = parent_screen
        self.font = pygame.font.Font(None, 18)
        self.small_font = pygame.font.Font(None, 14)
        
    def render(self, camera_frame: np.ndarray, detections: List[Dict], 
               tracks: List, lock_state: Dict, ui_mode: UIMode = UIMode.COMPETITION):
        """Render the camera inset"""
        # Calculate scale factors based on actual frame size if available
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        has_frame = camera_frame is not None and camera_frame.size > 0
        
        if has_frame:
            # frame shape: (H, W, 3)
            fh, fw = camera_frame.shape[:2]
            self.scale_x = self.width / fw
            self.scale_y = self.height / fh
            
        # Draw background (camera frame)
        self._draw_camera_frame(camera_frame)
        
        # Part A: Draw synthetic target shapes (projected locations)
        # ONLY if we don't have a camera frame (Sim-only mode)
        if not has_frame:
            self._draw_synthetic_targets(detections)
        
        # Draw detections/tracks
        self._draw_detections(detections, tracks, lock_state, ui_mode)
        
        # Draw crosshair
        self._draw_crosshair()
        
        # Draw lock timer bar
        self._draw_lock_timer(lock_state)
        
        # Draw lock state indicator and detailed HUD
        self._draw_hud(lock_state, ui_mode)
        
        # Draw border
        pygame.draw.rect(self.surface, (100, 100, 100), 
                        (0, 0, self.width, self.height), 2)
    
    def _draw_synthetic_targets(self, detections: List[Dict]):
        """Draw realistic UAV silhouettes for targets"""
        # Scale factor
        scale_x = getattr(self, 'scale_x', 1.0)
        scale_y = getattr(self, 'scale_y', 1.0)
        
        for det in detections:
            center = det.get('center')
            if not center:
                continue
                
            cx = int(center[0] * scale_x)
            cy = int(center[1] * scale_y)
            
            # Mesafe ve boyut bilgisi
            distance = det.get('distance', 100)
            bbox = det.get('bbox', (0, 0, 50, 50))
            apparent_size = max(10, int((bbox[2] - bbox[0]) * scale_x))
            
            # Haze efekti - mesafeye bağlı renk soluklaştırma
            haze_distance = 300.0  # metre
            haze_factor = min(0.6, distance / haze_distance)
            
            # Takım rengine göre temel renk
            team = det.get('team', 'red')
            if team == 'red':
                base_color = (200, 50, 50)  # Kırmızı düşman
            else:
                base_color = (50, 100, 200)  # Mavi dost
                
            # Haze uygula
            sky_color = (180, 200, 200)
            color = tuple(int(base_color[i] * (1 - haze_factor) + sky_color[i] * haze_factor) 
                         for i in range(3))
            
            # İHA silüeti çiz
            self._draw_uav_silhouette(cx, cy, apparent_size, color)
    
    def _draw_uav_silhouette(self, cx: int, cy: int, size: int, color: Tuple[int, int, int]):
        """Draw a simple UAV silhouette (body + wings)"""
        # Minimum boyut
        size = max(8, size)
        
        # Gövde (elips)
        body_w = size // 2
        body_h = size // 4
        pygame.draw.ellipse(self.surface, color, 
                           (cx - body_w, cy - body_h, body_w * 2, body_h * 2))
        
        # Kanatlar
        wing_span = size
        wing_h = max(2, size // 6)
        pygame.draw.ellipse(self.surface, color,
                           (cx - wing_span // 2, cy - wing_h // 2, wing_span, wing_h))
        
        # Kuyruk
        tail_w = size // 4
        tail_h = size // 3
        pygame.draw.ellipse(self.surface, color,
                           (cx - tail_w // 2, cy + body_h // 2, tail_w, tail_h))
        
        # Kenar çizgisi (daha açık renk)
        outline = tuple(min(255, c + 60) for c in color)
        pygame.draw.ellipse(self.surface, outline,
                           (cx - body_w, cy - body_h, body_w * 2, body_h * 2), 1)

    def _draw_camera_frame(self, frame: np.ndarray):
        """Draw the camera frame as background (Optimized)"""
        if frame is None or frame.size == 0:
            # Dark background placeholder
            self.surface.fill((30, 30, 40))
            return
        
        try:
            fh, fw = frame.shape[:2]
            
            # Assuming frame is BGR (standard OpenCV/GL read), convert to RGB for Pygame
            # [..., ::-1] creates a view with negative strides. .tobytes() handles the copy.
            rgb_data = frame[..., ::-1].tobytes()
            
            frame_surface = pygame.image.frombuffer(rgb_data, (fw, fh), 'RGB')
            
            if (fw, fh) != (self.width, self.height):
                frame_surface = pygame.transform.scale(frame_surface, (self.width, self.height))
                
            self.surface.blit(frame_surface, (0, 0))
            
        except Exception as e:
            print(f"Frame render error: {e}")
            self.surface.fill((50, 0, 0))
        
    def _draw_detections(self, detections: List[Dict], tracks: List, lock_state: Dict, ui_mode: UIMode):
        """Draw compliantly: Detection (Yellow) -> Lock (Red)"""
        locked_id = lock_state.get('target_id') if lock_state else None
        
        # Scale factor if camera resolution differs from inset size
        scale_x = getattr(self, 'scale_x', 1.0)
        scale_y = getattr(self, 'scale_y', 1.0)
        
        # 1. Draw raw detections (faint)
        for det in detections:
            bbox = det.get('bbox')
            if not bbox: continue
            
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            w = x2 - x1
            h = y2 - y1
            
            # Faint yellow for raw detection
            pygame.draw.rect(self.surface, (150, 150, 50), (x1, y1, w, h), 1)

        # 2. Draw all tracks as detections first (Yellow/White)
        for track in tracks:
            # Normalize track data
            if hasattr(track, 'bbox'):
                bbox = track.bbox
                track_id = track.id
                confidence = track.confidence
            else:
                bbox = track.get('bbox')
                track_id = track.get('id')
                confidence = track.get('confidence', 0)
                
            if bbox is None: continue
            
            # Scale bbox
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            w = x2 - x1
            h = y2 - y1
            
            # Base detection (Yellow)
            color = self.COLOR_DETECTION
            thickness = GeometryValidator.validate_thickness(2)
            
            # Draw standard detection
            pygame.draw.rect(self.surface, color, (x1, y1, w, h), thickness)
            
            # 2. If this is the LOCKED target, draw the LOCK RECT (Red) + HIT AREA
            if track_id == locked_id:
                # HIT AREA (inner, maybe same as bbox for now, but conceptually distinct)
                # We can draw it slightly offset or just overdraw with RED
                
                # LOCK RECT (Red, thick)
                lock_color = self.COLOR_LOCKED
                lock_thick = GeometryValidator.validate_thickness(3)
                
                # Draw slightly larger to encase the detection? 
                # Or just overwrite it. Requirement says "LOCK rectangle (RED)".
                pygame.draw.rect(self.surface, lock_color, (x1, y1, w, h), lock_thick)
                
                # Draw "LOCK" label
                label = f"LOCK {confidence:.0%}"
                text_surf = self.font.render(label, True, lock_color)
                self.surface.blit(text_surf, (x1, y1 - 20))
            else:
                # Standard confidence
                conf_text = f"{confidence:.0%}"
                text_surface = self.small_font.render(conf_text, True, color)
                self.surface.blit(text_surface, (x1, y1 - 15))
            
    def _draw_crosshair(self):
        """Draw crosshair at center"""
        cx, cy = self.width // 2, self.height // 2
        size = 20
        # Strict thickness
        thickness = GeometryValidator.validate_thickness(2)
        
        # Horizontal line
        pygame.draw.line(self.surface, self.COLOR_CROSSHAIR,
                        (cx - size, cy), (cx - 5, cy), thickness)
        pygame.draw.line(self.surface, self.COLOR_CROSSHAIR,
                        (cx + 5, cy), (cx + size, cy), thickness)
                        
        # Vertical line
        pygame.draw.line(self.surface, self.COLOR_CROSSHAIR,
                        (cx, cy - size), (cx, cy - 5), thickness)
        pygame.draw.line(self.surface, self.COLOR_CROSSHAIR,
                        (cx, cy + 5), (cx, cy + size), thickness)
                        
        # Center circle
        pygame.draw.circle(self.surface, self.COLOR_CROSSHAIR, (cx, cy), 5, 1)
        
    def _draw_lock_timer(self, lock_state: Dict):
        """Draw lock timer bar at bottom"""
        if not lock_state:
            return
            
        progress = lock_state.get('progress', 0)
        lock_time = lock_state.get('lock_time', 0)
        
        # Timer bar dimensions
        bar_x = 10
        bar_y = self.height - 25
        bar_width = self.width - 20
        bar_height = 15
        
        # Background
        pygame.draw.rect(self.surface, self.COLOR_TIMER_BG,
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Fill
        if progress > 0:
            fill_width = int(bar_width * min(progress, 1.0))
            fill_color = self.COLOR_TIMER_SUCCESS if progress >= 1.0 else self.COLOR_TIMER_FILL
            pygame.draw.rect(self.surface, fill_color,
                            (bar_x, bar_y, fill_width, bar_height))
                            
        # Border
        pygame.draw.rect(self.surface, (100, 100, 100),
                        (bar_x, bar_y, bar_width, bar_height), 1)
                        
        # Time text
        time_text = f"{lock_time:.1f}s"
        text_surface = self.font.render(time_text, True, self.COLOR_TEXT)
        text_x = bar_x + (bar_width - text_surface.get_width()) // 2
        self.surface.blit(text_surface, (text_x, bar_y + 1))

    def _draw_hud(self, lock_state: Dict, ui_mode: UIMode):
        """Draw HUD info. Minimal in COMPETITION, Detailed in DEBUG."""
        if not lock_state:
            return
            
        state = lock_state.get('state', 'idle').upper()
        reason = lock_state.get('reason_invalid')
        
        # Audit Metrics
        dx = lock_state.get('dx', 0)
        dy = lock_state.get('dy', 0)
        cov_w = lock_state.get('coverage_w', 0)
        cov_h = lock_state.get('coverage_h', 0)
        
        # Booleans
        size_ok = lock_state.get('size_ok', False)
        center_ok = lock_state.get('center_ok', False)
        # conf_ok = lock_state.get('confidence_ok', False) # Optional to show
        
        # Position
        x, y = 10, 10
        line_height = 14
        
        if state == 'IDLE':
            color = (150, 150, 150)
            icon = "○"
        elif state == 'LOCKING':
            color = (255, 200, 50)
            icon = "◐"
        elif state == 'SUCCESS':
            color = (0, 255, 0) # Green for success text
            icon = "●"
        else:
            color = (150, 150, 150)
            icon = "?"
            
        # Lines to draw
        # In COMPETITION mode, show simple status label
        if ui_mode == UIMode.COMPETITION:
            # Simple top-center label? Or keep it here but minimal.
            # User request: "State label: SEARCHING / LOCKING / LOCKED"
            # Remove icons/reasons
            lines = [(f"{state}", color)]
        else:
            # DEBUG / TRAINING
            lines = [
                (f"{icon} {state}", color),
            ]
            
            if reason:
                lines.append((f"REASON: {reason}", self.COLOR_ERROR))
                
            # Geometry info (only if tracking something or reason given)
            if state != 'IDLE' or reason:
                # Format: "dX: -4.2  dY: 10.5"
                lines.append((f"dXY: {dx:.1f}, {dy:.1f}", (200, 200, 200)))
                # Format: "Cov: 6.5% / 4.2%"
                lines.append((f"Cov: {cov_w:.1%} W / {cov_h:.1%} H", (200, 200, 200)))
                
                # Flags: [SIZE] [CENTER] [CONF]
                s_chk = "OK" if size_ok else "FAIL"
                c_chk = "OK" if center_ok else "FAIL"
                lines.append((f"Size: {s_chk}  Center: {c_chk}", (180, 180, 180)))

        # Draw background
        total_h = len(lines) * line_height + 4
        # approximate width
        max_w = 140 
        
        bg_rect = pygame.Rect(x - 2, y - 2, max_w, total_h)
        pygame.draw.rect(self.surface, (0, 0, 0, 180), bg_rect)
        
        # Draw lines
        for i, (text, col) in enumerate(lines):
            surf = self.small_font.render(text, True, col)
            self.surface.blit(surf, (x, y + i * line_height))
        
    def flash_success(self):
        """Flash green on successful lock"""
        # Create green overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 255, 0, 50))
        self.surface.blit(overlay, (0, 0))
        
    def flash_lost(self):
        """Flash red when lock is lost"""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((255, 0, 0, 50))
        self.surface.blit(overlay, (0, 0))
