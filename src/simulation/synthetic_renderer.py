"""
Sentetik Kamera Frame Oluşturucu

Gerçekçi sentetik kamera görüntüsü oluşturur.
Tüm post-processing efektlerini içerir:
- Gökyüzü ve zemin render
- Hedef İHA render
- Lens distorsiyon
- Motion blur
- Chromatic aberration
- Lens flare
- Vignette ve haze
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


class SyntheticFrameRenderer:
    """
    Gelişmiş sentetik kamera görüntüsü oluşturucu.
    
    Post-processing pipeline:
    1. Sky/ground render
    2. Target render
    3. Lens distortion
    4. Motion blur
    5. Chromatic aberration
    6. Vignette & haze
    7. Lens flare (sun direction)
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Resolution
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        
        # Lens distortion
        self.distortion_enabled = config.get('distortion_enabled', True)
        self.k1 = config.get('k1', -0.1)
        self.k2 = config.get('k2', 0.02)
        
        # Motion blur
        self.motion_blur_enabled = config.get('motion_blur', False)
        self.blur_strength = config.get('blur_strength', 5)
        
        # Chromatic aberration
        self.chromatic_aberration_enabled = config.get('chromatic_aberration', True)
        self.chromatic_strength = config.get('chromatic_strength', 2.0)
        
        # Vignette
        self.vignette_enabled = config.get('vignette_enabled', True)
        self.vignette_strength = config.get('vignette_strength', 0.4)
        
        # Haze
        self.haze_enabled = config.get('haze_enabled', True)
        self.haze_distance = config.get('haze_distance', 300.0)
        
        # Lens flare
        self.lens_flare_enabled = config.get('lens_flare_enabled', False)
        self.sun_direction = np.array([0.5, 0.3, 0.8])  # Normalized sun direction
        
        # Pre-compute vignette mask
        self._vignette_mask = None
        self._compute_vignette_mask()
        
    def _compute_vignette_mask(self):
        """Pre-compute vignette mask for efficiency"""
        Y, X = np.ogrid[:self.height, :self.width]
        cx, cy = self.width / 2, self.height / 2
        
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist / max_dist
        
        self._vignette_mask = 1 - (dist_norm ** 2) * self.vignette_strength
        self._vignette_mask = np.clip(self._vignette_mask, 0.5, 1.0)
        
    def render_frame(self, camera_pos: np.ndarray,
                     camera_orient: np.ndarray,
                     uav_states: List[Dict],
                     own_velocity: np.ndarray = None,
                     focal_length: float = 554.0) -> np.ndarray:
        """
        Sentetik kamera frame'i oluştur.
        
        Args:
            camera_pos: Kamera pozisyonu
            camera_orient: Kamera oryantasyonu [roll, pitch, yaw]
            uav_states: Hedef İHA durumları listesi
            own_velocity: Kendi hız vektörü (motion blur için)
            focal_length: Focal length in pixels
            
        Returns:
            RGB numpy array (h, w, 3)
        """
        # 1. Render sky and ground
        frame = self._render_sky_ground(camera_orient)
        
        # 2. Render target UAVs
        frame = self._render_targets(frame, camera_pos, camera_orient, 
                                     uav_states, focal_length)
        
        # 3. Apply lens distortion
        if self.distortion_enabled:
            frame = self._apply_lens_distortion(frame)
            
        # 4. Apply motion blur
        if self.motion_blur_enabled and own_velocity is not None:
            frame = self._apply_motion_blur(frame, own_velocity, camera_orient)
            
        # 5. Apply chromatic aberration
        if self.chromatic_aberration_enabled:
            frame = self._apply_chromatic_aberration(frame)
            
        # 6. Apply vignette
        if self.vignette_enabled:
            frame = self._apply_vignette(frame)
            
        # 7. Apply lens flare (if sun is visible)
        if self.lens_flare_enabled:
            frame = self._apply_lens_flare(frame, camera_orient)
            
        return frame
        
    def _render_sky_ground(self, camera_orient: np.ndarray) -> np.ndarray:
        """Render dynamic sky and ground based on camera orientation"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        pitch_deg = np.degrees(camera_orient[1])
        roll_deg = np.degrees(camera_orient[0])
        
        # Calculate horizon position based on pitch
        horizon_offset = int(pitch_deg * self.height / 60)
        base_horizon = self.height // 2 + horizon_offset
        
        for y in range(self.height):
            # Apply roll to horizon calculation
            roll_offset = 0
            if abs(roll_deg) > 1:
                x_center = self.width / 2
                roll_offset = int(np.tan(np.radians(roll_deg)) * (x_center - self.width/2))
            
            effective_horizon = base_horizon + roll_offset
            
            if y < effective_horizon:
                # Sky - blue gradient (darker at top)
                ratio = y / max(effective_horizon, 1)
                # Atmospheric scattering - more blue at horizon
                b = int(200 + ratio * 55)  # 200 -> 255
                g = int(150 + ratio * 70)  # 150 -> 220
                r = int(120 + ratio * 80)  # 120 -> 200
            else:
                # Ground - brownish green gradient
                ground_ratio = (y - effective_horizon) / max(self.height - effective_horizon, 1)
                b = int(70 - ground_ratio * 20)
                g = int(100 - ground_ratio * 30)
                r = int(80 - ground_ratio * 20)
                
            frame[y, :] = [b, g, r]
            
        # Draw horizon line
        horizon_y = np.clip(base_horizon, 0, self.height - 1)
        cv2.line(frame, (0, horizon_y), (self.width, horizon_y), (180, 180, 180), 1)
        
        return frame
        
    def _render_targets(self, frame: np.ndarray,
                        camera_pos: np.ndarray,
                        camera_orient: np.ndarray,
                        uav_states: List[Dict],
                        focal_length: float) -> np.ndarray:
        """Render target UAVs with perspective"""
        
        for uav in uav_states:
            uav_pos = np.array(uav['position'])
            distance = np.linalg.norm(uav_pos - camera_pos)
            
            if distance < 5 or distance > 500:
                continue
                
            # Project to screen
            screen_pos = self._project_point(uav_pos, camera_pos, camera_orient, focal_length)
            if screen_pos is None:
                continue
                
            x, y = int(screen_pos[0]), int(screen_pos[1])
            if not (10 < x < self.width - 10 and 10 < y < self.height - 10):
                continue
                
            # Calculate apparent size
            uav_size = uav.get('size', 2.0)
            apparent_size = int(focal_length * uav_size / distance)
            apparent_size = max(5, min(apparent_size, 100))
            
            # Haze effect - fade color with distance
            if self.haze_enabled:
                haze_factor = min(0.7, distance / self.haze_distance)
            else:
                haze_factor = 0
                
            # Team color
            team = uav.get('team', 'red')
            if team == 'red':
                base_color = np.array([50, 50, 200])  # BGR
            else:
                base_color = np.array([200, 100, 50])  # BGR
                
            # Apply haze
            sky_color = np.array([200, 180, 160])
            color = (base_color * (1 - haze_factor) + sky_color * haze_factor).astype(np.uint8)
            color = tuple(map(int, color))
            
            # Draw UAV silhouette
            self._draw_uav_silhouette(frame, x, y, apparent_size, color)
            
        return frame
        
    def _draw_uav_silhouette(self, frame: np.ndarray, cx: int, cy: int, 
                             size: int, color: Tuple[int, int, int]):
        """Draw UAV silhouette with body, wings, and tail"""
        size = max(5, size)
        
        # Body (ellipse)
        body_w = size // 2
        body_h = size // 4
        cv2.ellipse(frame, (cx, cy), (body_w, max(2, body_h)), 0, 0, 360, color, -1)
        
        # Wings
        wing_span = size
        wing_h = max(2, size // 6)
        cv2.ellipse(frame, (cx, cy), (wing_span // 2, wing_h), 0, 0, 360, color, -1)
        
        # Tail
        tail_w = size // 4
        tail_h = size // 3
        cv2.ellipse(frame, (cx, cy + body_h // 2), (max(2, tail_w // 2), max(2, tail_h)), 
                   0, 0, 360, color, -1)
        
        # Outline
        outline = tuple(min(255, c + 40) for c in color)
        cv2.ellipse(frame, (cx, cy), (body_w, max(2, body_h)), 0, 0, 360, outline, 1)
        
    def _project_point(self, world_point: np.ndarray,
                       camera_pos: np.ndarray,
                       camera_orient: np.ndarray,
                       focal_length: float) -> Optional[Tuple[float, float]]:
        """Project 3D world point to 2D screen coordinates"""
        rel_pos = world_point - camera_pos
        
        # Camera rotation matrix
        R = self._rotation_matrix(*camera_orient)
        cam_coords = R.T @ rel_pos
        
        # Behind camera check
        if cam_coords[0] <= 0.1:
            return None
            
        # Perspective projection
        x = focal_length * cam_coords[1] / cam_coords[0] + self.width / 2
        y = focal_length * cam_coords[2] / cam_coords[0] + self.height / 2
        
        return (x, y)
        
    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Euler angles to rotation matrix (ZYX convention)"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        return np.array([
            [cy*cp, sy*cp, -sp],
            [cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr],
            [cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr]
        ])
        
    def _apply_lens_distortion(self, frame: np.ndarray) -> np.ndarray:
        """Apply barrel/pincushion lens distortion using OpenCV"""
        h, w = frame.shape[:2]
        
        # Camera matrix
        fx = fy = w / (2 * np.tan(np.radians(30)))  # Approximate
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([self.k1, self.k2, 0, 0, 0], dtype=np.float32)
        
        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0.5, (w, h)
        )
        
        # Apply undistortion (we want distortion, so we reverse the effect conceptually)
        # Actually, we need to apply the inverse - let's create distorted coordinates
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )
        
        distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        
        return distorted
        
    def _apply_motion_blur(self, frame: np.ndarray, 
                           velocity: np.ndarray,
                           camera_orient: np.ndarray) -> np.ndarray:
        """Apply directional motion blur based on velocity"""
        speed = np.linalg.norm(velocity)
        if speed < 5:  # Minimum speed for blur
            return frame
            
        # Calculate blur direction in screen space
        yaw = camera_orient[2]
        # Velocity direction relative to camera
        rel_vel = np.array([
            velocity[0] * np.cos(yaw) + velocity[1] * np.sin(yaw),
            -velocity[0] * np.sin(yaw) + velocity[1] * np.cos(yaw)
        ])
        
        # Blur kernel size based on speed
        blur_size = min(int(speed / 10) + 1, self.blur_strength)
        blur_size = max(3, blur_size if blur_size % 2 == 1 else blur_size + 1)
        
        # Create motion blur kernel
        kernel_size = blur_size
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        # Calculate kernel angle from relative velocity
        if np.linalg.norm(rel_vel) > 0.1:
            angle = np.arctan2(rel_vel[1], rel_vel[0])
        else:
            angle = 0
            
        # Line kernel at angle
        mid = kernel_size // 2
        for i in range(kernel_size):
            x = int(mid + (i - mid) * np.cos(angle))
            y = int(mid + (i - mid) * np.sin(angle))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
                
        kernel /= np.sum(kernel) if np.sum(kernel) > 0 else 1
        
        # Apply kernel
        blurred = cv2.filter2D(frame, -1, kernel)
        
        # Blend with original (partial blur)
        alpha = min(0.5, speed / 100)  # More blur at higher speeds
        result = cv2.addWeighted(frame, 1 - alpha, blurred, alpha, 0)
        
        return result
        
    def _apply_chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Apply chromatic aberration (color fringing at edges)"""
        h, w = frame.shape[:2]
        
        # Split channels
        b, g, r = cv2.split(frame)
        
        # Calculate displacement maps for R and B channels
        # More displacement at edges
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        
        # Radial distance (normalized)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist / max_dist
        
        # Displacement (stronger at edges)
        displacement = dist_norm * self.chromatic_strength
        
        # Direction vectors from center
        dx = (X - cx) / (dist + 1e-6)
        dy = (Y - cy) / (dist + 1e-6)
        
        # Red channel - shift outward
        map_x_r = (X + dx * displacement).astype(np.float32)
        map_y_r = (Y + dy * displacement).astype(np.float32)
        r_shifted = cv2.remap(r, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Blue channel - shift inward (opposite)
        map_x_b = (X - dx * displacement * 0.5).astype(np.float32)
        map_y_b = (Y - dy * displacement * 0.5).astype(np.float32)
        b_shifted = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Merge back
        result = cv2.merge([b_shifted, g, r_shifted])
        
        return result
        
    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Apply vignette effect (edge darkening)"""
        if self._vignette_mask is None:
            self._compute_vignette_mask()
            
        # Apply mask to all channels
        result = (frame * self._vignette_mask[:, :, np.newaxis]).astype(np.uint8)
        
        return result
        
    def _apply_lens_flare(self, frame: np.ndarray, 
                          camera_orient: np.ndarray) -> np.ndarray:
        """Apply lens flare effect when looking toward sun"""
        
        # Check if sun is in view direction
        yaw, pitch = camera_orient[2], camera_orient[1]
        camera_forward = np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch)
        ])
        
        # Dot product with sun direction
        sun_dot = np.dot(camera_forward, self.sun_direction / np.linalg.norm(self.sun_direction))
        
        if sun_dot < 0.5:  # Sun not in view
            return frame
            
        # Calculate flare intensity
        flare_intensity = (sun_dot - 0.5) * 2  # 0 to 1
        
        # Project sun position to screen
        # Simplified: assume sun is at infinity in sun_direction
        sun_screen_x = int(self.width / 2 + self.sun_direction[1] * self.width * 0.3)
        sun_screen_y = int(self.height / 2 - self.sun_direction[2] * self.height * 0.3)
        
        # Create flare overlay
        flare_overlay = frame.copy()
        
        # Main sun glow
        glow_radius = int(50 * flare_intensity)
        if 0 < sun_screen_x < self.width and 0 < sun_screen_y < self.height:
            cv2.circle(flare_overlay, (sun_screen_x, sun_screen_y), glow_radius, 
                      (200, 220, 255), -1)
            
            # Smaller flare artifacts (ghosts)
            for i in range(3):
                ghost_x = int(self.width / 2 + (sun_screen_x - self.width / 2) * (0.3 + i * 0.2))
                ghost_y = int(self.height / 2 + (sun_screen_y - self.height / 2) * (0.3 + i * 0.2))
                ghost_radius = int(20 * (1 - i * 0.3) * flare_intensity)
                ghost_color = (180 - i * 30, 200 - i * 20, 240 - i * 10)
                cv2.circle(flare_overlay, (ghost_x, ghost_y), ghost_radius, ghost_color, -1)
        
        # Blend flare with original
        alpha = 0.3 * flare_intensity
        result = cv2.addWeighted(frame, 1 - alpha, flare_overlay, alpha, 0)
        
        return result
        
    def set_resolution(self, width: int, height: int):
        """Update resolution and recompute masks"""
        self.width = width
        self.height = height
        self._compute_vignette_mask()
