"""
Autonomous Combat Logic for Fixed-Wing UAVs.

State Machine:
- SEARCH: Patrol area to find targets.
- TRACK: Fly towards detected target.
- LOCK: Fine maneuvering for lock-on.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

class CombatState(Enum):
    SEARCH = "search"
    TRACK = "track"
    LOCK = "lock"
    EVADE = "evade"  # Future extension

@dataclass
class CombatConfig:
    engagement_distance: float = 300.0  # m
    search_altitude: float = 100.0      # m
    search_speed: float = 25.0          # m/s
    track_speed_boost: float = 1.2      # mult
    lock_margin_h: float = 0.4
    lock_margin_v: float = 0.4
    geofence_limit: float = 1900.0      # 2000m world sınırı için 1900m'de dön
    evade_trigger_dist: float = 150.0   # 150m altında biri varsa kaçınma başlat
    world_center: tuple = (1000.0, 1000.0)  # Dünya merkezi (x, y)

class SearchPattern:
    """Generates waypoints for area search."""
    
    @staticmethod
    def lawnmower(center: np.ndarray, width: float, length: float, spacing: float) -> List[np.ndarray]:
        """Generates lawnmower pattern waypoints."""
        waypoints = []
        x_start = center[0] - width / 2
        y_start = center[1] - length / 2
        z = center[2]
        
        rows = int(length / spacing)
        for i in range(rows):
            y = y_start + i * spacing
            x_left = x_start
            x_right = x_start + width
            
            if i % 2 == 0:
                waypoints.append(np.array([x_left, y, z]))
                waypoints.append(np.array([x_right, y, z]))
            else:
                waypoints.append(np.array([x_right, y, z]))
                waypoints.append(np.array([x_left, y, z]))
                
        return waypoints

class PursuitLogic:
    """Guidance logic for interception."""
    
    @staticmethod
    def pure_pursuit(my_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """Returns heading to fly directly at target."""
        diff = target_pos - my_pos
        return np.arctan2(diff[1], diff[0])
        
    @staticmethod
    def proportional_navigation(my_pos: np.ndarray, my_vel: np.ndarray, 
                                target_pos: np.ndarray, target_vel: np.ndarray,
                                n: float = 3.0) -> Tuple[float, float]:
        """
        Proportional Navigation Guidance.
        
        Args:
            n: Navigation constant (usually 3-5)
            
        Returns:
            (heading_cmd, pitch_cmd)
        """
        # Line of Sight (LOS) vector
        r = target_pos - my_pos
        distance = np.linalg.norm(r)
        
        # Relative velocity
        v_rel = target_vel - my_vel
        
        # Rotation rate of LOS vector (omega)
        # omega = (r x v_rel) / |r|^2
        cross_prod = np.cross(r, v_rel)
        omega = cross_prod / (distance**2 + 1e-6)
        
        # Acceleration command (a_cmd = N * |v_closing| * omega)
        # For fixed wing, we convert this to heading/pitch changes roughly
        # Simplified: We just want to effectively adjust our heading rate.
        
        # Standard PN implementation for heading/pitch rate:
        # We need the LOS rate in azimuth and elevation.
        # This implementation returns a desired heading, not rate, 
        # so we'll stick to a Lead Pursuit approximation which is easier for the Autopilot PID.
        
        time_to_go = distance / (np.linalg.norm(v_rel) + 1e-6)
        predicted_pos = target_pos + target_vel * time_to_go
        
        diff = predicted_pos - my_pos
        heading_cmd = np.arctan2(diff[1], diff[0])
        
        dist_2d = np.linalg.norm(diff[:2])
        pitch_cmd = np.arctan2(diff[2], dist_2d)
        
        return heading_cmd, pitch_cmd


class AdvancedPursuitLogic:
    """
    Gelişmiş takip algoritmaları
    
    - Lead pursuit: Hedefin geleceği noktaya yönel
    - Collision course: Çarpışma noktasına yönel
    - Pure pursuit: Direkt hedefe yönel
    - Proportional Navigation: Açısal hız tabanlı güdüm
    """
    
    @staticmethod
    def lead_pursuit(my_pos: np.ndarray, my_vel: np.ndarray,
                     target_pos: np.ndarray, target_vel: np.ndarray,
                     lead_time: float = 2.0) -> Tuple[float, float]:
        """
        Lead Pursuit - Hedefin gelecek konumuna yönel
        
        Args:
            my_pos: Kendi pozisyonumuz
            my_vel: Kendi hızımız  
            target_pos: Hedef pozisyonu
            target_vel: Hedef hızı
            lead_time: Kaç saniye ileriye bakılacak
            
        Returns:
            (heading_cmd, pitch_cmd) derece cinsinden
        """
        # Hedefin gelecek pozisyonunu tahmin et
        if target_vel is None:
            target_vel = np.zeros(3)
        elif len(target_vel) == 2:
            target_vel = np.array([target_vel[0], target_vel[1], 0.0])
            
        future_target_pos = np.array(target_pos) + np.array(target_vel) * lead_time
        
        # Bu noktaya yönel
        to_target = future_target_pos - my_pos
        dist = np.linalg.norm(to_target[:2])
        
        heading_rad = np.arctan2(to_target[1], to_target[0])
        pitch_rad = np.arctan2(to_target[2], dist) if dist > 0 else 0
        
        return (np.degrees(heading_rad), np.degrees(pitch_rad))
    
    @staticmethod
    def collision_course(my_pos: np.ndarray, my_speed: float,
                         target_pos: np.ndarray, target_vel: np.ndarray) -> Tuple[float, float]:
        """
        Çarpışma kursu - En kısa sürede kesişim noktasına yönel
        
        Returns:
            (heading_cmd, pitch_cmd) derece cinsinden
        """
        if target_vel is None:
            target_vel = np.zeros(3)
        elif len(target_vel) == 2:
            target_vel = np.array([target_vel[0], target_vel[1], 0.0])
            
        # Mesafe
        to_target = np.array(target_pos) - np.array(my_pos)
        distance = np.linalg.norm(to_target)
        
        if distance < 1:
            return (0.0, 0.0)
            
        # Intercept noktası iteratif hesaplama
        t_intercept = distance / (my_speed + 1e-6)
        
        for _ in range(3):  # 3 iterasyon yeterli
            intercept_point = np.array(target_pos) + np.array(target_vel) * t_intercept
            new_dist = np.linalg.norm(intercept_point - my_pos)
            t_intercept = new_dist / (my_speed + 1e-6)
            
        to_intercept = intercept_point - my_pos
        
        heading_rad = np.arctan2(to_intercept[1], to_intercept[0])
        dist_2d = np.linalg.norm(to_intercept[:2])
        pitch_rad = np.arctan2(to_intercept[2], dist_2d) if dist_2d > 0 else 0
        
        return (np.degrees(heading_rad), np.degrees(pitch_rad))


class TargetSelector:
    """
    Akıllı hedef seçimi
    
    Şartname kuralı: Aynı hedefe art arda kilitlenemez
    (Madde 6.1.1: Bir İHA'ya kilitlendikten sonra aynı İHA'ya 
    kilitlenmek için en az bir farklı İHA'ya kilitlenmek gereklidir)
    """
    
    def __init__(self):
        self.last_locked_target_id = None
        self.lock_history: List[Tuple[int, float]] = []  # [(target_id, timestamp), ...]
        self.successful_locks = set()  # Başarılı kilitlenme yapılan hedefler
        
    def select_target(self, available_tracks: List, 
                      my_position: np.ndarray) -> Optional[object]:
        """
        En uygun hedefi seç
        
        Kriterler:
        1. Son kilitlenilen hedef olmamalı (şartname zorunluluğu)
        2. Onaylanmış (confirmed) hedefler tercih edilir
        3. En yakın mesafedeki hedef tercih edilir
        
        Args:
            available_tracks: Takip edilen hedef listesi
            my_position: Kendi pozisyonumuz
            
        Returns:
            En uygun hedef veya None
        """
        if not available_tracks:
            return None
            
        # Şartname: Son kilitlenilen hedefi filtrele
        valid_targets = [
            t for t in available_tracks 
            if t.id != self.last_locked_target_id
        ]
        
        if not valid_targets:
            # Tüm hedefler son kilitlenilen ise, yine de en yakını seç
            # (bu durumda kilitlenme geçersiz sayılacak)
            valid_targets = available_tracks
            
        # Onaylanmış olanları öncelikle
        confirmed = [t for t in valid_targets if getattr(t, 'is_confirmed', False)]
        if confirmed:
            valid_targets = confirmed
            
        # Mesafeye göre sırala
        def get_distance(track):
            if hasattr(track, 'world_pos') and track.world_pos is not None:
                return np.linalg.norm(np.array(track.world_pos[:2]) - my_position[:2])
            return float('inf')
            
        valid_targets.sort(key=get_distance)
        
        return valid_targets[0] if valid_targets else None
    
    def can_lock_target(self, target_id: int) -> bool:
        """
        Bu hedefe kilitlenebilir mi?
        
        Şartname: Aynı hedefe art arda kilitlenemez
        """
        return target_id != self.last_locked_target_id
        
    def register_lock(self, target_id: int, timestamp: float):
        """Başarılı kilitlenmeyi kaydet"""
        self.last_locked_target_id = target_id
        self.lock_history.append((target_id, timestamp))
        self.successful_locks.add(target_id)
        
    def get_lock_count(self) -> int:
        """Toplam başarılı kilitlenme sayısı"""
        return len(self.lock_history)
    
    def get_unique_lock_count(self) -> int:
        """Farklı hedeflere yapılan kilitlenme sayısı"""
        return len(self.successful_locks)
    
    def reset(self):
        """Seçici durumunu sıfırla"""
        self.last_locked_target_id = None
        self.lock_history.clear()
        self.successful_locks.clear()


class VisualServo:
    """
    Visual Servoing Logic.
    Calculates control commands based on 2D image error.
    """
    
    def __init__(self, config: CombatConfig = None):
        self.config = config or CombatConfig()
        # PID Constants for visual servoing
        # Increased gains to ensure tight lock alignment (correction for 1-degree errors)
        self.kp_h = 4.0  # Horizontal (Yaw/Roll) gain
        self.kp_v = 4.0  # Vertical (Pitch) gain
        
    def calculate_commands(self, target_center: Tuple[float, float], frame_size: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate roll and pitch commands to center the target.
        
        Args:
            target_center: (x, y) pixel coordinates
            frame_size: (width, height)
            
        Returns:
            (roll_cmd, pitch_cmd) in radians
        """
        w, h = frame_size
        cx, cy = w / 2, h / 2
        
        # Normalized error (-1 to 1)
        err_x = (target_center[0] - cx) / (w / 2)
        err_y = (target_center[1] - cy) / (h / 2)
        
        # Control Logic:
        # To correct x-error (horizontal), we roll/turn towards it.
        # To correct y-error (vertical), we pitch up/down.
        # Note: Y-axis in image is usually down-positive.
        # If target is below center (positive err_y), we need to pitch DOWN (negative pitch).
        # If target is above center (negative err_y), we need to pitch UP (positive pitch).
        
        roll_cmd = self.kp_h * err_x
        pitch_cmd = -self.kp_v * err_y
        
        return roll_cmd, pitch_cmd


class AdvancedVisualServo:
    """
    Gelişmiş görsel servo kontrol
    
    PID tabanlı görüntü merkezleme ile daha stabil kilitlenme.
    """
    
    def __init__(self, config: CombatConfig = None):
        self.config = config or CombatConfig()
        
        # PID kazançları
        self.kp_h = 3.0
        self.ki_h = 0.1
        self.kd_h = 0.5
        
        self.kp_v = 2.5
        self.ki_v = 0.08
        self.kd_v = 0.4
        
        # PID durumları
        self.integral_h = 0.0
        self.integral_v = 0.0
        self.prev_error_h = 0.0
        self.prev_error_v = 0.0
        
        # Integral limitleri
        self.integral_limit = 0.5
        
        # Hedef merkez (normalize)
        self.target_center = (0.5, 0.5)  # Ekran merkezi
        
    def calculate_commands(self, detection_center: Tuple[float, float],
                          frame_size: Tuple[int, int],
                          dt: float = 0.016) -> Tuple[float, float]:
        """
        Roll ve pitch komutlarını hesapla
        
        Args:
            detection_center: Tespit merkezi (x, y) piksel
            frame_size: Ekran boyutu (w, h)
            dt: Zaman adımı
            
        Returns:
            (roll_cmd, pitch_cmd) -1 ile 1 arası
        """
        w, h = frame_size
        
        # Normalize et
        norm_x = detection_center[0] / w
        norm_y = detection_center[1] / h
        
        # Hata hesapla
        error_h = self.target_center[0] - norm_x
        error_v = self.target_center[1] - norm_y
        
        # PID - Yatay (Roll)
        self.integral_h += error_h * dt
        self.integral_h = np.clip(self.integral_h, -self.integral_limit, self.integral_limit)
        derivative_h = (error_h - self.prev_error_h) / dt if dt > 0 else 0
        
        roll_cmd = (self.kp_h * error_h + 
                   self.ki_h * self.integral_h + 
                   self.kd_h * derivative_h)
        
        # PID - Dikey (Pitch)
        self.integral_v += error_v * dt
        self.integral_v = np.clip(self.integral_v, -self.integral_limit, self.integral_limit)
        derivative_v = (error_v - self.prev_error_v) / dt if dt > 0 else 0
        
        pitch_cmd = (self.kp_v * error_v + 
                    self.ki_v * self.integral_v + 
                    self.kd_v * derivative_v)
        
        # Önceki hataları güncelle
        self.prev_error_h = error_h
        self.prev_error_v = error_v
        
        return (np.clip(roll_cmd, -1, 1), np.clip(-pitch_cmd, -1, 1))
    
    def reset(self):
        """PID durumlarını sıfırla"""
        self.integral_h = 0.0
        self.integral_v = 0.0
        self.prev_error_h = 0.0
        self.prev_error_v = 0.0


class CombatStateManager:
    """
    Manages high-level combat behavior.
    
    Inputs:
    - Current state (position, heading, etc.)
    - Detected targets (from Tracker)
    
    Outputs:
    - Autopilot commands (Mode, Target Heading/Alt/Speed)
    """
    
    def __init__(self, config: CombatConfig = None):
        self.config = config or CombatConfig()
        self.visual_servo = VisualServo(self.config)
        self.state = CombatState.SEARCH
        self.target_id: Optional[int] = None
        self.search_waypoints: List[np.ndarray] = []
        self.current_wp_index = 0
        
    def update(self, uav_state: Dict, tracks: List) -> Dict:
        """
        Decide next action based on state and tracks.
        
        Returns dict with keys for Autopilot:
        - mode: AutopilotMode
        - params: Dict of set values (heading, alt, etc.)
        """
        
        # 1. Update State Transitions
        self._check_transitions(uav_state, tracks)
        
        # 2. Execute State Logic
        if self.state == CombatState.SEARCH:
            return self._bbox_search(uav_state)
            
        elif self.state == CombatState.TRACK:
            target = self._get_target(tracks)
            if target:
                return self._bbox_track(uav_state, target)
                
        elif self.state == CombatState.LOCK:
            target = self._get_target(tracks)
            if target:
                return self._bbox_lock(uav_state, target)
                
        elif self.state == CombatState.EVADE:
            return self._bbox_evade(uav_state)

        # Default fallback
        return {'mode': 'manual', 'params': {}}

    def _bbox_evade(self, uav_state):
        """Kaçınma manevrası: Sert dönüş yap ve merkeze yönel."""
        pos = np.array(uav_state['position'])
        center_x, center_y = self.config.world_center
        center = np.array([center_x, center_y, pos[2]])  # Mevcut irtifayı koru
        diff = center - pos
        
        # Merkeze dönüş yönü (hedef heading)
        target_heading = np.arctan2(diff[1], diff[0])
        
        # Mevcut heading (radyan olarak)
        current_heading = np.radians(uav_state.get('heading', 0))
        
        # Heading error hesapla ve [-π, π] aralığına normalize et
        heading_error = target_heading - current_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        
        # Heading error'a göre roll komutu (proportional control)
        # Pozitif error = sola dön = pozitif roll
        roll_cmd = np.clip(heading_error * 0.8, -0.8, 0.8)
        
        return {
            'mode': 'direct',
            'params': {
                'roll': roll_cmd,
                'pitch': 0.1,  # Hafif tırmanış
                'throttle': 1.0  # Tam gaz
            }
        }

        
    def _check_transitions(self, uav_state, tracks):
        """Update state machine based on environment."""
        target = self._get_target(tracks)
        
        # Default priority: confirmed targets close to center
        if not target and tracks:
            confirmed = [t for t in tracks if t.is_confirmed]
            if confirmed:
                # Pick closest
                self.target_id = confirmed[0].id
                target = confirmed[0]
        
        if target:
            # Distance check
            dist = getattr(target, 'distance', float('inf'))
            if hasattr(target, 'world_pos'):
                 dist = np.linalg.norm(np.array(target.world_pos) - np.array(uav_state['position']))

            # 1. State Transitions based on distance
            if self.state == CombatState.SEARCH:
                self.state = CombatState.TRACK
            
            elif self.state == CombatState.TRACK:
                # Transition to LOCK if close enough
                if dist < self.config.engagement_distance:
                     self.state = CombatState.LOCK
                     
            elif self.state == CombatState.LOCK:
                 # Lose lock if too far
                 if dist > self.config.engagement_distance * 1.5: # Hysteresis
                      self.state = CombatState.TRACK
            
            # 2. Check for Evade (Overriding priority)
            # Only evade if target is close AND potentially threatening (e.g. behind us or head-on collision risk?)
            # Test expects TRACK at 100m. Config evade is 150m.
            # We should only EVADE if target is BEHIND us (Rear Hemisphere).
            
            # Calculate bearing to target
            my_pos = np.array(uav_state['position'])
            target_pos = None
            if hasattr(target, 'world_pos') and target.world_pos is not None:
                target_pos = np.array(target.world_pos)
            
            if target_pos is not None:
                diff = target_pos - my_pos
                bearing_rad = np.arctan2(diff[1], diff[0])
                bearing_deg = np.degrees(bearing_rad)
                
                my_heading = uav_state.get('heading', 0)
                
                # Relative angle in [-180, 180]
                rel_angle = (bearing_deg - my_heading + 180) % 360 - 180
                
                # If target is in rear hemisphere (abs(rel_angle) > 90), EVADE
                # If target is in front (abs(rel_angle) <= 90), we ATTACK (TRACK/LOCK)
                if dist < self.config.evade_trigger_dist and abs(rel_angle) > 90:
                    self.state = CombatState.EVADE
            
        else:
            self.state = CombatState.SEARCH
            
        # 3. Check for Geofence (Dünya dışına çıkıyor muyuz?)
        pos = uav_state['position']
        center_x, center_y = self.config.world_center
        if (abs(pos[0] - center_x) > self.config.geofence_limit / 2 or 
            abs(pos[1] - center_y) > self.config.geofence_limit / 2):
            self.state = CombatState.EVADE  # Kaçınma manevrası ile merkeze dön
            
    def _get_target(self, tracks):
        if self.target_id is None:
            return None
        for t in tracks:
            if t.id == self.target_id:
                return t
        return None

    def _bbox_search(self, uav_state):
        # Generate waypoints if empty
        if not self.search_waypoints:
            pos = np.array(uav_state['position'])
            self.search_waypoints = SearchPattern.lawnmower(
                center=np.array([500, 500, self.config.search_altitude]),
                width=1000, length=1000, spacing=200
            )
            
        return {
            'mode': 'waypoint',
            'params': {
                'waypoints': [wp.tolist() for wp in self.search_waypoints]
            }
        }
    
    def _bbox_track(self, uav_state, target):
        # Tracking logic: Intercept target (PN Guidance)
        
        # Get 3D positions
        my_pos = np.array(uav_state['position'])
        my_vel = np.array(uav_state['velocity'])
        
        target_pos = None
        target_vel = None
        
        if hasattr(target, 'world_pos'):
             target_pos = target.world_pos
             target_vel = target.velocity
             
        if target_pos is not None:
             # Ensure velocity is 3D
             t_vel = target_vel if target_vel is not None else np.zeros(3)
             if len(t_vel) == 2:
                 t_vel = np.array([t_vel[0], t_vel[1], 0.0])
             
             # Use PN Guidance
             heading_cmd, pitch_cmd = PursuitLogic.proportional_navigation(
                 my_pos, my_vel, target_pos, t_vel
             )
             
             return {
                 'mode': 'track',
                 'params': {
                     'heading': np.degrees(heading_cmd),
                     'altitude': target_pos[2], # Match altitude for now
                     'speed': self.config.search_speed * self.config.track_speed_boost
                 }
             }
             
        return {'mode': 'manual', 'params': {}}

    def _bbox_lock(self, uav_state, target):
         # Lock logic: Visual Servoing (Image-based)
         # We need to center the target in the frame.
         
         if not hasattr(target, 'center'):
             # Fallback to track if no 2D center info (unlikely for visual target)
             return self._bbox_track(uav_state, target)
             
         roll_cmd, pitch_cmd = self.visual_servo.calculate_commands(
             target_center=target.center,
             frame_size=(640, 480) # TODO: Pass actual frame size from tracker/detector?
         )
         
         # Convert visual servo commands to autopilot directives
         # 'direct' mode is needed in autopilot to accept roll/pitch directly.
         
         return {
             'mode': 'direct',
             'params': {
                 'roll': roll_cmd,
                 'pitch': pitch_cmd,
                 'throttle': 0.8 # Maintain high speed for maneuvering
             }
         }
