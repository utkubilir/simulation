"""
Otonom Uçuş Sistemi (Autopilot)

Waypoint takibi, hedef takibi ve otonom manevralar.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AutopilotMode(Enum):
    """Autopilot modları"""
    # MANUAL = "manual"  # KALDIRILDI - Tam otonom sistem
    ALTITUDE_HOLD = "alt_hold"  # İrtifa sabitle
    HEADING_HOLD = "hdg_hold"   # Yön sabitle
    WAYPOINT = "waypoint"       # Waypoint takibi
    ORBIT = "orbit"             # Hedef etrafında dönme
    TARGET_TRACK = "track"      # Hedef takibi
    RETURN_HOME = "rth"         # Eve dön
    COMBAT = "combat"           # Otonom hava savaşı
    KAMIKAZE = "kamikaze"       # Yer hedefi dalış görevi


@dataclass
class Waypoint:
    """Waypoint tanımı"""
    position: np.ndarray  # x, y, z
    speed: float = 25.0   # Hedef hız (m/s)
    radius: float = 20.0  # Kabul yarıçapı (m)


class Autopilot:
    """
    Otonom Uçuş Sistemi
    
    PID kontrolcüler kullanarak otonom uçuş sağlar:
    - İrtifa tutma
    - Yön tutma 
    - Waypoint takibi
    - Hedef takibi (İHA kovalama)
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        self.mode = AutopilotMode.COMBAT  # Varsayılan: otonom savaş modu
        self.enabled = True  # Her zaman aktif
        
        # Hedefler
        self.target_altitude = 100.0  # metre
        self.target_heading = 0.0     # radyan
        self.target_speed = 25.0      # m/s
        
        # Waypoint listesi
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_index = 0
        
        # Hedef takibi
        self.track_target_position: Optional[np.ndarray] = None
        self.track_target_velocity: Optional[np.ndarray] = None
        
        # Orbit parametreleri
        self.orbit_center: Optional[np.ndarray] = None
        self.orbit_radius = 100.0  # metre
        self.orbit_direction = 1   # 1: saat yönü, -1: saat yönü tersi
        
        # PID kontrolcü kazançları
        self.altitude_pid = PIDController(
            kp=config.get('altitude_kp', 0.1),
            ki=config.get('altitude_ki', 0.01),
            kd=config.get('altitude_kd', 0.05)
        )
        
        self.heading_pid = PIDController(
            kp=config.get('heading_kp', 1.0),
            ki=config.get('heading_ki', 0.1),
            kd=config.get('heading_kd', 0.2)
        )
        
        self.speed_pid = PIDController(
            kp=config.get('speed_kp', 0.1),
            ki=config.get('speed_ki', 0.01),
            kd=config.get('speed_kd', 0.0)
        )
        
        # Ev pozisyonu
        self.home_position = np.array([0.0, 0.0, 100.0])

        # Combat Manager
        from src.uav.combat import CombatStateManager
        self.combat_manager = CombatStateManager()
        
    def set_mode(self, mode: AutopilotMode):
        """Autopilot modunu değiştir"""
        self.mode = mode
        self._reset_controllers()
        
    def enable(self):
        """Autopilot'u etkinleştir"""
        self.enabled = True
        
    def disable(self):
        """Autopilot'u devre dışı bırak (Şartname: Tam otonom zorunlu, bu fonksiyon pasif)"""
        # NOT: Şartname gereği tam otonom mod zorunlu
        # Bu fonksiyon artık modu değiştirmez
        # self.enabled = False  # KALDIRILDI
        # self.mode = AutopilotMode.MANUAL  # KALDIRILDI
        print("⚠️ UYARI: Tam otonom sistem - manuel mod devre dışı!")
        
    def set_target_altitude(self, altitude: float):
        """Hedef irtifayı ayarla"""
        self.target_altitude = max(10.0, altitude)
        
    def set_target_heading(self, heading: float):
        """Hedef yönü ayarla (derece)"""
        self.target_heading = np.radians(heading)
        
    def set_waypoints(self, waypoints: List[Tuple[float, float, float]]):
        """Waypoint listesi ayarla"""
        self.waypoints = [
            Waypoint(position=np.array(wp)) for wp in waypoints
        ]
        self.current_waypoint_index = 0
        
    def set_track_target(self, position: np.ndarray, velocity: np.ndarray = None):
        """Takip edilecek hedefi ayarla"""
        self.track_target_position = position.copy()
        if velocity is not None:
            velocity = np.array(velocity)
            if velocity.shape == (2,):
                velocity = np.array([velocity[0], velocity[1], 0.0])
            self.track_target_velocity = velocity
        else:
            self.track_target_velocity = np.zeros(3)
        
    def set_orbit(self, center: np.ndarray, radius: float = 100.0, direction: int = 1):
        """Orbit parametrelerini ayarla"""
        self.orbit_center = center.copy()
        self.orbit_radius = radius
        self.orbit_direction = direction
        
    def update(self, uav_state: dict, dt: float) -> dict:
        """
        Autopilot güncellemesi
        
        Args:
            uav_state: İHA durumu {position, velocity, orientation, speed, altitude, heading}
            dt: Zaman adımı
            
        Returns:
            Kontrol komutları {aileron, elevator, rudder, throttle}
        """
        if not self.enabled:
            return None
            
        position = np.array(uav_state['position'])
        velocity = np.array(uav_state['velocity'])
        altitude = uav_state['altitude']
        speed = uav_state['speed']
        heading = np.radians(uav_state['heading'])
        
        # Varsayılan kontrol
        controls = {
            'aileron': 0.0,
            'elevator': 0.0,
            'rudder': 0.0,
            'throttle': 0.5
        }
        
        if self.mode == AutopilotMode.ALTITUDE_HOLD:
            controls = self._altitude_hold(altitude, speed, heading, dt)
            
        elif self.mode == AutopilotMode.HEADING_HOLD:
            controls = self._heading_hold(altitude, speed, heading, dt)
            
        elif self.mode == AutopilotMode.WAYPOINT:
            controls = self._waypoint_follow(position, altitude, speed, heading, dt)
            
        elif self.mode == AutopilotMode.ORBIT:
            controls = self._orbit_target(position, altitude, speed, heading, dt)
            
        elif self.mode == AutopilotMode.TARGET_TRACK:
            controls = self._track_target(position, velocity, altitude, speed, heading, dt)
            
        elif self.mode == AutopilotMode.RETURN_HOME:
            controls = self._return_home(position, altitude, speed, heading, dt)

        elif self.mode == AutopilotMode.COMBAT:
            controls = self._run_combat_logic(uav_state, dt)
            
        return controls
        
    def _altitude_hold(self, altitude: float, speed: float, heading: float, dt: float) -> dict:
        """İrtifa tutma"""
        # İrtifa hatası
        alt_error = self.target_altitude - altitude
        elevator = self.altitude_pid.update(alt_error, dt)
        
        # Hız kontrolü
        speed_error = self.target_speed - speed
        throttle = 0.5 + self.speed_pid.update(speed_error, dt)
        
        return {
            'aileron': 0.0,
            'elevator': np.clip(elevator, -1, 1),
            'rudder': 0.0,
            'throttle': np.clip(throttle, 0, 1)
        }
        
    def _heading_hold(self, altitude: float, speed: float, heading: float, dt: float) -> dict:
        """Yön tutma"""
        # Yön hatası
        hdg_error = self._normalize_angle(self.target_heading - heading)
        aileron = self.heading_pid.update(hdg_error, dt)
        
        # İrtifa kontrolü
        alt_error = self.target_altitude - altitude
        elevator = self.altitude_pid.update(alt_error, dt)
        
        # Hız kontrolü
        speed_error = self.target_speed - speed
        throttle = 0.5 + self.speed_pid.update(speed_error, dt)
        
        return {
            'aileron': np.clip(aileron, -1, 1),
            'elevator': np.clip(elevator, -1, 1),
            'rudder': 0.0,
            'throttle': np.clip(throttle, 0, 1)
        }
        
    def _waypoint_follow(self, position: np.ndarray, altitude: float, 
                         speed: float, heading: float, dt: float) -> dict:
        """Waypoint takibi"""
        if not self.waypoints:
            return self._altitude_hold(altitude, speed, heading, dt)
            
        # Mevcut waypoint
        wp = self.waypoints[self.current_waypoint_index]
        
        # Waypoint'e olan mesafe (2D)
        diff = wp.position[:2] - position[:2]
        distance = np.linalg.norm(diff)
        
        # Waypoint'e ulaşıldı mı?
        if distance < wp.radius:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.current_waypoint_index = 0  # Döngüye al
                
        # Hedef yön
        target_heading = np.arctan2(diff[1], diff[0])
        hdg_error = self._normalize_angle(target_heading - heading)
        aileron = self.heading_pid.update(hdg_error, dt)
        
        # Hedef irtifa
        self.target_altitude = wp.position[2]
        alt_error = self.target_altitude - altitude
        elevator = self.altitude_pid.update(alt_error, dt)
        
        # Hız
        self.target_speed = wp.speed
        speed_error = self.target_speed - speed
        throttle = 0.5 + self.speed_pid.update(speed_error, dt)
        
        return {
            'aileron': np.clip(aileron, -1, 1),
            'elevator': np.clip(elevator, -1, 1),
            'rudder': 0.0,
            'throttle': np.clip(throttle, 0, 1)
        }
        
    def _orbit_target(self, position: np.ndarray, altitude: float,
                      speed: float, heading: float, dt: float) -> dict:
        """Hedef etrafında orbit"""
        if self.orbit_center is None:
            return self._altitude_hold(altitude, speed, heading, dt)
            
        # Merkeze olan vektör
        to_center = self.orbit_center[:2] - position[:2]
        dist_to_center = np.linalg.norm(to_center)
        
        # Teğet yön (orbit yönüne göre)
        angle_to_center = np.arctan2(to_center[1], to_center[0])
        tangent_angle = angle_to_center + self.orbit_direction * np.pi / 2
        
        # Yarıçap düzeltmesi
        radius_error = dist_to_center - self.orbit_radius
        tangent_angle -= self.orbit_direction * np.arctan(radius_error / self.orbit_radius) * 0.5
        
        # Yön kontrolü
        hdg_error = self._normalize_angle(tangent_angle - heading)
        aileron = self.heading_pid.update(hdg_error, dt)
        
        # İrtifa
        self.target_altitude = self.orbit_center[2] if len(self.orbit_center) > 2 else 100.0
        alt_error = self.target_altitude - altitude
        elevator = self.altitude_pid.update(alt_error, dt)
        
        # Hız
        speed_error = self.target_speed - speed
        throttle = 0.5 + self.speed_pid.update(speed_error, dt)
        
        return {
            'aileron': np.clip(aileron, -1, 1),
            'elevator': np.clip(elevator, -1, 1),
            'rudder': 0.0,
            'throttle': np.clip(throttle, 0, 1)
        }
        
    def _track_target(self, position: np.ndarray, velocity: np.ndarray,
                      altitude: float, speed: float, heading: float, dt: float) -> dict:
        """Hedef takibi (interception)"""
        if self.track_target_position is None:
            return self._altitude_hold(altitude, speed, heading, dt)
            
        # Basit takip: hedefe doğru uç
        # Gelişmiş versiyon: intercept noktası hesapla
        
        to_target = self.track_target_position - position
        distance = np.linalg.norm(to_target[:2])
        
        # Hedef yön
        target_heading = np.arctan2(to_target[1], to_target[0])
        
        # Lead pursuit: hedefin hareketini hesaba kat
        if self.track_target_velocity is not None:
            time_to_intercept = distance / max(speed, 1.0)
            predicted_pos = self.track_target_position + self.track_target_velocity * time_to_intercept * 0.5
            to_predicted = predicted_pos - position
            target_heading = np.arctan2(to_predicted[1], to_predicted[0])
        
        hdg_error = self._normalize_angle(target_heading - heading)
        aileron = self.heading_pid.update(hdg_error, dt) * 1.5  # Agresif takip
        
        # Hedef irtifasına çık
        self.target_altitude = self.track_target_position[2]
        alt_error = self.target_altitude - altitude
        elevator = self.altitude_pid.update(alt_error, dt)
        
        # Tam gaz takip
        throttle = 0.9 if distance > 50 else 0.7
        
        return {
            'aileron': np.clip(aileron, -1, 1),
            'elevator': np.clip(elevator, -1, 1),
            'rudder': 0.0,
            'throttle': throttle
        }
        
    def _return_home(self, position: np.ndarray, altitude: float,
                     speed: float, heading: float, dt: float) -> dict:
        """Eve dönüş"""
        to_home = self.home_position[:2] - position[:2]
        distance = np.linalg.norm(to_home)
        
        target_heading = np.arctan2(to_home[1], to_home[0])
        hdg_error = self._normalize_angle(target_heading - heading)
        aileron = self.heading_pid.update(hdg_error, dt)
        
        self.target_altitude = self.home_position[2]
        alt_error = self.target_altitude - altitude
        elevator = self.altitude_pid.update(alt_error, dt)
        
        throttle = 0.6
        
        return {
            'aileron': np.clip(aileron, -1, 1),
            'elevator': np.clip(elevator, -1, 1),
            'rudder': 0.0,
            'throttle': throttle
        }
        
    def _normalize_angle(self, angle: float) -> float:
        """Açıyı -pi, pi aralığına normalize et"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def _reset_controllers(self):
        """PID kontrolcüleri sıfırla"""
        self.altitude_pid.reset()
        self.heading_pid.reset()
        self.speed_pid.reset()


    def set_combat_detections(self, tracks: list):
        """Combat manager için tespitleri güncelle"""
        if hasattr(self.combat_manager, 'latest_tracks'):
             # Combat manager doesn't store tracks, it expects them in update.
             # We need to temporarily store them here or pass them in update.
             # Better design: pass them to autopilot.update?
             # But autopilot.update signature is fixed in main loop usually.
             # Let's add a state variable.
             pass
        self._latest_tracks = tracks

    def set_avoidance_heading(self, heading: float):
        """
        Hava savunma bölgesi kaçınma heading'i ayarla
        
        Args:
            heading: Kaçınma yönü (derece)
        """
        self._avoidance_heading = heading
        self._avoidance_active = True
        
    def clear_avoidance(self):
        """Kaçınma modunu temizle"""
        self._avoidance_heading = None
        self._avoidance_active = False

    def _run_combat_logic(self, uav_state: dict, dt: float) -> dict:
        """Combat mantığını koştur"""
        tracks = getattr(self, '_latest_tracks', [])
        
        # Combat manager'dan karar al
        decision = self.combat_manager.update(uav_state, tracks)
        
        mode = decision.get('mode')
        params = decision.get('params', {})
        
        # Kararı uygula (Alt seviye modlara yönlendir)
        if mode == 'waypoint':
            waypoints = params.get('waypoints', [])
            if waypoints:
                # Sadece waypointler değiştiyse güncelle
                # (Optimize edilebilir, şimdilik her seferinde set ediyoruz)
                # Waypoint check logic inside combat manager creates stable waypoints
                # Check if we need to reset waypoints
                current_wps = [w.position.tolist() for w in self.waypoints]
                if not current_wps or current_wps != waypoints:
                    self.set_waypoints(waypoints)
            
            return self._waypoint_follow(
                np.array(uav_state['position']), 
                uav_state['altitude'], 
                uav_state['speed'], 
                np.radians(uav_state['heading']), 
                dt
            )
            
        elif mode == 'direct':
            # Visual Servoing Direct Control
            # We assume these are roll/pitch commands in radians
            roll = params.get('roll', 0.0)
            pitch = params.get('pitch', 0.0)
            throttle = params.get('throttle', 0.8)
            
            # Autopilot maps roll -> aileron, pitch -> elevator directly?
            # PID controllers usually output control surface deflections (-1 to 1).
            # If VisualServo returns "command" (radians of desired attitude?), we might need PID.
            # BUT, let's assume VisualServo outputs proportional control commands that map roughly to sticks.
            # Stick inputs are -1 to 1.
            # VisualServo: (kp * pixel_error) -> can be large. we should clip.
            
            return {
                'aileron': np.clip(roll, -1, 1),
                'elevator': np.clip(pitch, -1, 1),
                'rudder': 0.0,
                'throttle': throttle
            }

        elif mode == 'track':
            # Parametrelerden hedef bilgilerini al
            # Combat manager henüz full track parametreleri dönmüyor olabilir
            # Şimdilik mevcut _track_target mantığını kullanalım
            # Ama hedef ID'yi combat manager biliyor
            target_id = self.combat_manager.target_id
            target = self.combat_manager._get_target(tracks)
            
            if target:
                # Hedef konumunu güncelle
                # Use world_pos if available
                if hasattr(target, 'world_pos') and target.world_pos is not None:
                    pos = target.world_pos
                    self.set_track_target(
                        position=pos,
                        velocity=target.velocity if hasattr(target, 'velocity') else None
                    )
                    return self._track_target(
                        np.array(uav_state['position']), 
                        np.array(uav_state['velocity']),
                        uav_state['altitude'], 
                        uav_state['speed'], 
                        np.radians(uav_state['heading']), 
                        dt
                    )
                elif hasattr(target, 'center') and target.center is not None:
                    # Sadece 2D bilgi var - 3D pozisyon tahmin edilemez
                    # Heading hold moduna geç ve hedefe yönelmeye çalış
                    # Bu durumda visual servoing daha uygun olurdu
                    # Şimdilik mevcut yönde devam et ve altitude hold yap
                    return self._altitude_hold(
                        uav_state['altitude'],
                        uav_state['speed'],
                        np.radians(uav_state['heading']),
                        dt
                    )
        
        # Fallback
        return self._altitude_hold(uav_state['altitude'], uav_state['speed'], np.radians(uav_state['heading']), dt)


class PIDController:
    """Basit PID kontrolcü"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = 10.0
        
    def update(self, error: float, dt: float) -> float:
        """PID çıktısını hesapla"""
        # Proportional
        p = self.kp * error
        
        # Integral
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self.integral
        
        # Derivative
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0
        d = self.kd * derivative
        
        self.prev_error = error
        
        return p + i + d
        
    def reset(self):
        """Kontrolcüyü sıfırla"""
        self.integral = 0.0
        self.prev_error = 0.0
