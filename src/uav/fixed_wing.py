"""
Sabit Kanatlı İHA Uçuş Modeli

Gerçekçi aerodinamik model ile sabit kanatlı İHA simülasyonu.
Kaldırma, sürükleme, yerçekimi ve kontrol yüzeyi etkileri hesaplanır.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import math
from src.utils import physics


@dataclass
class UAVState:
    """İHA durum değişkenleri"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))      # x, y, z (metre) - Inertial Frame
    velocity: np.ndarray = field(default_factory=lambda: np.array([20.0, 0.0, 0.0]))  # u, v, w (m/s) - Body Frame!
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))   # roll, pitch, yaw (radyan)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # p, q, r (rad/s) - Body Frame


@dataclass
class ControlInputs:
    """Kontrol girdileri"""
    aileron: float = 0.0     # [-1, 1] roll kontrolü
    elevator: float = 0.0    # [-1, 1] pitch kontrolü
    rudder: float = 0.0      # [-1, 1] yaw kontrolü
    throttle: float = 0.5    # [0, 1] motor gücü


class FixedWingUAV:
    """
    Sabit Kanatlı İHA Modeli (6DOF Physics)
    
    6 Serbestlik Dereceli Rijit Cisim Dinamiği kullanır:
    - Kuvvetler: Lift, Drag, Side Force, Thrust, Gravity
    - Momentler: Pitching, Rolling, Yawing (Aerodinamik katsayılar ile)
    - Eylemsizlik: Inertia Tensor
    """
    
    def __init__(self, config: dict = None, uav_id: str = None, team: str = "blue"):
        """
        Args:
            config: İHA konfigürasyonu
            uav_id: Benzersiz İHA kimliği
            team: Takım rengi (blue/red)
        """
        self.id = uav_id or f"uav_{id(self)}"
        self.team = team
        
        # Varsayılan config
        config = config or {}
        
        # Fiziksel parametreler
        self.mass = config.get('mass', 10.0)          # kg
        self.wingspan = config.get('wingspan', 2.0)   # metre
        self.wing_area = config.get('wing_area', 0.6) # m²
        self.chord = config.get('chord', 0.3)         # mean aerodynamic chord (m)
        
        # Eylemsizlik Momenti (Inertia Tensor) - kg*m^2
        # Varsayılan değerler basit bir dikdörtgen prizma veya benzeri bir gövde için
        self.Ixx = config.get('Ixx', 1.5)
        self.Iyy = config.get('Iyy', 2.5)
        self.Izz = config.get('Izz', 3.5)
        self.Ixz = config.get('Ixz', 0.1) # Cross product of inertia
        
        # Inertia Tensor
        self.J = np.array([
            [self.Ixx, 0, -self.Ixz],
            [0, self.Iyy, 0],
            [-self.Ixz, 0, self.Izz]
        ])
        self.J_inv = np.linalg.inv(self.J)
        
        # Performans Limitleri
        self.min_speed = config.get('min_speed', 20.0)  # m/s (stall) - INCREASED from 12
        self.max_speed = config.get('max_speed', 50.0)  # m/s - INCREASED from 45
        self.behavior = config.get('behavior', 'normal') 
        self.max_thrust = config.get('max_thrust', 50.0)  # Newton
        
        # Aerodinamik Katsayılar (Stability Derivatives)
        # Low-fidelity varsayılanları (Cessna benzeri stabil uçak)
        
        # Longitudinal (Lift, Drag, Pitch)
        self.CL0 = 0.3       # Zero-alpha lift (tuned for L=W at 30 m/s cruise)
        self.CLa = 5.0       # Lift slope (per rad) - for good pitch responsiveness
        self.CD0 = 0.03      # Parasitic drag
        self.CD_induced_k = 0.04  # Induced drag factor (1/(pi*e*AR))
        self.Cm0 = -0.08     # Zero-alpha pitch moment (INCREASED negative for stability)
        self.Cma = -1.0      # Pitch stability (per rad) - increased from -0.6
        self.Cmq = -20.0     # Pitch damping (per rad/s) - increased from -10
        self.Cm_de = -0.8    # Elevator control power (per rad)
        self.stall_drag_factor = config.get('stall_drag_factor', 0.12)
        
        # Lateral (Side force, Roll, Yaw) - Tuned for stability
        self.CYb = -0.3      # Side force stability
        self.Clb = -0.1      # Dihedral effect - REDUCED to prevent roll divergence
        self.Cnb = 0.12      # Weathercock stability
        self.Clp = -2.0      # Roll damping - SIGNIFICANTLY INCREASED from -0.8
        self.Cnr = -1.5      # Yaw damping - SIGNIFICANTLY INCREASED from -0.5
        self.Cl_da = 0.15    # Aileron control power
        self.Cn_dr = -0.1    # Rudder control power
        
        # Durum
        self.state = UAVState()
        self.controls = ControlInputs()
        
        # Kontrol yüzeyi hız limitleri (birim/saniye)
        self.aileron_rate = config.get('aileron_rate', 3.0)
        self.elevator_rate = config.get('elevator_rate', 3.0)
        self.rudder_rate = config.get('rudder_rate', 3.0)
        self.throttle_rate = config.get('throttle_rate', 1.5)
        
        # Simülasyon değişkenleri
        self.is_crashed = False
        self.is_stalled = False
        self._current_thrust = 0.0
        self._total_time = 0.0
        
        # Yer etkileşimi parametreleri
        self.ground_restitution = config.get('ground_restitution', 0.2)
        self.ground_friction = config.get('ground_friction', 0.6)
        self.crash_vertical_speed = config.get('crash_vertical_speed', 12.0)
        
    def reset(self, position: np.ndarray = None, heading: float = 0.0):
        """İHA'yı başlangıç durumuna sıfırla"""
        self.state = UAVState()
        if position is not None:
            self.state.position = np.array(position, dtype=float)
        
        self.state.orientation[2] = heading  # yaw
        
        # Başlangıç hızı (Body Frame: u, v, w)
        # Sadece ileri hız verelim (u)
        speed = self.min_speed * 1.5
        self.state.velocity = np.array([speed, 0.0, 0.0])
        
        self.is_crashed = False
        self.is_stalled = False
        self._current_thrust = self.max_thrust * 0.6  # Cruise thrust
        self.controls.throttle = 0.6
        
    def set_controls(self, aileron: float = None, elevator: float = None, 
                     rudder: float = None, throttle: float = None,
                     dt: Optional[float] = None):
        """Kontrol girdilerini ayarla"""
        if aileron is not None:
            target = np.clip(aileron, -1, 1)
            self.controls.aileron = self._apply_rate_limit(
                self.controls.aileron, target, self.aileron_rate, dt
            )
        if elevator is not None:
            target = np.clip(elevator, -1, 1)
            self.controls.elevator = self._apply_rate_limit(
                self.controls.elevator, target, self.elevator_rate, dt
            )
        if rudder is not None:
            target = np.clip(rudder, -1, 1)
            self.controls.rudder = self._apply_rate_limit(
                self.controls.rudder, target, self.rudder_rate, dt
            )
        if throttle is not None:
            target = np.clip(throttle, 0, 1)
            self.controls.throttle = self._apply_rate_limit(
                self.controls.throttle, target, self.throttle_rate, dt
            )
            
    def update(self, dt: float):
        """
        6DOF Fizik güncellemesi
        
        Args:
            dt: Zaman adımı (saniye)
        """
        if self.is_crashed or self.behavior == 'stationary':
            return
            
        self._total_time += dt
        
        # 1. Ortam ve Hava Hızı
        # Rüzgar (Inertial Frame)
        wind_inertial = physics.get_wind_at_pos(self.state.position, self._total_time)
        
        # Dönüşüm Matrisi (Body -> Inertial)
        phi, theta, psi = self.state.orientation
        R_b2i = self._rotation_matrix(phi, theta, psi)
        R_i2b = R_b2i.T
        
        # Hava Durumu (Body Frame)
        # v_air_body = v_body - R_i2b * v_wind_inertial
        # self.state.velocity IS NOW BODY FRAME VELOCITY (u, v, w)
        v_body = self.state.velocity
        v_wind_body = R_i2b @ wind_inertial
        v_air_body = v_body - v_wind_body
        
        airspeed = np.linalg.norm(v_air_body)
        if airspeed < 0.1: airspeed = 0.1
        
        # 2. Aerodinamik Açılar (Alpha, Beta)
        # u, v, w = v_air_body
        u, v, w = v_air_body
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / airspeed, -1.0, 1.0))
        
        self.is_stalled = (airspeed < self.min_speed) or (abs(np.degrees(alpha)) > 15.0)
        
        # Hava yoğunluğu
        rho = physics.get_air_density(self.state.position[2])
        q_bar = 0.5 * rho * airspeed**2  # Dinamik basınç
        
        # 3. Kuvvet ve Moment Katsayıları
        
        # Kontrol yüzeyleri
        da = self.controls.aileron  # Aileron
        de = self.controls.elevator # Elevator
        dr = self.controls.rudder   # Rudder
        
        # Lift Coefficient (Linear range + simple stall drop)
        CL = self.CL0 + self.CLa * alpha
        if self.is_stalled:
            CL *= 0.5 # Basit post-stall lift kaybı
            
        # Drag Coefficient
        CD = self.CD0 + self.CD_induced_k * CL**2
        if self.is_stalled:
            CD += self.stall_drag_factor
        
        # Side Force Coefficient
        CY = self.CYb * beta + self.Cn_dr * dr
        
        # Moment Coefficients
        b = self.wingspan
        c = self.chord
        # Açısal hızlar (normalize edilmiş)
        p, q, r = self.state.angular_velocity
        norm_p = p * b / (2 * airspeed)
        norm_q = q * c / (2 * airspeed)
        norm_r = r * b / (2 * airspeed)
        
        # Pitching Moment (Cm)
        Cm = self.Cm0 + self.Cma * alpha + self.Cmq * norm_q + self.Cm_de * de
        
        # Rolling Moment (Cl)
        Cl = self.Clb * beta + self.Clp * norm_p + self.Cl_da * da
        
        # Yawing Moment (Cn)
        Cn = self.Cnb * beta + self.Cnr * norm_r + self.Cn_dr * dr
        
        # 4. Kuvvetlerin Hesaplanması (Stability Axes -> Body Axes)
        # Lift/Drag, rüzgar ekseninden gövde eksenine çevrilecek
        # Basitleştirme: Küçük açılar için (Stability axis)
        # F_lift = q_bar * S * CL
        # F_drag = q_bar * S * CD
        
        S = self.wing_area
        
        # Rüzgar eksenindeki kuvvetler
        L = q_bar * S * CL
        D = q_bar * S * CD
        Y = q_bar * S * CY
        
        # İtki (Thrust) - Body X ekseninde kabul edilir
        target_thrust = self.max_thrust * self.controls.throttle
        self._current_thrust += (target_thrust - self._current_thrust) * (2.0 * dt)
        T = self._current_thrust
        
        # Body frame forces (Aerodynamic + Thrust)
        # Dönüşüm: Wind -> Body (Alpha ve Beta ile)
        # F_body_aero approximation:
        # Fx = -D * cos(a) + L * sin(a)
        # Fz = -D * sin(a) - L * cos(a)
        # Fy = Y
        
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        Fx_aero = -D * ca + L * sa
        Fz_aero = -D * sa - L * ca
        Fy_aero = Y
        
        F_body = np.array([Fx_aero + T, Fy_aero, Fz_aero])
        
        # 5. Momentlerin Hesaplanması (Body Frame)
        M_body = np.array([
            q_bar * S * b * Cl, # Roll (L)
            q_bar * S * c * Cm, # Pitch (M)
            q_bar * S * b * Cn  # Yaw (N)
        ])
        
        # 6. Yerçekimi (Body Frame)
        # Z ekseni yukarı olduğu için yerçekimi -Z yönündedir
        g_vec = np.array([0, 0, -physics.GRAVITY])
        # R_i2b ile dünya->gövde dönüşümü
        F_gravity_body = self.mass * (R_i2b @ g_vec)
        
        F_total = F_body + F_gravity_body
        
        # 7. Hareket Denklemleri (Equations of Motion)
        # Newton-Euler Equations rigid body
        
        # Lineer İvme: v_dot = F/m - w x v
        omega = self.state.angular_velocity
        v_dot = (F_total / self.mass) - np.cross(omega, v_body)
        
        # Açısal İvme: omega_dot = J_inv * (M - w x Jw)
        Jw = self.J @ omega
        # Gyroscopic term
        gyro_term = np.cross(omega, Jw)
        omega_dot = self.J_inv @ (M_body - gyro_term)
        
        # 8. İntegrasyon (Euler)
        self.state.velocity += v_dot * dt
        self.state.angular_velocity += omega_dot * dt
        
        # 8.5 Stability Augmentation System (SAS) - selective axis damping
        # Tuned for best stability without crash
        self.state.angular_velocity[0] *= 0.97  # Roll: 3% decay
        self.state.angular_velocity[1] *= 0.99  # Pitch: 1% decay (preserve altitude control)
        self.state.angular_velocity[2] *= 0.92  # Yaw: 8% decay
        
        # Pozisyon (Inertial Frame)
        # v_inertial = R_b2i * v_body
        v_inertial = R_b2i @ self.state.velocity
        self.state.position += v_inertial * dt
        
        # Oryantasyon (Quaternion tabanlı entegrasyon)
        quat = self._euler_to_quaternion(phi, theta, psi)
        quat = self._integrate_quaternion(quat, self.state.angular_velocity, dt)
        self.state.orientation = self._quaternion_to_euler(quat)
        
        # 9. Sınırlandırma ve cleanup
        # Wrap yaw
        self.state.orientation[2] = self._normalize_angle(self.state.orientation[2])
        # Limit pitch/roll visual ranges? No, physics should allow generic motion.
        
        # Yere çarpma
        if self.state.position[2] < 0:
            self.state.position[2] = 0
            v_inertial = R_b2i @ self.state.velocity
            if abs(v_inertial[2]) > self.crash_vertical_speed:
                self.state.velocity = np.zeros(3)
                self.state.angular_velocity = np.zeros(3)
                self.is_crashed = True
            else:
                v_inertial[2] = -v_inertial[2] * self.ground_restitution
                v_inertial[0] *= max(0.0, 1.0 - self.ground_friction)
                v_inertial[1] *= max(0.0, 1.0 - self.ground_friction)
                self.state.velocity = R_i2b @ v_inertial
            
    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Euler açılarından dönüşüm matrisi oluştur (Body -> Inertial)"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Z-Y-X rotation sequence (Yaw, Pitch, Roll)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        return R
        
    def _normalize_angle(self, angle: float) -> float:
        """Açıyı -pi, pi aralığına normalize et"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Euler (roll, pitch, yaw) -> quaternion (w, x, y, z)."""
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])

    def _quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Quaternion (w, x, y, z) -> Euler (roll, pitch, yaw)."""
        w, x, y, z = quat
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion çarpımı (w, x, y, z)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    def _integrate_quaternion(self, quat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """Quaternionu açısal hızla RK4 entegre et."""
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])

        def q_dot(q):
            return 0.5 * self._quat_multiply(q, omega_quat)

        k1 = q_dot(quat)
        k2 = q_dot(quat + 0.5 * dt * k1)
        k3 = q_dot(quat + 0.5 * dt * k2)
        k4 = q_dot(quat + dt * k3)
        quat_next = quat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return quat_next / np.linalg.norm(quat_next)

    def _apply_rate_limit(
        self,
        current: float,
        target: float,
        rate: float,
        dt: Optional[float]
    ) -> float:
        """Kontrol değerine hız limiti uygula."""
        if dt is None:
            return target
        max_delta = rate * dt
        return current + np.clip(target - current, -max_delta, max_delta)
        
    # --- Getters & Helpers ---
    # Not: Bazı getter'lar artık doğrudan _state_ değişkenlerinden alınabilir,
    # ancak geriye dönük uyumluluk için koruyoruz.
        
    def get_speed(self) -> float:
        """Mevcut hızı döndür (m/s)"""
        return np.linalg.norm(self.state.velocity)
        
    def get_altitude(self) -> float:
        """Mevcut irtifayı döndür (metre)"""
        return self.state.position[2]
        
    def get_heading(self) -> float:
        """Mevcut yön açısını döndür (radyan)"""
        return self.state.orientation[2]
        
    def get_heading_degrees(self) -> float:
        """Mevcut yön açısını döndür (derece)"""
        return np.degrees(self.state.orientation[2])
        
    def get_position(self) -> np.ndarray:
        """Mevcut pozisyonu döndür"""
        return self.state.position.copy()
        
    def get_orientation(self) -> np.ndarray:
        """Mevcut oryantasyonu döndür (roll, pitch, yaw) radyan"""
        return self.state.orientation.copy()
        
    def get_orientation_degrees(self) -> Tuple[float, float, float]:
        """Mevcut oryantasyonu döndür (roll, pitch, yaw) derece"""
        return tuple(np.degrees(self.state.orientation))
        
    def get_forward_vector(self) -> np.ndarray:
        """İleri yön vektörünü döndür (Inertial Frame)"""
        yaw = self.state.orientation[2]
        pitch = self.state.orientation[1]
        return np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch)
        ])
        
    def get_camera_position(self) -> np.ndarray:
        """Kamera pozisyonunu döndür (İHA burnunda)"""
        forward = self.get_forward_vector()
        return self.state.position + forward * 0.5  # 0.5m ileri
        
    def to_dict(self) -> dict:
        """İHA durumunu sözlük olarak döndür"""
        # Velocity inertial frame'e dönüştürülmeli visualizer için
        R = self._rotation_matrix(*self.state.orientation)
        v_inertial = R @ self.state.velocity
        
        return {
            'id': self.id,
            'team': self.team,
            'position': self.state.position.tolist(),
            'velocity': v_inertial.tolist(), # Visualizer inertial velocity bekliyor
            'orientation': self.state.orientation.tolist(),
            'speed': self.get_speed(),
            'altitude': self.get_altitude(),
            'heading': self.get_heading_degrees(),
            'is_crashed': self.is_crashed,
            'is_stalled': self.is_stalled,
            'controls': {
                'aileron': self.controls.aileron,
                'elevator': self.controls.elevator,
                'rudder': self.controls.rudder,
                'throttle': self.controls.throttle
            }
        }
