"""
Matematik yardımcı fonksiyonları
"""

import numpy as np
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """Açıyı -pi, pi aralığına normalize et"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def degrees_to_radians(degrees: float) -> float:
    return np.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    return np.degrees(radians)


def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """3D mesafe"""
    return np.linalg.norm(p1 - p2)


def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """2D mesafe (x, y)"""
    return np.linalg.norm(p1[:2] - p2[:2])


def heading_to_target(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """Hedefe doğru yön açısı (radyan)"""
    diff = to_pos - from_pos
    return np.arctan2(diff[1], diff[0])


def rotation_matrix_z(yaw: float) -> np.ndarray:
    """Z ekseni etrafında dönüşüm matrisi"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler açılarından tam dönüşüm matrisi"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Değeri sınırla"""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation"""
    return a + (b - a) * t


def compute_intercept_point(target_pos: np.ndarray, target_vel: np.ndarray,
                            pursuer_pos: np.ndarray, pursuer_speed: float) -> np.ndarray:
    """
    Yakalama noktası hesapla (lead pursuit)
    
    Basitleştirilmiş hesaplama - sabit hız varsayımı
    """
    to_target = target_pos - pursuer_pos
    distance = np.linalg.norm(to_target)
    
    if distance < 1.0:
        return target_pos
        
    # Tahmini yakalama süresi
    time_to_intercept = distance / pursuer_speed
    
    # Hedefin tahmini konumu
    predicted = target_pos + target_vel * time_to_intercept * 0.5
    
    return predicted
