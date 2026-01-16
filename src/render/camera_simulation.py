"""
Kamera Simülasyonu

Gerçekçi kamera parametreleri ve davranışları.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class CameraConfig:
    """Kamera konfigürasyonu"""
    # Çözünürlük
    width: int = 640
    height: int = 480
    fps: int = 30
    
    # Lens
    fov: float = 60.0              # Görüş alanı (derece)
    focal_length_mm: float = 35.0  # Odak uzaklığı (mm)
    aperture: float = 2.8          # Diyafram (f-stop)
    
    # Sensör
    sensor_width_mm: float = 6.4   # Sensör genişliği (mm)
    sensor_height_mm: float = 4.8  # Sensör yüksekliği (mm)
    iso: int = 200
    exposure_time: float = 1/500   # Pozlama süresi (saniye)
    
    # Montaj
    mount_offset: Tuple[float, float, float] = (0.3, 0.0, -0.1)  # İHA'ya göre offset
    mount_pitch: float = -5.0      # Montaj pitch açısı (derece)
    
    # Atmosfer
    visibility: float = 10000.0    # Görüş mesafesi (metre)
    haze_color: Tuple[float, float, float] = (0.8, 0.85, 0.9)
    
    # Aydınlatma
    sun_azimuth: float = 45.0      # Güneş azimut (derece)
    sun_elevation: float = 60.0    # Güneş yükseklik (derece)


class CameraSimulation:
    """
    Gerçekçi kamera simülasyonu
    
    Lens parametreleri, sensör karakteristikleri ve
    atmosferik koşulları simüle eder.
    
    Usage:
        cam = CameraSimulation(CameraConfig())
        K = cam.get_intrinsic_matrix()
        frame = cam.apply_atmospheric_effects(raw_frame, distances)
    """
    
    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        
        # Hesaplanmış değerler
        self._K = None  # İçsel matris
        self._update_intrinsics()
        
    def _update_intrinsics(self):
        """Kamera içsel matrisini hesapla"""
        w = self.config.width
        h = self.config.height
        fov = self.config.fov
        
        # Focal length (pixel)
        fx = w / (2 * np.tan(np.radians(fov / 2)))
        fy = fx  # Kare piksel varsayımı
        
        # Principal point
        cx = w / 2
        cy = h / 2
        
        self._K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
    @property
    def K(self) -> np.ndarray:
        """İçsel (intrinsic) matris"""
        return self._K
        
    @property
    def focal_length_px(self) -> float:
        """Piksel cinsinden focal length"""
        return self._K[0, 0]
        
    def get_camera_pose(self, 
                        uav_position: np.ndarray,
                        uav_orientation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        İHA durumundan kamera pozisyon ve oryantasyonunu hesapla
        
        Args:
            uav_position: İHA pozisyonu [x, y, z]
            uav_orientation: İHA oryantasyonu [roll, pitch, yaw] (radyan)
            
        Returns:
            (camera_position, camera_orientation) tuple
        """
        roll, pitch, yaw = uav_orientation
        
        # İHA rotasyon matrisi
        R = self._euler_to_rotation_matrix(roll, pitch, yaw)
        
        # Kamera pozisyonu
        mount_offset = np.array(self.config.mount_offset)
        camera_pos = uav_position + R @ mount_offset
        
        # Kamera oryantasyonu
        mount_pitch_rad = np.radians(self.config.mount_pitch)
        camera_orient = np.array([
            roll,
            pitch + mount_pitch_rad,
            yaw
        ])
        
        return camera_pos, camera_orient
        
    def project_point(self, 
                      world_point: np.ndarray,
                      camera_pos: np.ndarray,
                      camera_orient: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        3D noktayı 2D ekran koordinatlarına projete et
        
        Args:
            world_point: Dünya koordinatlarında 3D nokta
            camera_pos: Kamera pozisyonu
            camera_orient: Kamera oryantasyonu [roll, pitch, yaw]
            
        Returns:
            (x, y) ekran koordinatları veya None
        """
        # Kamera koordinatlarına dönüştür
        rel_pos = world_point - camera_pos
        R = self._euler_to_rotation_matrix(*camera_orient)
        cam_coords = R.T @ rel_pos
        
        # Kameranın arkasında mı?
        if cam_coords[0] <= 0.1:
            return None
            
        # Perspektif projeksiyon
        # Koordinat sistemi: x=ileri, y=sağ, z=aşağı
        x = self._K[0, 0] * cam_coords[1] / cam_coords[0] + self._K[0, 2]
        y = self._K[1, 1] * cam_coords[2] / cam_coords[0] + self._K[1, 2]
        
        # Ekran sınırları
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            return (x, y)
            
        return None
        
    def calculate_apparent_size(self, 
                                 distance: float,
                                 object_size: float = 2.0) -> float:
        """
        Nesnenin ekrandaki görünür boyutunu hesapla
        
        Args:
            distance: Hedefe mesafe (metre)
            object_size: Gerçek nesne boyutu (metre)
            
        Returns:
            Piksel cinsinden görünür boyut
        """
        if distance <= 0:
            return 0
        return self.focal_length_px * object_size / distance
        
    def is_in_fov(self, 
                  world_point: np.ndarray,
                  camera_pos: np.ndarray,
                  camera_orient: np.ndarray) -> bool:
        """Noktanın görüş alanında olup olmadığını kontrol et"""
        rel_pos = world_point - camera_pos
        R = self._euler_to_rotation_matrix(*camera_orient)
        cam_coords = R.T @ rel_pos
        
        if cam_coords[0] <= 0:
            return False
            
        # Açısal kontrol
        half_fov = self.config.fov / 2
        aspect = self.config.width / self.config.height
        
        angle_h = np.degrees(np.arctan2(abs(cam_coords[1]), cam_coords[0]))
        angle_v = np.degrees(np.arctan2(abs(cam_coords[2]), cam_coords[0]))
        
        return angle_h <= half_fov and angle_v <= half_fov / aspect
        
    def apply_atmospheric_effects(self, 
                                   frame: np.ndarray,
                                   distance_map: np.ndarray = None) -> np.ndarray:
        """
        Atmosferik efektler uygula (haze/fog)
        
        Args:
            frame: RGB görüntü
            distance_map: Her piksel için mesafe (varsa)
            
        Returns:
            Efekt uygulanmış görüntü
        """
        if distance_map is None:
            return frame
            
        visibility = self.config.visibility
        haze_color = np.array(self.config.haze_color) * 255
        
        # Haze faktörü (mesafeye göre)
        haze_factor = 1 - np.exp(-distance_map / visibility)
        haze_factor = np.clip(haze_factor, 0, 0.8)
        
        # Haze uygula
        result = frame * (1 - haze_factor[:, :, np.newaxis]) + \
                 haze_color * haze_factor[:, :, np.newaxis]
                 
        return result.astype(np.uint8)
        
    def get_sun_direction(self) -> np.ndarray:
        """Güneş yönü vektörü"""
        az = np.radians(self.config.sun_azimuth)
        el = np.radians(self.config.sun_elevation)
        
        return np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])
        
    @staticmethod
    def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Euler açıları → Rotasyon matrisi (ZYX convention)"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        return R
        
    def update_config(self, **kwargs):
        """Konfigürasyon güncelle"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
        self._update_intrinsics()
