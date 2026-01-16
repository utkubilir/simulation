"""
Post-Processing Efektleri

Render edilmiş görüntüye gerçekçi kamera efektleri uygular:
- Lens distorsiyon (barrel/pincushion)
- Chromatic aberration
- Motion blur
- Sensor noise
- Vignette
- Auto exposure
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PostProcessingConfig:
    """Post-processing konfigürasyonu"""
    # Lens
    lens_distortion: bool = True
    k1: float = -0.15          # Radyal distorsiyon (barrel < 0)
    k2: float = 0.02           # İkinci derece
    
    # Chromatic aberration
    chromatic_aberration: bool = False
    chromatic_strength: float = 2.0
    
    # Motion blur
    motion_blur: bool = False
    motion_blur_strength: float = 0.5
    
    # Sensor noise
    sensor_noise: bool = True
    noise_sigma: float = 5.0
    
    # Vignette
    vignette: bool = True
    vignette_strength: float = 0.4
    
    # Depth of field
    dof: bool = False
    focus_distance: float = 100.0
    dof_strength: float = 1.0
    
    # Auto exposure
    auto_exposure: bool = False
    target_brightness: int = 128


class PostProcessing:
    """
    GPU/CPU tabanlı post-processing efektleri
    
    Gerçekçi kamera görüntüsü için efekt pipeline.
    
    Usage:
        pp = PostProcessing(PostProcessingConfig())
        processed = pp.process(frame, velocity=vel)
    """
    
    def __init__(self, config: PostProcessingConfig = None):
        self.config = config or PostProcessingConfig()
        
        # Pre-computed masks
        self._vignette_mask = None
        self._vignette_size = None
        
        # Exposure tracking
        self._current_exposure = 1.0
        
    def process(self, frame: np.ndarray,
                velocity: np.ndarray = None,
                focus_distance: float = None) -> np.ndarray:
        """
        Tüm post-processing efektlerini uygula
        
        Args:
            frame: RGB görüntü (H, W, 3)
            velocity: Kamera hızı vektörü (motion blur için)
            focus_distance: Odak mesafesi (DoF için)
            
        Returns:
            İşlenmiş RGB görüntü
        """
        if frame is None or frame.size == 0:
            return frame
            
        result = frame.copy()
        
        # 1. Lens distorsiyon
        if self.config.lens_distortion:
            result = self._apply_lens_distortion(result)
            
        # 2. Chromatic aberration
        if self.config.chromatic_aberration:
            result = self._apply_chromatic_aberration(result)
            
        # 3. Motion blur
        if self.config.motion_blur and velocity is not None:
            result = self._apply_motion_blur(result, velocity)
            
        # 4. Depth of field
        if self.config.dof and focus_distance is not None:
            result = self._apply_depth_of_field(result, focus_distance)
            
        # 5. Sensor noise
        if self.config.sensor_noise:
            result = self._apply_sensor_noise(result)
            
        # 6. Vignette
        if self.config.vignette:
            result = self._apply_vignette(result)
            
        # 7. Auto exposure
        if self.config.auto_exposure:
            result = self._apply_auto_exposure(result)
            
        return result
        
    def _apply_lens_distortion(self, frame: np.ndarray) -> np.ndarray:
        """
        Brown-Conrady lens distorsiyon modeli
        
        Barrel (k1 < 0) veya pincushion (k1 > 0) distorsiyon.
        """
        h, w = frame.shape[:2]
        
        # Kamera içsel matrisi
        fx = fy = w * 0.8
        cx, cy = w / 2, h / 2
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distorsiyon katsayıları: [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([
            self.config.k1,
            self.config.k2,
            0, 0,  # Tangential distortion
            0      # k3
        ], dtype=np.float64)
        
        # Optimal yeni kamera matrisi
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort (aslında distort yapmak için inverse gerekli)
        # OpenCV undistort simüle edilmiş distortion için
        result = cv2.undistort(frame, K, dist_coeffs, None, new_K)
        
        return result
        
    def _apply_chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """
        Kromatik aberasyon (renk kayması)
        
        Kenarlardan merkeze doğru RGB kanallarının kayması.
        """
        h, w = frame.shape[:2]
        strength = self.config.chromatic_strength
        
        # Maping koordinatları
        center_x, center_y = w / 2, h / 2
        
        # Red channel: dışa doğru
        map_r = self._create_radial_map(w, h, 1 + strength * 0.002)
        
        # Green channel: orijinal
        map_g_x, map_g_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Blue channel: içe doğru
        map_b = self._create_radial_map(w, h, 1 - strength * 0.002)
        
        # Kanalları ayrı remap et
        r = cv2.remap(frame[:, :, 0], map_r[0].astype(np.float32), 
                      map_r[1].astype(np.float32), cv2.INTER_LINEAR)
        g = frame[:, :, 1]
        b = cv2.remap(frame[:, :, 2], map_b[0].astype(np.float32), 
                      map_b[1].astype(np.float32), cv2.INTER_LINEAR)
        
        return np.stack([r, g, b], axis=2)
        
    def _create_radial_map(self, w: int, h: int, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """Radyal ölçekleme için koordinat haritası"""
        cx, cy = w / 2, h / 2
        
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Merkeze göre koordinatlar
        dx = x - cx
        dy = y - cy
        
        # Radyal ölçekleme
        map_x = cx + dx * scale
        map_y = cy + dy * scale
        
        return map_x, map_y
        
    def _apply_motion_blur(self, frame: np.ndarray, 
                           velocity: np.ndarray) -> np.ndarray:
        """
        Hız tabanlı motion blur
        
        Kamera hareket yönünde bulanıklık.
        """
        # Hız büyüklüğü
        speed = np.linalg.norm(velocity[:2])  # XY düzleminde
        if speed < 1:
            return frame
            
        # Bulanıklık miktarı (hıza orantılı)
        blur_length = int(min(30, speed * self.config.motion_blur_strength))
        if blur_length < 3:
            return frame
            
        # Hareket yönü
        angle = np.degrees(np.arctan2(velocity[1], velocity[0]))
        
        # Motion blur kernel
        kernel = self._create_motion_blur_kernel(blur_length, angle)
        
        # Uygula
        result = cv2.filter2D(frame, -1, kernel)
        
        return result
        
    def _create_motion_blur_kernel(self, length: int, angle: float) -> np.ndarray:
        """Yönlü motion blur kernel oluştur"""
        kernel = np.zeros((length, length))
        
        # Çizgi çiz
        center = length // 2
        cv2.line(kernel, 
                (center, center),
                (center + int(center * np.cos(np.radians(angle))),
                 center + int(center * np.sin(np.radians(angle)))),
                1.0, 1)
        cv2.line(kernel,
                (center, center),
                (center - int(center * np.cos(np.radians(angle))),
                 center - int(center * np.sin(np.radians(angle)))),
                1.0, 1)
                
        # Normalize
        kernel = kernel / kernel.sum()
        
        return kernel
        
    def _apply_depth_of_field(self, frame: np.ndarray, 
                               focus_distance: float) -> np.ndarray:
        """
        Depth of Field (alan derinliği) efekti
        
        Odak dışı alanlar bulanık.
        """
        # Basit DoF: sadece gaussian blur
        # Gerçek DoF için depth buffer gerekli
        
        blur_amount = int(self.config.dof_strength * 5)
        if blur_amount > 0:
            blurred = cv2.GaussianBlur(frame, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
            
            # Merkez net, kenarlar bulanık (basit simülasyon)
            h, w = frame.shape[:2]
            mask = self._create_dof_mask(w, h)
            
            result = (frame * mask + blurred * (1 - mask)).astype(np.uint8)
            return result
            
        return frame
        
    def _create_dof_mask(self, w: int, h: int) -> np.ndarray:
        """DoF için merkez maskesi"""
        Y, X = np.ogrid[:h, :w]
        center = (w / 2, h / 2)
        
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        
        # Merkezde 1, kenarlarda 0
        mask = 1 - np.clip(dist / max_dist, 0, 1)
        
        return mask[:, :, np.newaxis]
        
    def _apply_sensor_noise(self, frame: np.ndarray) -> np.ndarray:
        """
        Sensör gürültüsü (Gaussian noise)
        
        ISO değerine göre gürültü miktarı.
        """
        noise = np.random.normal(0, self.config.noise_sigma, frame.shape)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
        
        return noisy.astype(np.uint8)
        
    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """
        Vignette efekti (kenar karartma)
        
        Optik lens karakteristiği simülasyonu.
        """
        h, w = frame.shape[:2]
        
        # Mask cache
        if self._vignette_mask is None or self._vignette_size != (w, h):
            self._vignette_mask = self._create_vignette_mask(w, h)
            self._vignette_size = (w, h)
            
        result = frame * self._vignette_mask
        
        return result.astype(np.uint8)
        
    def _create_vignette_mask(self, w: int, h: int) -> np.ndarray:
        """Vignette maskesi oluştur"""
        Y, X = np.ogrid[:h, :w]
        center = (w / 2, h / 2)
        
        # Normalize mesafe
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        dist_norm = dist / max_dist
        
        # Vignette faktörü
        strength = self.config.vignette_strength
        mask = 1 - (dist_norm ** 2) * strength
        mask = np.clip(mask, 1 - strength, 1.0)
        
        return mask[:, :, np.newaxis]
        
    def _apply_auto_exposure(self, frame: np.ndarray) -> np.ndarray:
        """
        Otomatik pozlama ayarı
        
        Ortalama parlaklığı hedef değere yaklaştır.
        """
        # Mevcut parlaklık
        current_brightness = np.mean(frame)
        
        # Hedef farkı
        diff = self.config.target_brightness - current_brightness
        
        # Yumuşak geçiş
        self._current_exposure += diff * 0.02
        self._current_exposure = np.clip(self._current_exposure, 0.5, 2.0)
        
        # Uygula
        result = np.clip(frame * self._current_exposure, 0, 255)
        
        return result.astype(np.uint8)
        
    def update_config(self, **kwargs):
        """Konfigürasyon güncelle"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
        # Mask cache'i temizle
        self._vignette_mask = None
