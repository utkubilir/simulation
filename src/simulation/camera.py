"""
Sabit Monte Edilmiş Kamera Simülasyonu

Gimbal YOK - Kamera İHA'nın burnuna sabit monte edilmiş.
İHA'nın oryantasyonunu doğrudan takip eder.

Özellikler:
- Sabit montaj pozisyonu ve açısı
- Lens distorsiyon simülasyonu (barrel/pincushion)
- Kamera sarsıntısı (Perlin noise tabanlı)
- Motion blur desteği
- Chromatic aberration
- Lens flare
- Depth of Field (DoF)
- Sensör gürültüsü
- Auto Exposure
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import cv2
import functools

@functools.lru_cache(maxsize=128)
def get_motion_blur_kernel(blur_size: int, angle_rad_quantized: float) -> np.ndarray:
    """
    Cached motion blur kernel generation.
    angle_rad_quantized is expected to be rounded (e.g. to 2 decimal places)
    to maximize cache hits.
    """
    kernel = np.zeros((blur_size, blur_size), dtype=np.float32)

    mid = blur_size // 2
    for i in range(blur_size):
        x = int(mid + (i - mid) * np.cos(angle_rad_quantized))
        y = int(mid + (i - mid) * np.sin(angle_rad_quantized))
        if 0 <= x < blur_size and 0 <= y < blur_size:
            kernel[y, x] = 1.0

    kernel /= np.sum(kernel) if np.sum(kernel) > 0 else 1
    return kernel


try:
    from opensimplex import OpenSimplex
    OPENSIMPLEX_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn(
        "OpenSimplex not available. Camera shake will use sinusoidal fallback.", 
        ImportWarning
    )
    OPENSIMPLEX_AVAILABLE = False

from src.simulation.utils import euler_to_rotation_matrix
try:
    from src.rendering.renderer import GLRenderer
    OPENGL_AVAILABLE = True
except ImportError as e:
    print(f"Info: OpenGL Renderer not available ({e}). Using CPU rendering.")
    OPENGL_AVAILABLE = False


class FixedCamera:
    """
    İHA'ya sabit monte edilmiş kamera.
    
    Gimbal yok - kamera İHA'nın oryantasyonunu doğrudan takip eder.
    Gerçekçi lens efektleri ve titreşim simülasyonu içerir.
    """
    
    
    def __init__(self, position: List[float], config: dict = None):
        """
        Args:
            position: [x, y, z] kamera konumu (NED frame)
            config: Konfigürasyon sözlüğü
        """
        self.position = np.array(position, dtype=np.float32)
        config = config or {}
        
        # Temel parametreler
        self.fov = config.get('fov', 60.0)  # derece
        self.resolution = tuple(config.get('resolution', (640, 480)))
        self.fps = config.get('fps', 30)
        self.distortion_clamp = config.get('distortion_clamp', False)
        
        # OpenGL Renderer
        self.renderer = None
        if OPENGL_AVAILABLE:
            try:
                # Renderer'ı başlat (Context var varsayılır)
                self.renderer = GLRenderer(width=self.resolution[0], height=self.resolution[1])
            except Exception as e:
                print(f"Failed to initialize OpenGL Renderer (Maybe no context?): {e}")
                self.renderer = None
        
        # Environment referansı (terrain + world objects)
        self._environment = None
        self._environment_initialized = False
        
        # Sabit montaj pozisyonu (İHA gövdesine göre)
        # [ileri, sağ, aşağı] metre cinsinden
        mount_offset = config.get('mount_offset', [0.3, 0.0, -0.1])
        self.mount_offset = np.array(mount_offset)
        
        # Sabit montaj açısı (kameranın İHA'ya göre pitch açısı)
        # Negatif = aşağı bakış (daha büyük negatif = daha fazla aşağı)
        self.mount_pitch = np.radians(config.get('mount_pitch', -30.0))
        
        # Lens distorsiyon parametreleri (Brown-Conrady model)
        self.distortion_enabled = config.get('distortion_enabled', True)
        self.k1 = config.get('k1', -0.1)   # Radyal distorsiyon (barrel < 0, pincushion > 0)
        self.k2 = config.get('k2', 0.02)   # İkinci derece radyal
        self.k3 = config.get('k3', 0.0)    # Üçüncü derece radyal (opsiyonel)
        self.p1 = config.get('p1', 0.0)    # Tangential distorsiyon
        self.p2 = config.get('p2', 0.0)    # Tangential distorsiyon
        
        # Kamera sarsıntısı (Perlin noise tabanlı)
        self.shake_enabled = config.get('shake_enabled', True)
        self.shake_intensity = config.get('shake_intensity', 0.005)  # radyan
        self._noise_time = 0.0
        self._shake_offset = np.zeros(2)  # pitch, yaw offset
        
        # Perlin noise generator (gerçekçi titreşim için)
        self._noise_seed = config.get('noise_seed', 42)
        if OPENSIMPLEX_AVAILABLE:
            self._noise_gen_x = OpenSimplex(seed=self._noise_seed)
            self._noise_gen_y = OpenSimplex(seed=self._noise_seed + 1)
        else:
            self._noise_gen_x = None
            self._noise_gen_y = None
        
        # Motion blur
        self.motion_blur_enabled = config.get('motion_blur', True)  # Varsayılan AÇIK
        self.exposure_time = config.get('exposure_time', 0.01)  # saniye
        self.blur_strength = config.get('blur_strength', 3)  # Daha düşük
        
        # Chromatic aberration
        self.chromatic_aberration_enabled = config.get('chromatic_aberration', False)
        self.chromatic_strength = config.get('chromatic_strength', 2.0)
        
        # Lens flare
        self.lens_flare_enabled = config.get('lens_flare', False)
        self.sun_direction = np.array(config.get('sun_direction', [0.5, 0.3, 0.8]))
        self.sun_direction = self._normalize_vector(self.sun_direction)
        self.sun_intensity = config.get('sun_intensity', 1.0)
        self.sun_angular_radius = config.get('sun_angular_radius', 0.01)
        
        # Görsel efektler
        self.vignette_enabled = config.get('vignette_enabled', True)
        self.haze_enabled = config.get('haze_enabled', True)
        self.haze_distance = config.get('haze_distance', 500.0)  # metre (Increased from 250 for clarity)
        
        # Depth of Field (odak bulanıklığı)
        self.dof_enabled = config.get('dof_enabled', False)
        self.focus_distance = config.get('focus_distance', 100.0)  # metre
        self.dof_strength = config.get('dof_strength', 1.0)  # bulanıklık şiddeti
        
        # Sensör gürültüsü
        self.sensor_noise_enabled = config.get('sensor_noise', True)  # Varsayılan AÇIK
        self.iso = config.get('iso', 100)  # ISO değeri (Clean, daylight)
        
        # Auto Exposure
        self.auto_exposure_enabled = config.get('auto_exposure', False)
        self.target_brightness = config.get('target_brightness', 128)
        self._current_exposure = 1.0  # Dinamik exposure faktörü
        
        # Tonemapping (HDR)
        self.tonemapping_enabled = config.get('tonemapping_enabled', True)
        
        # Color Grading (New)
        self.color_grading_enabled = config.get('color_grading_enabled', True)
        # BGR Scale factors: (Blue gain, Green gain, Red gain)
        # Subtler tint (was [1.05, 1.1, 0.9])
        self.color_grading_values = config.get('color_grading_values', [1.02, 1.05, 0.95]) 
        
        # Lens Softness (New - remove "CG sharp" look)
        self.softness_enabled = config.get('softness_enabled', True)
        self.softness_strength = config.get('softness_strength', 0.2)  # Reduced from 0.5
        
        # Atmosferik efektler (Yağmur/Kar)
        self.rain_enabled = config.get('rain_enabled', False)
        self.rain_density = config.get('rain_density', 0.5)
        self.snow_enabled = config.get('snow_enabled', False)
        self.snow_density = config.get('snow_density', 0.5)
        
        # Rolling shutter için durum
        self._prev_camera_orient = None
        self._angular_velocity = np.zeros(3)
        
        # Kamera intrinsic matrisi
        self.K = None
        self._update_intrinsics()
        
        
        # Load Assets for Realistic Rendering
        self.sky_texture = None
        self.ground_texture = None

        # İstatistikler
        self.frame_count = 0
        
        # Precomputed noise pool for performance optimization
        # (avoids per-frame np.random.normal which is expensive)
        self._noise_pool_size = 16  # Number of precomputed noise frames
        self._noise_pool = None
        self._noise_pool_index = 0
        self._init_noise_pool()
    
    def _init_noise_pool(self):
        """Precompute noise textures for faster sensor noise application."""
        w, h = self.resolution
        noise_level = np.sqrt(self.iso / 100.0) * 8.0
        
        # Generate pool of noise frames
        self._noise_pool = np.random.normal(
            0, noise_level, (self._noise_pool_size, h, w, 3)
        ).astype(np.float32)
        
    def _update_intrinsics(self):
        """Kamera içsel matrisini güncelle"""
        w, h = self.resolution
        # Focal length from FOV
        fx = w / (2 * np.tan(np.radians(self.fov / 2)))
        fy = fx  # Kare piksel varsayımı
        cx, cy = w / 2, h / 2
        
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.focal_length = fx
        self.fov_h = np.degrees(2 * np.arctan(w / (2 * fx)))
        self.fov_v = np.degrees(2 * np.arctan(h / (2 * fy)))

    def load_calibration(self, yaml_path: str):
        """
        OpenCV kalibrasyon dosyasından kamera parametrelerini yükle.
        
        Args:
            yaml_path: YAML dosya yolu (OpenCV FileStorage formatı)
        """
        import yaml
        from pathlib import Path
        
        path = Path(yaml_path)
        if not path.exists():
            print(f"Warning: Calibration file not found: {yaml_path}")
            return
            
        try:
            with open(path, 'r') as f:
                # OpenCV YAML formatı bazen özel tagler içerir, safe_load ile dene
                # Genelde {camera_matrix: {data: [...]}, dist_coeff: ...}
                data = yaml.safe_load(f)
                
            if not data:
                return
                
            # Kamera Matrisi (K)
            if 'camera_matrix' in data:
                km = data['camera_matrix']
                if 'data' in km:
                    k_data = np.array(km['data']).reshape(3, 3)
                    self.K = k_data
                    self.focal_length = (self.K[0,0] + self.K[1,1]) / 2
                    
                    # FOV güncelle
                    w, h = self.resolution
                    self.fov_h = np.degrees(2 * np.arctan(w / (2 * self.K[0,0])))
                    
            # Distorsiyon Katsayıları
            if 'dist_coeff' in data:
                dc = data['dist_coeff']
                if 'data' in dc:
                    d_data = dc['data']
                    self.k1 = d_data[0] if len(d_data) > 0 else 0
                    self.k2 = d_data[1] if len(d_data) > 1 else 0
                    self.p1 = d_data[2] if len(d_data) > 2 else 0
                    self.p2 = d_data[3] if len(d_data) > 3 else 0
                    self.k3 = d_data[4] if len(d_data) > 4 else 0
                    
            print(f"Loaded calibration from {yaml_path}")
            
        except Exception as e:
            print(f"Error loading calibration: {e}")

    def set_resolution(self, width: int, height: int):
        """Kamera çözünürlüğünü güncelle ve intrinsics hesapla."""
        self.resolution = (int(width), int(height))
        self._update_intrinsics()

    def set_fov(self, fov: float):
        """Kamera FOV değerini güncelle ve intrinsics hesapla."""
        self.fov = float(fov)
        self._update_intrinsics()

    def set_distortion_params(self, k1: float = None, k2: float = None,
                              k3: float = None, p1: float = None, p2: float = None):
        """Lens distorsiyon parametrelerini güncelle."""
        if k1 is not None:
            self.k1 = float(k1)
        if k2 is not None:
            self.k2 = float(k2)
        if k3 is not None:
            self.k3 = float(k3)
        if p1 is not None:
            self.p1 = float(p1)
        if p2 is not None:
            self.p2 = float(p2)

    def set_sun_direction(self, direction: np.ndarray):
        """Update sun direction (normalized)."""
        self.sun_direction = self._normalize_vector(np.array(direction))

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector, returning default forward vector [1,0,0] if zero-length."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:  # Near-zero threshold instead of exact zero
            return np.array([1.0, 0.0, 0.0])  # Default forward direction
        return vec / norm
    
    def set_environment(self, environment):
        """
        Kamera için environment referansını ayarla.
        OpenGL renderer varsa GPU kaynaklarını hazırlar.
        
        Args:
            environment: Environment objesi (terrain + world objects)
        """
        self._environment = environment
        
        # OpenGL Renderer varsa environment'ı initialize et
        if self.renderer and not self._environment_initialized:
            try:
                self.renderer.init_environment(environment)
                self._environment_initialized = True
            except Exception as e:
                print(f"Failed to initialize environment in renderer: {e}")
    
    def set_arena(self, arena):
        """
        Kamera için arena referansını ayarla.
        OpenGL renderer varsa GPU kaynaklarını hazırlar.
        
        Args:
            arena: TeknofestArena objesi (sınırlar, safe zones, markers)
        """
        self._arena = arena
        
        # OpenGL Renderer varsa arena'yı initialize et
        if self.renderer and arena:
            try:
                self.renderer.init_arena(arena)
            except Exception as e:
                print(f"Failed to initialize arena in renderer: {e}")
        
    def update(self, dt: float, uav_velocity: np.ndarray = None):
        """
        Kamera durumunu güncelle (her frame çağrılmalı)
        
        Args:
            dt: Zaman adımı (saniye)
            uav_velocity: İHA hız vektörü (sarsıntı şiddeti için)
        """
        self._noise_time += dt
        
        if self.shake_enabled:
            # İHA hızına göre sarsıntı şiddetini ayarla
            speed = np.linalg.norm(uav_velocity) if uav_velocity is not None else 20.0
            intensity = self.shake_intensity * (0.5 + speed / 40.0)
            
            # Perlin noise kullan (varsa), yoksa fallback sinüs
            if self._noise_gen_x is not None and self._noise_gen_y is not None:
                # Gerçek Perlin noise - çok daha organik
                # Düşük frekans: büyük hareketler
                low_freq = 3.0
                self._shake_offset[0] = intensity * self._noise_gen_x.noise2(
                    self._noise_time * low_freq, 0.0)
                self._shake_offset[1] = intensity * self._noise_gen_y.noise2(
                    self._noise_time * low_freq, 0.0)
                
                # Yüksek frekans: ince titreşimler
                high_freq = 12.0
                self._shake_offset[0] += intensity * 0.4 * self._noise_gen_x.noise2(
                    self._noise_time * high_freq, 1.0)
                self._shake_offset[1] += intensity * 0.4 * self._noise_gen_y.noise2(
                    self._noise_time * high_freq, 1.0)
            else:
                # Fallback: basit sinüs (OpenSimplex yoksa)
                freq = 5.0
                self._shake_offset[0] = intensity * np.sin(self._noise_time * freq * 2.1 + 1.3)
                self._shake_offset[1] = intensity * np.sin(self._noise_time * freq * 1.7 + 2.7)
                
                high_freq = 15.0
                self._shake_offset[0] += intensity * 0.3 * np.sin(self._noise_time * high_freq * 3.1)
                self._shake_offset[1] += intensity * 0.3 * np.sin(self._noise_time * high_freq * 2.3)
        else:
            self._shake_offset = np.zeros(2)
            
        self.frame_count += 1
        
    def get_camera_pose(self, uav_position: np.ndarray, 
                        uav_orientation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kameranın dünya koordinatlarındaki pozisyon ve oryantasyonunu hesapla
        
        Args:
            uav_position: İHA pozisyonu [x, y, z]
            uav_orientation: İHA oryantasyonu [roll, pitch, yaw] radyan
            
        Returns:
            (camera_position, camera_orientation) tuple
        """
        roll, pitch, yaw = uav_orientation
        
        # İHA dönüşüm matrisi
        # İHA'nın oryantasyon matrisi
        R_uav = euler_to_rotation_matrix(roll, pitch, yaw)
        
        # Kamera pozisyonu: İHA pozisyonu + döndürülmüş offset
        camera_pos = uav_position + R_uav @ self.mount_offset
        
        # Kamera oryantasyonu: İHA oryantasyonu + montaj açısı + sarsıntı
        camera_roll = roll
        camera_pitch = pitch + self.mount_pitch + self._shake_offset[0]
        camera_yaw = yaw + self._shake_offset[1]
        
        camera_orient = np.array([camera_roll, camera_pitch, camera_yaw])
        
        return camera_pos, camera_orient
        
    def project_point(self, world_point: np.ndarray,
                      camera_pos: np.ndarray,
                      camera_orient: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        3D noktayı ekran koordinatlarına projeksiyonla
        
        Args:
            world_point: Dünya koordinatlarında 3D nokta
            camera_pos: Kamera pozisyonu
            camera_orient: Kamera oryantasyonu [roll, pitch, yaw]
            
        Returns:
            (x, y) ekran koordinatları veya görüş dışında ise None
        """
        # Kamera koordinat sistemine dönüşüm
        rel_pos = world_point - camera_pos
        
        # Kamera dönüşüm matrisi
        R = euler_to_rotation_matrix(*camera_orient)
        cam_coords = R @ rel_pos  # World -> Camera dönüşümü (R.T değil R)
        
        # Kameranın arkasında mı?
        if cam_coords[0] <= 0.1:  # Minimum mesafe
            return None
            
        # Perspektif projeksiyon
        x = self.K[0, 0] * cam_coords[1] / cam_coords[0] + self.K[0, 2]
        y = self.K[1, 1] * cam_coords[2] / cam_coords[0] + self.K[1, 2]
        
        # Lens distorsiyon uygula
        if self.distortion_enabled:
            x, y = self._apply_distortion(x, y)

        # Ekran sınırları
        w, h = self.resolution
        if self.distortion_clamp:
            x = float(np.clip(x, 0, w - 1))
            y = float(np.clip(y, 0, h - 1))
            return (x, y)
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return None
        
    def _apply_distortion(self, x: float, y: float) -> Tuple[float, float]:
        """Lens distorsiyonu uygula (barrel/pincushion)"""
        w, h = self.resolution
        cx, cy = w / 2, h / 2
        
        # Normalize koordinatlar
        x_norm = (x - cx) / self.focal_length
        y_norm = (y - cy) / self.focal_length
        
        # Radyal mesafe
        r2 = x_norm**2 + y_norm**2
        r4 = r2 * r2
        
        # Radyal distorsiyon faktörü
        distortion_factor = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r2 * r4

        # Tangential distorsiyon
        x_tangential = 2 * self.p1 * x_norm * y_norm + self.p2 * (r2 + 2 * x_norm**2)
        y_tangential = self.p1 * (r2 + 2 * y_norm**2) + 2 * self.p2 * x_norm * y_norm

        # Distorsiyon uygula
        x_dist = x_norm * distortion_factor + x_tangential
        y_dist = y_norm * distortion_factor + y_tangential
        
        # Piksel koordinatlarına geri dönüştür
        x_out = x_dist * self.focal_length + cx
        y_out = y_dist * self.focal_length + cy
        
        return x_out, y_out
    
    def undistort_point(self, x_dist: float, y_dist: float, 
                        iterations: int = 10) -> Tuple[float, float]:
        """
        Distorted koordinatları ideal koordinatlara çevir (Newton-Raphson iterasyonu).
        
        Detection sonuçlarını kamera kalibrasyonu için ideal koordinatlara dönüştürürken kullanılır.
        
        Args:
            x_dist: Distorted x koordinatı (piksel)
            y_dist: Distorted y koordinatı (piksel)
            iterations: Newton-Raphson iterasyon sayısı
            
        Returns:
            (x_ideal, y_ideal) ideal koordinatlar
        """
        w, h = self.resolution
        cx, cy = w / 2, h / 2
        
        # Normalize
        x_norm = (x_dist - cx) / self.focal_length
        y_norm = (y_dist - cy) / self.focal_length
        
        # İlk tahmin: distorted = ideal (distorsiyon küçükse iyi başlangıç)
        x_u = x_norm
        y_u = y_norm
        
        for _ in range(iterations):
            # Mevcut tahmin için distorsiyon hesapla
            r2 = x_u**2 + y_u**2
            r4 = r2 * r2
            
            # Radyal distorsiyon faktörü
            k = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r2 * r4
            
            # Tangential
            dx = 2 * self.p1 * x_u * y_u + self.p2 * (r2 + 2 * x_u**2)
            dy = self.p1 * (r2 + 2 * y_u**2) + 2 * self.p2 * x_u * y_u
            
            # Distorted tahmin
            x_d = x_u * k + dx
            y_d = y_u * k + dy
            
            # Hata
            err_x = x_norm - x_d
            err_y = y_norm - y_d
            
            # Güncelle (basit gradient descent yaklaşımı)
            x_u += err_x
            y_u += err_y
            
            # Convergence kontrolü
            if abs(err_x) < 1e-7 and abs(err_y) < 1e-7:
                break
        
        # Piksel koordinatlarına dönüştür
        x_ideal = x_u * self.focal_length + cx
        y_ideal = y_u * self.focal_length + cy
        
        return x_ideal, y_ideal
    
    def apply_rolling_shutter(self, frame: np.ndarray, 
                              angular_velocity: np.ndarray,
                              readout_time: float = 0.02) -> np.ndarray:
        """
        Rolling shutter efekti simülasyonu.
        
        Hızlı hareket eden nesnelerde "jello effect" oluşturur.
        Her satır farklı bir zamanda okunur, bu da eğik görüntülere yol açar.
        
        Args:
            frame: Giriş görüntüsü
            angular_velocity: Kamera açısal hızı [roll_rate, pitch_rate, yaw_rate] (rad/s)
            readout_time: Tüm frame'in okunma süresi (saniye)
            
        Returns:
            Rolling shutter efekti uygulanmış görüntü
        """
        if angular_velocity is None or np.linalg.norm(angular_velocity) < 0.01:
            return frame
            
        h, w = frame.shape[:2]
        angular_velocity = np.array(angular_velocity)
        
        # Yaw ve pitch değişimi daha görünür efekt yaratır
        yaw_rate = angular_velocity[2]   # Yatay kayma
        pitch_rate = angular_velocity[1]  # Dikey kayma
        
        # Maksimum piksel kayması (efekt şiddeti)
        max_shift_x = yaw_rate * readout_time * self.focal_length * 0.5
        max_shift_y = pitch_rate * readout_time * self.focal_length * 0.5
        
        # Her satır için zaman offset'i
        result = np.zeros_like(frame)
        
        for row in range(h):
            # Bu satırın okunma zamanı (0 = üst, 1 = alt)
            t = row / h
            
            # Shift miktarı (parabolik profil daha gerçekçi)
            shift_x = int(max_shift_x * (t - 0.5) * 2)
            shift_y = int(max_shift_y * (t - 0.5) * 2)
            
            # Satırı kaydır
            if shift_x != 0:
                if shift_x > 0:
                    result[row, shift_x:] = frame[row, :-shift_x] if shift_x < w else 0
                else:
                    result[row, :shift_x] = frame[row, -shift_x:] if -shift_x < w else 0
            else:
                result[row] = frame[row]
        
        return result
        

        
    def calculate_apparent_size(self, distance: float, object_size: float = 2.0) -> float:
        """
        Hedefin ekrandaki görünür boyutunu hesapla
        
        Args:
            distance: Hedefe mesafe (metre)
            object_size: Gerçek nesne boyutu (metre)
            
        Returns:
            Piksel cinsinden görünür boyut
        """
        if distance <= 0:
            return 0
        return self.focal_length * object_size / distance
        
    def is_in_fov(self, world_point: np.ndarray, 
                  camera_pos: np.ndarray,
                  camera_orient: np.ndarray) -> bool:
        """Noktanın görüş alanında olup olmadığını kontrol et"""
        rel_pos = world_point - camera_pos
        R = euler_to_rotation_matrix(*camera_orient)
        cam_coords = R @ rel_pos  # World -> Camera dönüşümü (düzeltildi)
        
        if cam_coords[0] <= 0:
            return False
            
        # Açısal kontrol
        angle_h = np.degrees(np.arctan2(abs(cam_coords[1]), cam_coords[0]))
        angle_v = np.degrees(np.arctan2(abs(cam_coords[2]), cam_coords[0]))
        
        return angle_h <= self.fov_h / 2 and angle_v <= self.fov_v / 2
    
    def check_occlusion(self, target_pos: np.ndarray, 
                        all_objects: List[Dict],
                        camera_pos: np.ndarray,
                        target_id: str = None,
                        occlusion_radius: float = 3.0) -> Tuple[bool, Optional[str]]:
        """
        Hedefin başka nesneler tarafından kapatılıp kapatılmadığını kontrol et.
        
        Ray-sphere intersection kullanarak basit occlusion testi yapar.
        
        Args:
            target_pos: Hedef pozisyonu [x, y, z]
            all_objects: Tüm nesnelerin listesi [{'id': str, 'position': [x,y,z], 'size': float}, ...]
            camera_pos: Kamera pozisyonu [x, y, z]
            target_id: Hedefin ID'si (kendisini atlama için)
            occlusion_radius: Nesne engelleyici yarıçapı (metre)
            
        Returns:
            (is_occluded: bool, occluder_id: Optional[str])
        """
        target_pos = np.array(target_pos)
        camera_pos = np.array(camera_pos)
        
        # Kameradan hedefe ray
        ray_dir = target_pos - camera_pos
        target_dist = np.linalg.norm(ray_dir)
        
        if target_dist < 0.1:
            return False, None
            
        ray_dir = ray_dir / target_dist  # Normalize
        
        for obj in all_objects:
            obj_id = obj.get('id', '')
            
            # Kendisini atla
            if obj_id == target_id:
                continue
                
            obj_pos = np.array(obj.get('position', [0, 0, 0]))
            obj_size = obj.get('size', 2.0)
            sphere_radius = max(occlusion_radius, obj_size)
            
            # Nesnenin kameraya mesafesi
            obj_dist = np.linalg.norm(obj_pos - camera_pos)
            
            # Nesne hedeften uzakta ise atla
            if obj_dist >= target_dist:
                continue
            
            # Ray-Sphere intersection testi
            # Sphere center relative to ray origin
            oc = obj_pos - camera_pos
            
            # Projeksiyon - ray üzerindeki en yakın nokta
            t = np.dot(oc, ray_dir)
            
            if t < 0:
                continue  # Nesne kameranın arkasında
                
            # Ray'e en yakın nokta
            closest_point = camera_pos + t * ray_dir
            
            # Mesafe kontrolü
            distance_to_ray = np.linalg.norm(obj_pos - closest_point)
            
            if distance_to_ray < sphere_radius:
                return True, obj_id
                
        return False, None
        
    def generate_synthetic_frame(self, uav_states: list,
                               camera_pos: np.ndarray,
                               camera_orient: np.ndarray,
                               own_velocity: np.ndarray = None) -> np.ndarray:
        """
        Sentetik kamera görüntüsü oluştur (OpenGL veya CPU)
        
        Args:
            uav_states: Hedef İHA listesi
            camera_pos: Kamera pozisyonu
            camera_orient: Kamera oryantasyonu
            own_velocity: Kendi İHA hızı
            
        Returns:
            RGB numpy array (h, w, 3)
        """
        # OpenGL Render Path
        if self.renderer:
            # --- SHADOW PASS ---
            if hasattr(self.renderer, 'begin_shadow_pass'):
                self.renderer.begin_shadow_pass()
                
                # Render Scene for Shadows (Only casters)
                for uav in uav_states:
                    pos = np.array(uav['position'])
                    # Simple FOV check for shadows? Maybe skippable or use Light FOV.
                    # For simplicity, render all close enough UAVs.
                    
                    heading = np.radians(uav.get('heading', 0.0))
                    roll = np.radians(uav.get('roll', 0.0))
                    pitch = np.radians(uav.get('pitch', 0.0))
                    
                    self.renderer.render_aircraft(
                        position=pos,
                        heading=heading,
                        roll=roll,
                        pitch=pitch,
                        program=self.renderer.prog_shadow # Use Shadow Shader
                    )
            
            # --- MAIN PASS ---
            # 1. Kamera Güncelle (environment rendering için önce yapılmalı)
            # Apply mount_pitch offset to camera orientation (UAV'a göre aşağı bakış)
            adjusted_orient = np.array(camera_orient, dtype=np.float32).copy()
            adjusted_orient[1] += self.mount_pitch  # pitch'e ekle (radyan)
            
            # Coordinate Swap for GL (Sim Z=Altitude -> GL Y=Up)
            # Sim: [X, Y, Altitude] -> GL: [X, Altitude, Y] (assuming Y is depth)
            # Note: Checking debug results, [500, 200, 500] worked where Y=200 was altitude.
            # So GL expects [X, Height, Depth].
            # Sim input is likely [X, Y, Height] or [X, Y, Z].
            # We map Sim [0] -> GL X, Sim [2] -> GL Y, Sim [1] -> GL Z.
            gl_camera_pos = np.array([camera_pos[0], camera_pos[2], camera_pos[1]], dtype=np.float32)
            
            # Debug override
            # gl_camera_pos = np.array([500.0, 200.0, 500.0], dtype=np.float32)

            self.renderer.update_camera(position=gl_camera_pos, rotation=adjusted_orient)
            
            # 2. Sahne Başlat (environment + sky rendering)
            self.renderer.begin_frame()
            
            
            # 3. İHA'ları Çiz
            for uav in uav_states:
                pos = np.array(uav['position'])
                
                # Görünürlük kontrolü (Basit frustum culling)
                if not self.is_in_fov(pos, camera_pos, camera_orient):
                    continue
                    
                # Occlusion kontrolü
                is_occluded, _ = self.check_occlusion(pos, uav_states, camera_pos, target_id=uav['id'])
                if is_occluded:
                    continue
                
                # İHA Yönelimi
                heading = np.radians(uav.get('heading', 0.0))
                roll = np.radians(uav.get('roll', 0.0))
                pitch = np.radians(uav.get('pitch', 0.0))
                
                # Renk: Player (Kırmızı), Enemy (Mavi)
                color = (1.0, 0.0, 0.0) if uav.get('is_player', False) else (0.0, 0.0, 1.0)
                
                self.renderer.render_aircraft(
                    position=pos,
                    heading=heading,
                    roll=roll,
                    pitch=pitch,
                    color=color
                )
            
            # 4. Sahne Bitir (Post-Process & Draw)
            self.renderer.end_frame(time=self._noise_time)
            
            # 5. CPU'ya Oku
            frame = self.renderer.read_pixels()
            
            # Açısal hız hesabı (Noise ve Rolling Shutter için)
            if self._prev_camera_orient is not None:
                dt_est = 1.0 / self.fps if self.fps > 0 else 0.033
                diff = camera_orient - self._prev_camera_orient
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                self._angular_velocity = diff / dt_est
            self._prev_camera_orient = camera_orient.copy()
            
            # 6. CPU Bazlı Efektler (Hali hazırda Shader'da olmayanlar)
            
            # Rolling Shutter (Opsiyonel - Shader ile de yapılabilir ama şimdilik CPU)
            # frame = self.apply_rolling_shutter(frame, self._angular_velocity)
            
            # Sensor Noise (Now handled in Shader)
            # if self.sensor_noise_enabled:
            #     frame = self._apply_sensor_noise(frame)
                
            return frame
            
        # Fallback: CPU Render Path
        else:
            return self._generate_synthetic_frame_cpu(uav_states, camera_pos, camera_orient, own_velocity)
        
    def _generate_synthetic_frame_cpu(self, uav_states: list,
                                 camera_pos: np.ndarray,
                                 camera_orient: np.ndarray,
                                 own_velocity: np.ndarray = None) -> np.ndarray:
        """
        Gerçekçi sentetik görüntü oluştur
        
        Args:
            uav_states: Hedef İHA'ların durumları
            camera_pos: Kamera pozisyonu
            camera_orient: Kamera oryantasyonu
            own_velocity: Kendi İHA hızı (motion blur için)
            
        Returns:
            RGB numpy array (h, w, 3)
        """
        w, h = self.resolution
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 1. Background (Sky, Clouds, Ground Grid, Sun)
        frame = self._render_background(frame, camera_pos, camera_orient)
        
        # 2. Gölgeler (Zemin üzerine)
        for uav in uav_states:
            self._render_shadow(frame, uav, camera_pos, camera_orient)
            
        # 3. Hedef İHA'lar (3D Render) - Painter's Algorithm (Uzak -> Yakın)
        uav_states_sorted = sorted(uav_states, 
            key=lambda u: np.linalg.norm(np.array(u['position']) - camera_pos), 
            reverse=True)
            
        for uav in uav_states_sorted:
            self._render_uav_target_3d(frame, uav, camera_pos, camera_orient)
            
        # 4. Depth of Field (hedef mesafelerine göre)
        if self.dof_enabled and uav_states:
            frame = self._apply_depth_of_field(frame, uav_states, camera_pos)
            
        # 5. Post-processing efektleri
        # 4.5. Atmosferik Efektler (Rain/Snow)
        if self.rain_enabled:
            frame = self._apply_rain_snow(frame, 'rain', self.rain_density)
        if self.snow_enabled:
            frame = self._apply_rain_snow(frame, 'snow', self.snow_density)

        # 5. Post-processing efektleri
        
        # Açısal hız hesabı (Rolling shutter için)
        if self._prev_camera_orient is not None:
            # Basit fark (frame hızı 60fps varsayımı - veya dt parametresi eklenmeli)
            # generate_synthetic_frame dt almıyor, o yüzden yaklaşık hesap
            # Veya update() metodunda hesaplanıp saklanmalı.
            # Şimdilik basit fark (dt=0.016 gibi)
            dt_est = 1.0 / self.fps if self.fps > 0 else 0.033
            diff = camera_orient - self._prev_camera_orient
            # Wrap angles
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            self._angular_velocity = diff / dt_est
        self._prev_camera_orient = camera_orient.copy()
        
        # Rolling Shutter (Hızlı hareketlerde jello effect)
        frame = self.apply_rolling_shutter(frame, self._angular_velocity)
        
        if self.motion_blur_enabled and own_velocity is not None:
            frame = self._apply_motion_blur(frame, own_velocity, camera_orient)
            
        if self.chromatic_aberration_enabled:
            frame = self._apply_chromatic_aberration(frame)
            
        if self.lens_flare_enabled:
            frame = self._apply_lens_flare(frame, camera_orient)
            
        if self.vignette_enabled:
            frame = self._apply_vignette(frame)
            
        if self.sensor_noise_enabled:
            frame = self._apply_sensor_noise(frame)
            
        if self.auto_exposure_enabled:
            frame = self._apply_auto_exposure(frame)
            
        if self.tonemapping_enabled:
            frame = self._apply_tonemapping(frame)
            
        if self.color_grading_enabled:
            frame = self._apply_color_grading(frame)
            
        if self.softness_enabled:
            frame = self._apply_lens_softness(frame)
            
        return frame

    def _get_uav_mesh(self, size: float) -> Tuple[np.ndarray, List[List[int]]]:
        """Basit 3D İHA Modeli (Vertices, Faces)"""
        # Yerel koordinatlar (x=ileri, y=sağ, z=aşağı)
        # Gövde (kutu benzeri)
        l, w, h = size, size*0.15, size*0.15
        
        vertices = np.array([
            [l/2, 0, 0],   # 0: Burun
            [-l/2, w, -h], # 1: Arka Üst Sağ
            [-l/2, -w, -h],# 2: Arka Üst Sol
            [-l/2, 0, h],  # 3: Arka Alt
            
            # Kanatlar
            [0, size, 0],   # 4: Sağ Kanat Ucu
            [0, -size, 0],  # 5: Sol Kanat Ucu
            
            # Kuyruk
            [-l/2 - size*0.3, 0, -size*0.3] # 6: Kuyruk Tepe
        ])
        
        # Yüzler (vertex indisleri)
        faces = [
            [0, 1, 2], # Üst Gövde
            [0, 1, 3], # Sağ Yan
            [0, 2, 3], # Sol Yan
            [1, 2, 3], # Arka Kapak
            # [1, 2, 4, 5], # Kanatlar (basit quad/triangles) -> aslında gövdeye bağlı olmalı
            # Daha düzgün kanat:
            [0, 4, 1], # Sağ Kanat Ön
            [0, 5, 2], # Sol Kanat Ön
            
            [1, 2, 6]  # Kuyruk
        ]
        
        return vertices, faces

    def _render_uav_target_3d(self, frame: np.ndarray, uav: dict,
                              camera_pos: np.ndarray, camera_orient: np.ndarray):
        """3D Wireframe/Solid Render"""
        uav_pos = np.array(uav['position'])
        size = uav.get('size', 2.0)
        heading = np.radians(uav.get('heading', 0.0))
        
        # 1. Model oluştur
        vertices, faces = self._get_uav_mesh(size)
        
        # 2. Döndür (Heading - Yaw)
        c, s = np.cos(heading), np.sin(heading)
        R_model = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        vertices_world = (R_model @ vertices.T).T + uav_pos
        
        # 3. Projeksiyon
        projections = self._project_points(vertices_world, camera_pos, camera_orient)
        
        # Görünürlük kontrolü
        if np.isnan(projections).all():
            return
            
        # 4. Çiz
        base_color = (0, 0, 200) if uav.get('team') == 'red' else (200, 100, 0)
        
        # Mesafeye göre renk (haze)
        dist = np.linalg.norm(uav_pos - camera_pos)
        if self.haze_enabled:
            haze_factor = min(1.0, dist / self.haze_distance)
            sky_color = (200, 180, 160)
            color = tuple(int(base_color[i] * (1 - haze_factor * 0.8) + sky_color[i] * haze_factor * 0.8) 
                         for i in range(3))
        else:
            color = base_color
            
        # Pervane hareketi (propeller blur)
        nose_idx = 0
        if not np.isnan(projections[nose_idx]).any():
            center = (int(projections[nose_idx][0]), int(projections[nose_idx][1]))
            radius = int(200 * size / dist) # Mesafeye göre ölçekle
            if radius > 2:
                overlay = frame.copy()
                cv2.circle(overlay, center, radius, (200, 200, 200), -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Yüzeyleri Çiz
        for face in faces:
            pts = projections[face]
            if np.isnan(pts).any(): continue
            pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
            
            # Basit shading (Lambertian)
            v1 = vertices_world[face[1]] - vertices_world[face[0]]
            v2 = vertices_world[face[2]] - vertices_world[face[0]]
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal) + 1e-6
            
            sun_dir = getattr(self, 'sun_direction', np.array([0.5, 0.3, 0.8]))
            sun_dir /= np.linalg.norm(sun_dir)
            
            diffuse = max(0.2, np.dot(normal, sun_dir))
            face_color = tuple(min(255, int(c * (0.5 + diffuse * 0.5))) for c in color)
            
            cv2.fillPoly(frame, [pts_int], face_color, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [pts_int], True, tuple(min(255, c+30) for c in face_color), 1, lineType=cv2.LINE_AA)

    def _render_shadow(self, frame: np.ndarray, uav: dict,
                       camera_pos: np.ndarray, camera_orient: np.ndarray):
        """Zemin gölgesi"""
        uav_pos = np.array(uav['position'])
        size = uav.get('size', 2.0)
        heading = np.radians(uav.get('heading', 0.0))
        
        vertices, _ = self._get_uav_mesh(size)
        c, s = np.cos(heading), np.sin(heading)
        R_model = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        vertices_world = (R_model @ vertices.T).T + uav_pos
        
        sun_dir = getattr(self, 'sun_direction', np.array([0.5, 0.3, 0.8]))
        sun_dir /= np.linalg.norm(sun_dir)
        
        if sun_dir[2] >= 0: return 
        
        t = -vertices_world[:, 2] / sun_dir[2]
        valid = t > 0
        if not np.any(valid): return
        
        shadow_verts = vertices_world.copy()
        shadow_verts[:, 0] += t * sun_dir[0]
        shadow_verts[:, 1] += t * sun_dir[1]
        shadow_verts[:, 2] = 0 
        
        projections = self._project_points(shadow_verts, camera_pos, camera_orient)
        if np.isnan(projections).all(): return
        
        valid_proj = projections[~np.isnan(projections[:, 0])]
        if len(valid_proj) >= 3:
            hull_indices = cv2.convexHull(valid_proj.astype(np.int32), returnPoints=False)
            hull_pts = valid_proj[hull_indices.flatten()].astype(np.int32).reshape((-1, 1, 2))
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [hull_pts], (20, 20, 20))
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
    def _project_points(self, points_3d: np.ndarray, 
                       camera_pos: np.ndarray,
                       camera_orient: np.ndarray) -> np.ndarray:
        """
        Çoklu 3D noktayı ekran koordinatlarına dönüştür
        Args:
            points_3d: (N, 3) array
            camera_pos: (3,) array
            camera_orient: (3,) array
        Returns:
            (N, 2) array. Görünmeyen noktalar için NaN
        """
        rel_pos = points_3d - camera_pos
        # Calculate flare intensity
        # Use centralized rotation logic
        # yaw, pitch = camera_orient[2], camera_orient[1]
        # camera_forward = ...
        
        # Or simpler: transform [1,0,0] (forward) by R
        R = euler_to_rotation_matrix(*camera_orient)
        camera_forward = R @ np.array([1, 0, 0])
        cam_coords = (R @ rel_pos.T).T  # (N, 3)  World -> Camera dönüşümü
        
        # Kameranın arkasındakileri filtrele
        valid_mask = cam_coords[:, 0] > 0.1
        
        projections = np.full((len(points_3d), 2), np.nan)
        
        if np.any(valid_mask):
            valid_coords = cam_coords[valid_mask]
            
            # Perspektif projeksiyon
            # x = fx * y / x + cx (kamera koordinat sistemine dikkat: x=ileri, y=sağ, z=aşağı)
            x = self.K[0, 0] * valid_coords[:, 1] / valid_coords[:, 0] + self.K[0, 2]
            y = self.K[1, 1] * valid_coords[:, 2] / valid_coords[:, 0] + self.K[1, 2]
            
            projections[valid_mask, 0] = x
            projections[valid_mask, 1] = y
            
        return projections

    def _render_background(self, frame: np.ndarray, 
                          camera_pos: np.ndarray,
                          camera_orient: np.ndarray) -> np.ndarray:
        """
        Zemin, gökyüzü ve ufuk çizgisi (Prosedürel)
        """
        h, w = frame.shape[:2]
        pitch = camera_orient[1]
        
        # 1. Temel Gradyanlar (Hızlı Render)
        horizon_y = int(h / 2 + np.tan(pitch) * self.focal_length)
        
        # Gökyüzü (gerçekçi mavi - RGB: 135, 206, 235 Sky Blue)
        frame[:max(0, min(h, horizon_y+20)), :] = [235, 206, 135]  # BGR format - Açık Mavi
        # Zemin (yeşil-kahverengi arazi rengi - RGB: 86, 125, 70)
        frame[max(0, min(h, horizon_y-20)):, :] = [70, 125, 86]   # BGR format - Yeşil Arazi
        
        # Ufuk geçişi (pus)
        if 0 <= horizon_y < h:
            cv2.line(frame, (0, horizon_y), (w, horizon_y), (200, 200, 200), 2)
            
        # 2. Bulutlar (OpenSimplex varsa)
        if OPENSIMPLEX_AVAILABLE and getattr(self, 'haze_enabled', True):
            self._render_clouds(frame, horizon_y, camera_orient)
            
        # 3. Zemin Gridi (Hız ve İrtifa Algısı için)
        self._render_ground_grid(frame, camera_pos, camera_orient)
        
        # 4. Güneş
        self._render_sun(frame, camera_orient)
        
        return frame

    def _render_clouds(self, frame: np.ndarray, horizon_y: int, camera_orient: np.ndarray):
        """Prosedürel bulut katmanı"""
        h, w = frame.shape[:2]
        if horizon_y < 0: return # Sadece zemin görünüyorsa
        
        # Basit "cloud puff" çizimi (performans için tam noise map yerine)
        # Zaman ve bakış açısına göre hareket eden bulutlar
        t = self._noise_time * 2.0
        yaw = camera_orient[2]
        
        # Gökyüzünde rastgele bulut öbekleri - yerel RNG kullan (global state'i bozma)
        cloud_rng = np.random.default_rng(42)  # Sabit bulut yerleşimi
        num_clouds = 20
        
        for i in range(num_clouds):
            # Bulut pozisyonu (spherical coordinates approx)
            cloud_yaw = cloud_rng.uniform(0, 2*np.pi)
            cloud_pitch = cloud_rng.uniform(-np.pi/2, -0.1) # Sadece yukarıda
            
            # Kameraya göre açı farkı
            rel_yaw = cloud_yaw - yaw
            # -pi, pi arasına normalize et
            rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi
            
            if abs(rel_yaw) > np.radians(self.fov): continue # Görüş dışı
            
            # Ekran pozisyonu tahmin et
            cx = w/2 + (rel_yaw / np.radians(self.fov/2)) * (w/2)
            cy = h/2 + (np.tan(cloud_pitch - camera_orient[1])) * self.focal_length
            
            if cy > horizon_y: continue # Ufuk altında
            
            # Bulut çiz - int() ile Python native int'e çevir (OpenCV uyumluluğu)
            size = int(cloud_rng.integers(40, 120))
            color = int(cloud_rng.integers(240, 255))
            alpha = 0.4
            
            # Basit oval bulut
            if 0 < cx < w and 0 < cy < horizon_y:
                overlay = frame.copy()
                cv2.ellipse(overlay, (int(cx), int(cy)), (size, size//2), 0, 0, 360, (color, color, color), -1)
                cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

    def _render_ground_grid(self, frame: np.ndarray, camera_pos: np.ndarray, camera_orient: np.ndarray):
        """Sonsuz zemin ızgarası (Grid)"""
        grid_spacing = 50.0  # metre
        range_max = 2000.0   # çizim mesafesi
        
        # İHA'nın bulunduğu grid hücresi
        start_x = int(camera_pos[0] / grid_spacing) * grid_spacing
        start_y = int(camera_pos[1] / grid_spacing) * grid_spacing
        
        lines_x = [] # X eksenine paralel çizgiler (Y sabit)
        lines_y = [] # Y eksenine paralel çizgiler (X sabit)
        
        num_lines = int(range_max / grid_spacing)
        
        # Çizgi noktalarını oluştur
        points = []
        segments = [] # (start_idx, end_idx)
        
        # X yönündeki çizgiler
        for i in range(-num_lines, num_lines):
            y = start_y + i * grid_spacing
            points.append([start_x - range_max, y, 0])
            points.append([start_x + range_max, y, 0])
            segments.append((len(points)-2, len(points)-1))
            
        # Y yönündeki çizgiler
        for i in range(-num_lines, num_lines):
            x = start_x + i * grid_spacing
            points.append([x, start_y - range_max, 0])
            points.append([x, start_y + range_max, 0])
            segments.append((len(points)-2, len(points)-1))
            
        points = np.array(points)
        projections = self._project_points(points, camera_pos, camera_orient)
        
        h, w = frame.shape[:2]
        
        for p1_idx, p2_idx in segments:
            p1 = projections[p1_idx]
            p2 = projections[p2_idx]
            
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue
                
            # Ekran dışı kontrolü (basit)
            if not ((0 <= p1[0] < w and 0 <= p1[1] < h) or (0 <= p2[0] < w and 0 <= p2[1] < h)):
                # Clipping gerekebilir ama şimdilik sadece her ikisi de dışındaysa çizme
                # (Daha hassas Cohen-Sutherland gerekebilir ama OpenCV line bunu handle eder)
                pass

            # Rengi mesafeye göre soluklaştır (fog)
            # Basitçe sabit renk
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (100, 130, 140), 1)

    def _render_sun(self, frame: np.ndarray, camera_orient: np.ndarray):
        """Güneş diski"""
        sun_dir = getattr(self, 'sun_direction', np.array([0.5, 0.3, 0.8]))
        sun_dir = sun_dir / np.linalg.norm(sun_dir)
        
        # Güneşi sonsuzda bir nokta olarak düşün
        # Projeksiyon için sanal bir uzaklık kullan
        sun_pos_virtual = sun_dir * 1000.0
        
        # Kamera koordinatlarına çevir
        # Gökyüzü gradient'i için ufuk çizgisi hesapla
        # Pitch açısına göre kaydır
        R = euler_to_rotation_matrix(*camera_orient)
        forward = R @ np.array([1, 0, 0])
        pitch = np.arcsin(-forward[2]) # Approximate pitch from forward vector
        rel_pos = sun_pos_virtual # Kamera orijinde kabul edelim (skybox)
        cam_coords = R.T @ rel_pos
        
        if cam_coords[0] <= 0: return
        
        h, w = frame.shape[:2]
        x = self.K[0, 0] * cam_coords[1] / cam_coords[0] + self.K[0, 2]
        y = self.K[1, 1] * cam_coords[2] / cam_coords[0] + self.K[1, 2]
        
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (int(x), int(y)), 40, (255, 255, 240), -1)
            # Glow
            cv2.circle(frame, (int(x), int(y)), 80, (255, 255, 200), -1) 

        
    def _render_uav_target(self, frame: np.ndarray, uav: dict,
                           camera_pos: np.ndarray, camera_orient: np.ndarray):
        """Hedef İHA'yı gerçekçi şekilde çiz - heading'e göre döndürülmüş"""
        uav_pos = np.array(uav['position'])
        
        # Projeksiyon
        screen_pos = self.project_point(uav_pos, camera_pos, camera_orient)
        if screen_pos is None:
            return
            
        x, y = int(screen_pos[0]), int(screen_pos[1])
        distance = np.linalg.norm(uav_pos - camera_pos)
        
        # Görünür boyut
        uav_size = uav.get('size', 2.0)
        apparent_size = max(5, int(self.calculate_apparent_size(distance, uav_size)))
        
        # İHA heading açısı (varsa)
        uav_heading = uav.get('heading', 0.0)  # radyan veya derece
        if isinstance(uav_heading, (int, float)) and abs(uav_heading) < 10:
            # Muhtemelen radyan
            heading_rad = uav_heading
        else:
            # Derece olarak kabul et
            heading_rad = np.radians(uav_heading)
        
        # Kamera açısına göre göreli heading hesapla
        camera_yaw = camera_orient[2]
        relative_heading = heading_rad - camera_yaw
        
        # Ekrandaki rotasyon açısı (derece)
        screen_rotation = np.degrees(relative_heading)
        
        # Mesafeye göre renk (haze efekti)
        if self.haze_enabled:
            haze_factor = min(1.0, distance / self.haze_distance)
            base_color = (0, 0, 200) if uav.get('team') == 'red' else (200, 100, 0)
            sky_color = (200, 180, 160)  # Haze rengi
            color = tuple(int(base_color[i] * (1 - haze_factor * 0.6) + sky_color[i] * haze_factor * 0.6) 
                         for i in range(3))
        else:
            color = (0, 0, 255) if uav.get('team') == 'red' else (255, 100, 0)
        
        # Perspektif deformasyonu: heading'e göre aspect ratio değişimi
        # Kameraya dik bakış: tam siluet
        # Kameraya paralel bakış: dar siluet
        aspect_factor = abs(np.cos(relative_heading))  # 0 (paralel) - 1 (dik)
        aspect_factor = max(0.3, aspect_factor)  # Minimum %30
            
        # İHA silüeti (heading'e göre döndürülmüş elipsler)
        # Ana gövde
        body_w = int(apparent_size * aspect_factor)
        body_h = max(2, apparent_size // 3)
        cv2.ellipse(frame, (x, y), (body_w, body_h), screen_rotation, 0, 360, color, -1)
        
        # Kanatlar
        wing_span = int(apparent_size * 2 * aspect_factor)
        wing_height = max(2, apparent_size // 4)
        cv2.ellipse(frame, (x, y), (wing_span // 2, wing_height), screen_rotation, 0, 360, color, -1)
        
        # Kuyruk (heading yönünde)
        tail_x = int(x - np.sin(np.radians(screen_rotation)) * apparent_size * 0.6)
        tail_y = int(y + np.cos(np.radians(screen_rotation)) * apparent_size * 0.6)
        tail_size = max(3, apparent_size // 4)
        cv2.ellipse(frame, (tail_x, tail_y), (tail_size, tail_size // 2), 
                   screen_rotation, 0, 360, color, -1)
        
        # Kenar çizgisi
        outline_color = tuple(min(255, c + 50) for c in color)
        cv2.ellipse(frame, (x, y), (body_w, body_h), screen_rotation, 0, 360, outline_color, 1)
        
    def _apply_vignette(self, frame: np.ndarray) -> np.ndarray:
        """Lens vignette efekti (kenar karartma)"""
        h, w = frame.shape[:2]
        
        # Radyal gradyan maske oluştur
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        
        # Normalize mesafe
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist
        
        # Vignette faktörü (kenarlar daha karanlık)
        vignette = 1 - (dist_norm ** 2) * 0.4
        vignette = np.clip(vignette, 0.6, 1.0)
        
        # Uygula
        frame = (frame * vignette[:, :, np.newaxis]).astype(np.uint8)
        
        return frame
        
    def _apply_motion_blur(self, frame: np.ndarray, 
                           velocity: np.ndarray,
                           camera_orient: np.ndarray) -> np.ndarray:
        """Hıza bağlı yönsel motion blur uygula"""
        speed = np.linalg.norm(velocity)
        if speed < 5:  # Minimum blur hızı
            return frame

        # Kamera koordinatlarında hız vektörü (x=ileri, y=sağ, z=aşağı)
        R = euler_to_rotation_matrix(*camera_orient)
        cam_vel = R.T @ velocity
        image_vel = np.array([cam_vel[1], cam_vel[2]])
        image_speed = np.linalg.norm(image_vel)

        # Blur kernel boyutu (hız + exposure etkisi)
        blur_strength = getattr(self, 'blur_strength', 5)
        exposure_scale = max(0.5, min(2.0, self.exposure_time / 0.01))
        blur_size = min(int(speed / 10 * exposure_scale) + 1, blur_strength)
        blur_size = max(3, blur_size if blur_size % 2 == 1 else blur_size + 1)
        
        # Motion blur kernel
        if image_speed > 0.1:
            angle = np.arctan2(image_vel[1], image_vel[0])
        else:
            angle = 0.0
            
        # Round angle to increase cache hits (resolution ~0.6 degrees)
        angle_quantized = round(float(angle), 2)

        # Get cached kernel
        kernel = get_motion_blur_kernel(blur_size, angle_quantized)
        
        blurred = cv2.filter2D(frame, -1, kernel)
        
        # Orijinal ile karıştır
        alpha = min(0.5, speed / 100)
        result = cv2.addWeighted(frame, 1 - alpha, blurred, alpha, 0)
        
        return result
        
    def _apply_chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Kenarlarda renk sapması (chromatic aberration) uygula"""
        h, w = frame.shape[:2]
        
        # Kanal ayırma
        b, g, r = cv2.split(frame)
        
        # Radyal mesafe hesapla
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist / max_dist
        
        # Chromatic strength
        strength = getattr(self, 'chromatic_strength', 2.0)
        displacement = dist_norm * strength
        
        # Merkezden uzaklaşma yönü
        dx = (X - cx) / (dist + 1e-6)
        dy = (Y - cy) / (dist + 1e-6)
        
        # Kırmızı kanalı dışa kaydır
        map_x_r = (X + dx * displacement).astype(np.float32)
        map_y_r = (Y + dy * displacement).astype(np.float32)
        r_shifted = cv2.remap(r, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Mavi kanalı içe kaydır (ters yön)
        map_x_b = (X - dx * displacement * 0.5).astype(np.float32)
        map_y_b = (Y - dy * displacement * 0.5).astype(np.float32)
        b_shifted = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Birleştir
        result = cv2.merge([b_shifted, g, r_shifted])
        
        return result
        
    def _apply_lens_flare(self, frame: np.ndarray, 
                          camera_orient: np.ndarray) -> np.ndarray:
        """Güneşe bakınca lens flare efekti uygula"""
        h, w = frame.shape[:2]
        
        # Güneş yönü (normalize)
        sun_direction = self.sun_direction
        R = euler_to_rotation_matrix(*camera_orient)
        cam_forward = R @ np.array([1, 0, 0])
        sun_dot = np.dot(cam_forward, sun_direction)
        
        if sun_dot < 0.5:  # Güneş görünmüyor
            return frame
            
        flare_intensity = (sun_dot - 0.5) * 2 * self.sun_intensity  # 0-1 arası

        # Güneş ekran pozisyonu (kameraya göre projeksiyon)
        sun_cam = R.T @ (sun_direction * 1000.0)
        if sun_cam[0] <= 0:
            return frame
        sun_x = self.K[0, 0] * sun_cam[1] / sun_cam[0] + self.K[0, 2]
        sun_y = self.K[1, 1] * sun_cam[2] / sun_cam[0] + self.K[1, 2]
        sun_screen_x = int(sun_x)
        sun_screen_y = int(sun_y)
        
        flare_overlay = frame.copy()
        
        # Ana güneş parlak noktası
        glow_radius = int(max(10, 50 * flare_intensity))
        if 0 < sun_screen_x < w and 0 < sun_screen_y < h:
            cv2.circle(flare_overlay, (sun_screen_x, sun_screen_y), glow_radius, 
                      (200, 220, 255), -1)
            
            # Hayalet yansımalar
            for i in range(3):
                ghost_x = int(w / 2 + (sun_screen_x - w / 2) * (0.3 + i * 0.2))
                ghost_y = int(h / 2 + (sun_screen_y - h / 2) * (0.3 + i * 0.2))
                ghost_radius = int(20 * (1 - i * 0.3) * flare_intensity)
                ghost_color = (180 - i * 30, 200 - i * 20, 240 - i * 10)
                cv2.circle(flare_overlay, (ghost_x, ghost_y), ghost_radius, ghost_color, -1)
        
        alpha = 0.3 * flare_intensity
        result = cv2.addWeighted(frame, 1 - alpha, flare_overlay, alpha, 0)
        
        return result
        
    def _apply_depth_of_field(self, frame: np.ndarray, 
                              uav_states: List[Dict],
                              camera_pos: np.ndarray) -> np.ndarray:
        """
        Depth of Field efekti - odak dışı nesneler bulanık
        
        Focus mesafesine göre arka plan ve ön plan bulanıklaştırılır.
        """
        h, w = frame.shape[:2]
        
        # Odak mesafesinden uzaklığa göre bulanıklık haritası oluştur
        # Basitleştirilmiş yaklaşım: genel bir bulanıklık uygula
        
        # En yakın hedef mesafesini bul ve focus olarak kullan
        if uav_states:
            distances = [np.linalg.norm(np.array(uav['position']) - camera_pos) 
                        for uav in uav_states]
            nearest_distance = min(distances)
            focus_dist = getattr(self, 'focus_distance', nearest_distance)
        else:
            focus_dist = self.focus_distance
        
        # Basit yaklaşım: çerçevenin kenarlarına doğru bulanıklık
        # (Gerçek DoF için per-pixel depth map gerekir)
        strength = getattr(self, 'dof_strength', 1.0)
        
        # Radyal blur maske (merkez net, kenarlar bulanık)
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        blur_mask = (dist / max_dist) ** 2 * strength
        blur_mask = np.clip(blur_mask, 0, 1)
        
        # Bulanık versiyon
        blur_size = int(7 * strength)
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
        
        # Blend
        blur_mask_3ch = blur_mask[:, :, np.newaxis]
        result = (frame * (1 - blur_mask_3ch) + blurred * blur_mask_3ch).astype(np.uint8)
        
        return result
        
    def _apply_sensor_noise(self, frame: np.ndarray) -> np.ndarray:
        """
        Kamera sensör gürültüsü simülasyonu (Optimized with precomputed pool)
        
        Uses precomputed noise textures and cyclic indexing for performance.
        Temporal smoothing reduces flicker.
        """
        # Check if noise pool needs reinitialization (resolution change)
        if self._noise_pool is None or self._noise_pool.shape[1:3] != frame.shape[:2]:
            self._init_noise_pool()
        
        # Get noise from precomputed pool (cyclic)
        current_noise = self._noise_pool[self._noise_pool_index]
        self._noise_pool_index = (self._noise_pool_index + 1) % self._noise_pool_size
        
        # Temporal smoothing (20% old, 80% new to reduce flicker)
        if hasattr(self, '_prev_noise') and self._prev_noise is not None and self._prev_noise.shape == frame.shape:
            self._prev_noise = 0.2 * self._prev_noise + 0.8 * current_noise
        else:
            self._prev_noise = current_noise.copy()
            
        # Apply noise
        noisy = frame.astype(np.float32) + self._prev_noise
        result = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return result
        
    def _apply_auto_exposure(self, frame: np.ndarray) -> np.ndarray:
        """
        Auto Exposure simülasyonu
        
        Sahne parlaklığına göre exposure ayarlar.
        Smooth geçiş için eksponansiyel yumuşatma kullanır.
        """
        # Mevcut ortalama parlaklık
        current_brightness = np.mean(frame)
        
        if current_brightness < 1:
            return frame
            
        # Hedef parlaklığa göre gerekli faktör
        target = self.target_brightness
        ideal_factor = target / current_brightness
        
        # Smooth geçiş (ani değişiklikleri önle)
        smooth_rate = 0.1  # Yavaş adaptasyon
        self._current_exposure = self._current_exposure * (1 - smooth_rate) + ideal_factor * smooth_rate
        
        # Sınırla
        self._current_exposure = np.clip(self._current_exposure, 0.5, 2.5)
        
        # Uygula
        result = frame.astype(np.float32) * self._current_exposure
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

    def _apply_tonemapping(self, frame: np.ndarray) -> np.ndarray:
        """
        HDR -> LDR Tonemapping (Reinhard operatörü)
        
        Yüksek dinamik aralıklı görüntüleri ekran için sıkıştırır.
        Parlak alanlardaki detayları korur.
        """
        # Normalize (0-1)
        img_float = frame.astype(np.float32) / 255.0
        
        # Reinhard Tonemapping: L / (1 + L)
        # Gamma correction ile birlikte (gamma=2.2)
        gamma = 2.2
        mapped = img_float / (1 + img_float)
        mapped = np.power(mapped, 1.0 / gamma)
        
        # Scale back to 0-255
        result = np.clip(mapped * 255, 0, 255).astype(np.uint8)
        return result

    def _apply_color_grading(self, frame: np.ndarray) -> np.ndarray:
        """
        Renk düzenleme (Color Grading)
        
        Görüntüye belirli bir renk tonu verir. BGR kanallarını ölçekler.
        """
        b, g, r = self.color_grading_values
        
        # NumPy broadcasting ile çarpma
        # Frame shape (H, W, 3), values (3) -> matches last dim
        graded = frame.astype(np.float32) * np.array([b, g, r])
        
        return np.clip(graded, 0, 255).astype(np.uint8)

    def _apply_lens_softness(self, frame: np.ndarray) -> np.ndarray:
        """
        Lens yumuşaklığı (Soft focus / Blur)
        
        Keskin CG görüntülerini yumuşatarak daha gerçekçi yapar.
        """
        if self.softness_strength <= 0:
            return frame
            
        # Hafif Gaussian Blur
        # Sigma değeri strength ile orantılı
        sigma = self.softness_strength * 0.8
        
        # Kernel size (tek sayı olmalı)
        ksize = int(sigma * 3)
        if ksize % 2 == 0: ksize += 1
        ksize = max(3, ksize)
        
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        
        return blurred

    def _apply_rain_snow(self, frame: np.ndarray, 
                         effect_type: str = 'rain', 
                         density: float = 0.5) -> np.ndarray:
        """
        Yağmur veya Kar Efekti 
        
        Args:
            effect_type: 'rain' veya 'snow'
            density: Yoğunluk (0.0 - 1.0)
        """
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)
        
        if effect_type == 'rain':
            # Yağmur çizgileri
            num_drops = int(500 * density)
            
            # Rastgele başlangıç noktaları
            for _ in range(num_drops):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                
                # Eğimli çizgiler (rüzgar etkisi)
                length = np.random.randint(10, 30)
                angle = np.radians(75) # Hafif eğik
                
                x2 = int(x + length * np.cos(angle))
                y2 = int(y + length * np.sin(angle))
                
                color = (200, 200, 200) # Gri-beyaz
                cv2.line(overlay, (x, y), (x2, y2), color, 1)
                
            # Blur (hareket hissi)
            overlay = cv2.blur(overlay, (3, 3))
            
            # Blend - Screen mode benzeri (sadece parlaklık ekle)
            result = cv2.add(frame, overlay)
            
        elif effect_type == 'snow':
            # Kar taneleri
            num_flakes = int(200 * density)
            
            for _ in range(num_flakes):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                r = np.random.randint(1, 4)
                
                color = (255, 255, 255)
                cv2.circle(overlay, (x, y), r, color, -1)
                
            # Hafif blur
            overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
            
            # Blend
            result = cv2.addWeighted(frame, 1.0, overlay, 0.8, 0)
            
        else:
            return frame
            
        return result
        
    def get_config(self) -> dict:
        """Mevcut konfigürasyonu döndür"""
        return {
            'fov': self.fov,
            'resolution': self.resolution,
            'fps': self.fps,
            'mount_offset': self.mount_offset.tolist(),
            'mount_pitch': np.degrees(self.mount_pitch),
            'distortion_enabled': self.distortion_enabled,
            'distortion_clamp': self.distortion_clamp,
            'k1': self.k1,
            'k2': self.k2,
            'k3': self.k3,
            'p1': self.p1,
            'p2': self.p2,
            'shake_enabled': self.shake_enabled,
            'shake_intensity': self.shake_intensity,
            'motion_blur_enabled': self.motion_blur_enabled,
            'chromatic_aberration_enabled': self.chromatic_aberration_enabled,
            'lens_flare_enabled': self.lens_flare_enabled,
            'sun_direction': self.sun_direction.tolist(),
            'sun_intensity': self.sun_intensity,
            'sun_angular_radius': self.sun_angular_radius,
            'vignette_enabled': self.vignette_enabled,
            'haze_enabled': self.haze_enabled,
            'haze_distance': self.haze_distance,
            'dof_enabled': self.dof_enabled,
            'focus_distance': self.focus_distance,
            'dof_strength': self.dof_strength,
            'sensor_noise_enabled': self.sensor_noise_enabled,
            'iso': self.iso,
            'color_grading_enabled': self.color_grading_enabled,
            'softness_enabled': self.softness_enabled,
            'auto_exposure_enabled': self.auto_exposure_enabled,
            'target_brightness': self.target_brightness
        }


# Geriye uyumluluk için alias
SimulatedCamera = FixedCamera
