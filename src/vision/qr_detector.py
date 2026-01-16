"""
QR Kod Tespit Modülü

Kamikaze görevi için yer hedefi QR kod okuyucu.
Simülasyonda sentetik QR tespiti kullanır.

Şartname 6.2 gereksinimleri:
- QR kod boyutu: 2m x 2m
- 4 tarafı 45° açılı, 3m yüksekliğinde plakalarla kapatılı
- Sadece dik açıyla bakıldığında görünür
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class QRDetection:
    """QR tespit sonucu"""
    detected: bool
    content: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    center: Optional[Tuple[int, int]] = None
    size_ratio: float = 0.0  # Ekran alanına oranı
    distance: float = 0.0    # Hedefe mesafe
    angle: float = 0.0       # Bakış açısı (derece)


class SyntheticQRDetector:
    """
    Sentetik QR Kod Dedektörü
    
    Simülasyonda gerçek QR okuma yerine geometrik hesaplama kullanır.
    Kamera açısı ve mesafeye göre QR kodun görünürlüğünü hesaplar.
    
    Şartname gereksinimleri:
    - QR kod 45° açılı plakalarla çevrili
    - Sadece dik açıdan (>45° pitch) görülebilir
    - Minimum boyut kontrolü
    """
    
    def __init__(self, camera_config: dict = None):
        camera_config = camera_config or {}
        
        self.fov_h = np.radians(camera_config.get('fov', 60))
        self.fov_v = self.fov_h * 0.75  # 4:3 aspect ratio
        self.width = camera_config.get('width', 640)
        self.height = camera_config.get('height', 480)
        
        # Minimum tespit parametreleri
        self.min_size_pixels = 30      # Minimum piksel boyutu
        self.min_pitch_angle = 45.0    # Minimum kamera pitch açısı (derece)
        self.max_detection_distance = 150.0  # Maksimum tespit mesafesi
        
    def detect(self, camera_state: dict, ground_target: dict) -> QRDetection:
        """
        QR kod tespiti
        
        Args:
            camera_state: Kamera durumu {position, orientation, forward_vector}
            ground_target: Yer hedefi bilgisi {position, size, qr_content, wall_height}
            
        Returns:
            QRDetection sonucu
        """
        cam_pos = np.array(camera_state['position'])
        cam_orient = camera_state.get('orientation', [0, 0, 0])
        target_pos = np.array(ground_target['position'])
        target_size = ground_target.get('size', 2.0)
        wall_height = ground_target.get('wall_height', 3.0)
        
        # Forward vector hesapla veya al
        if 'forward_vector' in camera_state:
            cam_forward = np.array(camera_state['forward_vector'])
        else:
            # Orientation'dan hesapla (roll, pitch, yaw)
            pitch = cam_orient[1] if len(cam_orient) > 1 else 0
            yaw = cam_orient[2] if len(cam_orient) > 2 else 0
            
            cam_forward = np.array([
                np.cos(yaw) * np.cos(pitch),
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch)
            ])
        
        # Kameradan hedefe vektör
        to_target = target_pos - cam_pos
        distance = np.linalg.norm(to_target)
        
        # Çok uzak veya çok yakın
        if distance < 1 or distance > self.max_detection_distance:
            return QRDetection(False, distance=distance)
            
        to_target_normalized = to_target / distance
        
        # Kamera açısı kontrolü (hedef görüş alanında mı?)
        dot_product = np.dot(cam_forward, to_target_normalized)
        angle_to_target = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
        
        # FOV kontrolü
        fov_degrees = np.degrees(self.fov_h / 2)
        if angle_to_target > fov_degrees:
            return QRDetection(False, distance=distance, angle=angle_to_target)
            
        # Dalış açısı kontrolü
        # QR kod sadece dik bakışta okunabilir (plaka açısı 45°)
        # Kamera aşağı bakmalı (negatif pitch)
        vertical_component = -to_target_normalized[2]  # Aşağı yön pozitif
        pitch_angle = np.degrees(np.arcsin(np.clip(vertical_component, -1, 1)))
        
        # Plaka kontrolü
        # Şartname: 45° açılı plakalar, 3m yükseklik
        # Kamera en az 45° aşağı bakmalı
        if pitch_angle < self.min_pitch_angle:
            return QRDetection(False, distance=distance, angle=pitch_angle)
            
        # Yatay mesafe kontrolü
        horizontal_dist = np.sqrt(to_target[0]**2 + to_target[1]**2)
        
        # Plaka gölgesi hesabı
        # Eğer yatay mesafe çok fazlaysa QR görünmez
        max_horizontal_dist_at_altitude = cam_pos[2] / np.tan(np.radians(45))
        if horizontal_dist > max_horizontal_dist_at_altitude:
            return QRDetection(False, distance=distance, angle=pitch_angle)
        
        # Ekrandaki boyut hesabı
        apparent_size_rad = target_size / distance
        pixel_size = int(apparent_size_rad / self.fov_h * self.width)
        
        # Minimum boyut kontrolü
        if pixel_size < self.min_size_pixels:
            return QRDetection(False, distance=distance, angle=pitch_angle,
                             size_ratio=pixel_size / self.width)
        
        # Ekrandaki konum
        # Basit projeksiyon
        # Right vector
        up_world = np.array([0, 0, 1])
        right = np.cross(cam_forward, up_world)
        if np.linalg.norm(right) < 0.01:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, cam_forward)
        
        local_x = np.dot(to_target, right)
        local_y = np.dot(to_target, up)
        
        # Normalize ve ekran koordinatlarına dönüştür
        screen_x = int(self.width / 2 + (local_x / distance) / np.tan(self.fov_h/2) * self.width / 2)
        screen_y = int(self.height / 2 - (local_y / distance) / np.tan(self.fov_v/2) * self.height / 2)
        
        # Ekran sınırları kontrolü
        if not (0 <= screen_x < self.width and 0 <= screen_y < self.height):
            return QRDetection(False, distance=distance, angle=pitch_angle)
        
        # Boyut oranı
        size_ratio = (pixel_size * pixel_size) / (self.width * self.height)
        
        # Bbox hesapla
        half_size = pixel_size // 2
        bbox = (
            max(0, screen_x - half_size),
            max(0, screen_y - half_size),
            min(pixel_size, self.width - screen_x + half_size),
            min(pixel_size, self.height - screen_y + half_size)
        )
        center = (screen_x, screen_y)
        
        # QR içeriği
        qr_content = ground_target.get('qr_content', f'TEKNOFEST2026_{np.random.randint(1000, 9999)}')
        
        return QRDetection(
            detected=True,
            content=qr_content,
            bbox=bbox,
            center=center,
            size_ratio=size_ratio,
            distance=distance,
            angle=pitch_angle
        )
    
    def detect_from_uav(self, uav_state: dict, ground_target: dict) -> QRDetection:
        """
        İHA durumundan QR tespiti
        
        Args:
            uav_state: İHA durumu (to_dict() formatı)
            ground_target: Yer hedefi
            
        Returns:
            QRDetection
        """
        camera_state = {
            'position': uav_state['position'],
            'orientation': uav_state['orientation'],
            'forward_vector': uav_state.get('forward_vector')
        }
        
        return self.detect(camera_state, ground_target)


class QRTargetManager:
    """
    Yer hedefi yöneticisi
    
    Simülasyonda yer hedeflerini yönetir ve QR kod tespitini koordine eder.
    """
    
    def __init__(self, camera_config: dict = None):
        self.detector = SyntheticQRDetector(camera_config)
        self.targets: Dict[str, dict] = {}
        
    def add_target(self, target_id: str, position: list, qr_content: str = None,
                   size: float = 2.0, wall_height: float = 3.0):
        """Yer hedefi ekle"""
        if qr_content is None:
            qr_content = f"TEKNOFEST2026_{target_id}"
            
        self.targets[target_id] = {
            'id': target_id,
            'position': np.array(position),
            'qr_content': qr_content,
            'size': size,
            'wall_height': wall_height
        }
        
    def remove_target(self, target_id: str):
        """Yer hedefi kaldır"""
        if target_id in self.targets:
            del self.targets[target_id]
            
    def detect_all(self, uav_state: dict) -> Dict[str, QRDetection]:
        """Tüm hedefler için QR tespiti yap"""
        results = {}
        for target_id, target in self.targets.items():
            detection = self.detector.detect_from_uav(uav_state, target)
            results[target_id] = detection
        return results
    
    def detect_nearest(self, uav_state: dict) -> Optional[Tuple[str, QRDetection]]:
        """En yakın tespit edilen QR'ı döndür"""
        all_detections = self.detect_all(uav_state)
        
        detected = [(tid, d) for tid, d in all_detections.items() if d.detected]
        
        if not detected:
            return None
            
        # En yakın olanı seç
        detected.sort(key=lambda x: x[1].distance)
        return detected[0]
