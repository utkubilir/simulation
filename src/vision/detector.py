"""
YOLOv8 Tabanlı İHA Tespit Sistemi

Harici model yükleme desteği ile modüler tasarım.

FIXED: SimulationDetector now uses seeded RNG for determinism and reliable detections.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
import time
import logging

from src.simulation.utils import project_point_simple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    Ensures consistent interface for both real YOLO and simulation detectors.
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.classes = ['uav']
        self.input_size = (640, 640)
        
        # Performance metrics
        self.last_inference_time = 0.0
        self.total_detections = 0
        
        if config_path:
            self._load_config(config_path)
            
    def _load_config(self, config_path: str):
        """Detector konfigürasyonunu yükle"""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"⚠️  Konfigürasyon bulunamadı: {config_path}")
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Common settings
        if 'detection' in config:
            det_config = config['detection']
            self.confidence_threshold = det_config.get('confidence', 0.5)
            self.nms_threshold = det_config.get('nms_threshold', 0.4)
            self.classes = det_config.get('classes', ['uav'])
            
            if 'input_size' in det_config:
                self.input_size = tuple(det_config['input_size'])
                
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in a frame."""
        pass
        
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """Batch detection."""
        return [self.detect(frame) for frame in frames]
        
    def set_confidence(self, threshold: float):
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        
    def set_nms_threshold(self, threshold: float):
        self.nms_threshold = max(0.0, min(1.0, threshold))
        
    def get_stats(self) -> Dict:
        return {
            'confidence_threshold': self.confidence_threshold,
            'last_inference_time_ms': self.last_inference_time * 1000,
            'total_detections': self.total_detections,
            'fps': 1.0 / self.last_inference_time if self.last_inference_time > 0 else 0
        }

    def draw_detections(self, frame: np.ndarray, detections: List[Dict],
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
        """Draw detections on frame."""
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['confidence']
            cls = det.get('class', 'uav')
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{cls}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(output, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 10, y1),
                         color, -1)
                         
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                       
            if 'center' in det:
                cx, cy = int(det['center'][0]), int(det['center'][1])
                cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
            
        return output


class UAVDetector(BaseDetector):
    """
    YOLOv8 based detector implementation.
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        self.model = None
        self.model_path = model_path
        super().__init__(config_path)
        
        # Load model explicitly here, not in base
        if model_path or config_path is None:
            self.load_model(model_path)
            
    def _load_config(self, config_path: str):
        super()._load_config(config_path)
        # Load model path from config if not provided
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if 'model' in config and config['model'].get('path'):
                self.model_path = config['model']['path']

    def load_model(self, model_path: str = None):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("⚠️  ultralytics paketi bulunamadı.")
            self.model = None
            return
            
        if model_path is None:
            self.model = YOLO('yolov8n.pt')
            self.model_path = 'yolov8n.pt'
            logger.info("✓ Varsayılan model yüklendi: yolov8n.pt")
        else:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model bulunamadı: {model_path}")
            self.model = YOLO(str(path))
            self.model_path = str(path)
            logger.info(f"✓ Harici model yüklendi: {model_path}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect using YOLO model."""
        if frame is None:
            return []
            
        start_time = time.time()
        
        if self.model is None:
            return []
            
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False
        )
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
                
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                    
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, 'unknown')
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': cls_name,
                    'class_id': cls_id,
                    'center': (cx, cy),
                    'width': w,
                    'height': h
                })
                
        self.last_inference_time = time.time() - start_time
        self.total_detections += len(detections)
        
        return detections
    
    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats['model_path'] = self.model_path
        return stats


class SimulationDetector(BaseDetector):
    """
    Simülasyon ortamı için özel detector
    
    Gerçek YOLO yerine simülasyondaki İHA pozisyonlarını kullanır.
    
    FIXED: 
    - Uses seeded RNG passed in for determinism
    - Always detects visible targets (no random probability check in headless)
    - Minimal noise for reliable lock testing
    """
    
    def __init__(self, config_path: str = None, rng: np.random.Generator = None):
        super().__init__(config_path=config_path)
        
        # Use passed-in RNG for determinism
        self.rng = rng if rng is not None else np.random.default_rng(42)
        
        # Simülasyon parametreleri
        self.detection_range = 500.0  # metre
        self.fov = 60.0  # derece
        self.noise_std = 2.0  # piksel gürültüsü (reduced for reliability)
        self.detection_probability = 1.0  # ALWAYS detect if visible
        self.false_positive_rate = 0.0   # No false positives for determinism
        
        # Gelişmiş tespit parametreleri
        self.min_apparent_size = 8.0  # Minimum görünür boyut (piksel)
        self.edge_confidence_penalty = 0.25  # Kenar güven düşüşü (max)
        
        # Simülasyondan alınan veriler
        self.world_uavs = []  # [(id, position, size), ...]
        self.camera_matrix = None
        self.camera_position = None
        self.camera_orientation = None
        
    def set_rng(self, rng: np.random.Generator):
        """Set RNG for deterministic noise."""
        self.rng = rng
        
    def set_world_state(self, uavs: List[Dict], camera_pos: np.ndarray,
                        camera_orient: np.ndarray, camera_matrix: np.ndarray = None):
        """
        Simülasyon dünya durumunu ayarla
        
        Args:
            uavs: İHA listesi [{'id', 'position', 'team'}, ...]
            camera_pos: Kamera pozisyonu
            camera_orient: Kamera oryantasyonu
            camera_matrix: Kamera içsel matrisi
        """
        self.world_uavs = uavs
        self.camera_position = camera_pos
        self.camera_orientation = camera_orient
        self.camera_matrix = camera_matrix
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Override detect to FORCE simulation logic regardless of model status."""
        # Always use simulation logic for SimulationDetector
        # This ensures determinism and ignores any loaded YOLO models
        return self._simulate_detections(frame)

    def _simulate_detections(self, frame: np.ndarray) -> List[Dict]:
        """Simülasyon bazlı tespitler üret - DETERMINISTIC"""
        if self.camera_position is None or not self.world_uavs:
            return []
            
        detections = []
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate focal length once
        f = frame_w / (2 * np.tan(np.radians(self.fov / 2)))
        
        for uav in self.world_uavs:
            uav_pos = np.array(uav['position'])
            
            # Mesafe kontrolü
            distance = np.linalg.norm(uav_pos - self.camera_position)
            if distance > self.detection_range:
                continue
                
            # Görüş alanı kontrolü (basitleştirilmiş)
            to_uav = uav_pos - self.camera_position
            to_uav_norm = to_uav / (np.linalg.norm(to_uav) + 1e-6)
            
            # Projeksiyon (utils kullanarak)
            screen_pos = project_point_simple(
                uav_pos, self.camera_position, self.camera_orientation,
                frame_w, frame_h, self.fov
            )
            
            if screen_pos is None:
                continue
                
            x, y = screen_pos
            
            # Ekran sınırları kontrolü
            if not (0 <= x < frame_w and 0 <= y < frame_h):
                continue
                
            # İHA boyutu (mesafeye bağlı)
            uav_size = uav.get('size', 2.0)  # metre
            apparent_size = f * uav_size / distance
            
            # Gelişmiş gürültü modeli: boyuta bağlı gürültü
            # Küçük hedefler = daha fazla gürültü (tespit daha zor)
            size_noise_factor = np.clip(50.0 / max(apparent_size, 10), 0.5, 2.0)
            x += self.rng.normal(0, self.noise_std * size_noise_factor)
            y += self.rng.normal(0, self.noise_std * size_noise_factor)
            apparent_size += self.rng.normal(0, apparent_size * 0.05)
            apparent_size = max(self.min_apparent_size, apparent_size)
            
            # Bounding box - ensure minimum size for lock criteria
            half_size = max(apparent_size / 2, 25)  # Min 50px box
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(frame_w, x + half_size)
            y2 = min(frame_h, y + half_size)
            
            # Gelişmiş güven skoru hesaplama
            confidence = self._calculate_realistic_confidence(
                distance, (x, y), apparent_size, frame_w, frame_h
            )
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class': 'uav',
                'class_id': 0,
                'center': (x, y),
                'width': x2 - x1,
                'height': y2 - y1,
                'distance': distance,
                'world_id': uav.get('id'),
                'team': uav.get('team'),
                'world_pos': uav_pos, # Ground truth position for autopilot
                'velocity': uav.get('velocity', np.zeros(3)) # Ground truth velocity
            })
            
        return detections
    
    def _calculate_realistic_confidence(self, distance: float, screen_pos: tuple,
                                        apparent_size: float, frame_w: int, frame_h: int) -> float:
        """
        Gerçekçi güven skoru hesapla
        
        Faktörler:
        1. Mesafe (hibrit decay - atmosferik etkiler)
        2. Boyut (küçük hedefler = düşük güven, ama makul aralıkta)
        3. Merkez uzaklığı (kenarlar = düşük güven, lens aberasyonu simülasyonu)
        """
        # 1. Mesafe faktörü (hibrit: temel güven + exponential decay)
        # 100m'de ~0.85, 300m'de ~0.65, 500m'de ~0.55
        dist_ratio = distance / self.detection_range
        dist_factor = 0.5 + 0.5 * np.exp(-dist_ratio * 1.5)
        
        # 2. Boyut faktörü (küçük hedefler tespit edilmesi daha zor)
        # 10px: 0.5, 25px: 0.75, 50px+: 1.0
        size_factor = np.clip(0.4 + apparent_size / 30.0, 0.5, 1.0)
        
        # 3. Merkez uzaklığı faktörü (kenarlar daha zor - vignette benzeri)
        cx, cy = frame_w / 2, frame_h / 2
        dx = abs(screen_pos[0] - cx) / cx if cx > 0 else 0
        dy = abs(screen_pos[1] - cy) / cy if cy > 0 else 0
        edge_distance = np.sqrt(dx**2 + dy**2)  # 0 = merkez, ~1.41 = köşe
        edge_factor = 1.0 - self.edge_confidence_penalty * min(edge_distance, 1.0)
        
        # Temel güven skoru
        base_confidence = dist_factor * size_factor * edge_factor
        
        # Deterministik gürültü ekle
        noise = self.rng.normal(0, 0.02)
        
        # Final skor
        confidence = np.clip(base_confidence + noise, 0.4, 0.99)
        
        return confidence
        

