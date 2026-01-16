"""
Görüntü Tabanlı İHA Dedektörü

Gerçek görüntü işleme teknikleri ile İHA tespiti.
Yarışmada kullanılacak yöntemlerin simülasyonu.

Teknikler:
1. Renk tabanlı segmentasyon (HSV)
2. Kontur analizi
3. Hareket tespiti (optical flow)
4. Morfolojik işlemler
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class Detection:
    """Tespit sonucu"""
    id: int                              # Tespit ID (frame içinde unique)
    bbox: Tuple[int, int, int, int]      # (x, y, w, h)
    center: Tuple[float, float]          # Merkez koordinatları
    confidence: float                     # Güven skoru (0-1)
    area: int                            # Piksel alanı
    contour: np.ndarray = None           # Orijinal kontur
    color_class: str = 'unknown'         # Renk sınıfı
    
    @property
    def x(self) -> int:
        return self.bbox[0]
        
    @property
    def y(self) -> int:
        return self.bbox[1]
        
    @property
    def w(self) -> int:
        return self.bbox[2]
        
    @property
    def h(self) -> int:
        return self.bbox[3]


@dataclass
class ImageDetectorConfig:
    """Dedektör konfigürasyonu"""
    # Renk filtreleme (HSV)
    target_colors: Dict[str, Dict] = None  # Hedef renkleri
    
    # Kontur filtreleme
    min_area: int = 50                     # Minimum alan (piksel²)
    max_area: int = 50000                  # Maksimum alan
    min_aspect_ratio: float = 0.2          # Minimum en boy oranı
    max_aspect_ratio: float = 5.0          # Maksimum en boy oranı
    
    # Morfolojik işlemler
    morph_kernel_size: int = 5
    
    # Hareket tespiti
    motion_threshold: float = 25.0
    history_length: int = 5
    
    # Güven eşikleri
    min_confidence: float = 0.3
    
    def __post_init__(self):
        if self.target_colors is None:
            # Varsayılan hedef renkleri (HSV aralıkları)
            self.target_colors = {
                'red': {
                    'lower1': np.array([0, 100, 100]),
                    'upper1': np.array([10, 255, 255]),
                    'lower2': np.array([160, 100, 100]),
                    'upper2': np.array([180, 255, 255])
                },
                'blue': {
                    'lower': np.array([100, 100, 100]),
                    'upper': np.array([130, 255, 255])
                },
                'dark': {
                    'lower': np.array([0, 0, 0]),
                    'upper': np.array([180, 255, 60])
                }
            }


class ImageBasedDetector:
    """
    Görüntü tabanlı İHA dedektörü
    
    Gerçek görüntü işleme teknikleri kullanarak İHA tespit eder.
    Yarışmada kullanılacak yöntemlerin simülasyonu.
    
    Pipeline:
    1. Preprocessing (blur, denoise)
    2. Renk segmentasyonu
    3. Morfolojik işlemler
    4. Kontur bulma
    5. Filtreleme ve skorlama
    6. Hareket tespiti ile doğrulama
    
    Usage:
        detector = ImageBasedDetector(ImageDetectorConfig())
        detections = detector.detect(frame)
    """
    
    def __init__(self, config: ImageDetectorConfig = None):
        self.config = config or ImageDetectorConfig()
        
        # Frame history (hareket tespiti için)
        self._frame_history: deque = deque(maxlen=self.config.history_length)
        self._prev_gray = None
        
        # Background subtractor
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=50,
            detectShadows=False
        )
        
        # Morfolojik kernel
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        
        # Debug
        self._last_masks = {}
        self._detection_count = 0
        
    def detect(self, frame: np.ndarray, 
               use_motion: bool = True) -> List[Detection]:
        """
        Frame'den İHA tespit et
        
        Args:
            frame: RGB görüntü
            use_motion: Hareket tespiti kullan
            
        Returns:
            Detection listesi
        """
        if frame is None or frame.size == 0:
            return []
            
        detections = []
        h, w = frame.shape[:2]
        
        # 1. Preprocessing
        processed = self._preprocess(frame)
        
        # 2. Renk tabanlı segmentasyon
        color_detections = self._color_based_detection(processed)
        detections.extend(color_detections)
        
        # 3. Hareket tabanlı tespit (opsiyonel)
        if use_motion:
            motion_detections = self._motion_based_detection(frame)
            detections.extend(motion_detections)
            
        # 4. Gökyüzü/zemin ayrımı ile doğrulama
        detections = self._validate_with_context(frame, detections)
        
        # 5. Duplicate temizleme
        detections = self._remove_duplicates(detections)
        
        # 6. ID ata
        for i, det in enumerate(detections):
            det.id = i
            
        return detections
        
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Görüntü ön işleme"""
        # Gaussian blur (gürültü azaltma)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Bilateral filter (kenarları koruyarak yumuşatma)
        filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
        
        return filtered
        
    def _color_based_detection(self, frame: np.ndarray) -> List[Detection]:
        """
        Renk tabanlı segmentasyon ile tespit
        
        Hedef İHA renklerini (kırmızı, mavi, koyu) arar.
        """
        detections = []
        
        # HSV'ye dönüştür
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        for color_name, color_range in self.config.target_colors.items():
            # Renk maskesi oluştur
            if 'lower1' in color_range:  # İki aralık (kırmızı için)
                mask1 = cv2.inRange(hsv, color_range['lower1'], color_range['upper1'])
                mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                
            # Morfolojik işlemler
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
            
            # Debug için sakla
            self._last_masks[color_name] = mask
            
            # Konturları bul
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Konturları filtrele ve detection oluştur
            for contour in contours:
                detection = self._contour_to_detection(contour, color_name)
                if detection is not None:
                    detections.append(detection)
                    
        return detections
        
    def _motion_based_detection(self, frame: np.ndarray) -> List[Detection]:
        """
        Hareket tabanlı tespit
        
        Background subtraction ve optical flow kullanır.
        """
        detections = []
        
        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Background subtraction
        fg_mask = self._bg_subtractor.apply(gray)
        
        # Morfolojik temizleme
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._morph_kernel)
        
        # Threshold
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        self._last_masks['motion'] = fg_mask
        
        # Konturları bul
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Optical flow ile hareket yönü doğrulama
        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (self.config.min_area < area < self.config.max_area):
                    continue
                    
                # Kontur merkezi
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Hareket vektörü
                    if 0 <= cy < flow.shape[0] and 0 <= cx < flow.shape[1]:
                        vx, vy = flow[cy, cx]
                        motion_mag = np.sqrt(vx**2 + vy**2)
                        
                        # Hareket eşiği
                        if motion_mag > self.config.motion_threshold:
                            detection = self._contour_to_detection(contour, 'motion')
                            if detection is not None:
                                detection.confidence *= 0.7  # Motion düşük güven
                                detections.append(detection)
                                
        self._prev_gray = gray.copy()
        
        return detections
        
    def _contour_to_detection(self, contour: np.ndarray, 
                               color_class: str) -> Optional[Detection]:
        """Konturu Detection nesnesine dönüştür"""
        area = cv2.contourArea(contour)
        
        # Alan filtresi
        if not (self.config.min_area < area < self.config.max_area):
            return None
            
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio filtresi
        aspect = w / max(h, 1)
        if not (self.config.min_aspect_ratio < aspect < self.config.max_aspect_ratio):
            return None
            
        # Merkez
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = x + w/2, y + h/2
            
        # Güven skoru hesapla
        confidence = self._calculate_confidence(contour, area, aspect)
        
        if confidence < self.config.min_confidence:
            return None
            
        return Detection(
            id=-1,  # Sonra atanacak
            bbox=(x, y, w, h),
            center=(cx, cy),
            confidence=confidence,
            area=area,
            contour=contour,
            color_class=color_class
        )
        
    def _calculate_confidence(self, contour: np.ndarray, 
                               area: int, aspect: float) -> float:
        """
        Tespit güven skoru hesapla
        
        Faktörler:
        - Kontur düzgünlüğü
        - Alan/convex hull oranı
        - Aspect ratio İHA'ya uygunluğu
        """
        confidence = 0.5  # Başlangıç
        
        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            confidence += solidity * 0.2
            
        # Aspect ratio (İHA genelde 1.5-4 arasında)
        if 1.0 < aspect < 4.0:
            confidence += 0.2
        elif 0.5 < aspect < 5.0:
            confidence += 0.1
            
        # Alan puanı (çok küçük veya büyük düşük güven)
        optimal_area = 500  # Orta mesafe boyutu
        area_score = 1 - abs(np.log10(area/optimal_area)) * 0.2
        confidence += max(0, area_score * 0.1)
        
        return min(1.0, max(0.0, confidence))
        
    def _validate_with_context(self, frame: np.ndarray, 
                                detections: List[Detection]) -> List[Detection]:
        """
        Bağlamsal doğrulama
        
        - Ufuk çizgisi altındakileri filtrele
        - Gökyüzündekilere öncelik ver
        """
        h, w = frame.shape[:2]
        validated = []
        
        # Basit ufuk tahmini (ortanın biraz üstü)
        horizon_y = h * 0.45
        
        for det in detections:
            # Gökyüzünde mi?
            if det.center[1] < horizon_y:
                det.confidence *= 1.2  # Gökyüzü bonus
            else:
                det.confidence *= 0.8  # Zemin ceza
                
            # Kenar tespitlerini düşür
            margin = 20
            if (det.x < margin or det.x + det.w > w - margin or
                det.y < margin or det.y + det.h > h - margin):
                det.confidence *= 0.7
                
            if det.confidence >= self.config.min_confidence:
                validated.append(det)
                
        return validated
        
    def _remove_duplicates(self, detections: List[Detection], 
                           iou_threshold: float = 0.5) -> List[Detection]:
        """
        Overlapping tespitleri birleştir (NMS benzeri)
        """
        if len(detections) <= 1:
            return detections
            
        # Güvene göre sırala
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)
            
            # Çakışanları kaldır
            detections = [
                d for d in detections
                if self._calculate_iou(best.bbox, d.bbox) < iou_threshold
            ]
            
        return kept
        
    @staticmethod
    def _calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """Intersection over Union hesapla"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Kesişim
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = max(0, min(x1+w1, x2+w2) - xi)
        hi = max(0, min(y1+h1, y2+h2) - yi)
        
        intersection = wi * hi
        union = w1*h1 + w2*h2 - intersection
        
        return intersection / max(union, 1)
        
    def get_debug_frame(self, frame: np.ndarray, 
                        detections: List[Detection]) -> np.ndarray:
        """
        Debug görüntüsü oluştur
        
        Tespitleri ve maskeleri gösterir.
        """
        debug = frame.copy()
        
        # Tespitleri çiz
        for det in detections:
            color = (0, 255, 0) if det.color_class != 'motion' else (0, 255, 255)
            
            # Bbox
            cv2.rectangle(debug, 
                         (det.x, det.y), 
                         (det.x + det.w, det.y + det.h),
                         color, 2)
                         
            # Merkez
            cv2.circle(debug, 
                      (int(det.center[0]), int(det.center[1])),
                      4, (0, 0, 255), -1)
                      
            # Label
            label = f"{det.color_class}: {det.confidence:.2f}"
            cv2.putText(debug, label,
                       (det.x, det.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        return debug
        
    def reset(self):
        """Dedektör durumunu sıfırla"""
        self._frame_history.clear()
        self._prev_gray = None
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False
        )
