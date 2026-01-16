"""
Debug Görselleştirme Araçları
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class DebugVisualizer:
    """OpenCV tabanlı debug görselleştirme"""
    
    @staticmethod
    def draw_detection_overlay(frame: np.ndarray, detections: List[Dict],
                                lock_state: Dict = None) -> np.ndarray:
        """Tespit ve kilitlenme overlay'i çiz"""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Crosshair
        cx, cy = w // 2, h // 2
        DebugVisualizer._draw_crosshair(output, cx, cy, lock_state)
        
        # Tespitler
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det.get('confidence', 0)
            
            # Renk (takıma göre)
            if det.get('team') == 'red':
                color = (0, 0, 255)
            else:
                color = (255, 100, 0)
                
            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Etiket
            label = f"{conf:.2f}"
            cv2.putText(output, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        # Kilit durumu
        if lock_state and lock_state.get('is_locked'):
            DebugVisualizer._draw_lock_indicator(output, lock_state)
            
        return output
        
    @staticmethod
    def _draw_crosshair(frame: np.ndarray, x: int, y: int, lock_state: Dict):
        """Crosshair çiz"""
        size = 40
        
        if lock_state and lock_state.get('is_valid'):
            color = (0, 0, 255)  # Kırmızı - kilitli
        elif lock_state and lock_state.get('is_locked'):
            color = (0, 200, 255)  # Turuncu - kilitleniyor
        else:
            color = (0, 255, 0)  # Yeşil - hazır
            
        # Çizgiler
        cv2.line(frame, (x-size, y), (x-10, y), color, 2)
        cv2.line(frame, (x+10, y), (x+size, y), color, 2)
        cv2.line(frame, (x, y-size), (x, y-10), color, 2)
        cv2.line(frame, (x, y+10), (x, y+size), color, 2)
        
        # Merkez
        cv2.circle(frame, (x, y), 3, color, -1)
        
    @staticmethod
    def _draw_lock_indicator(frame: np.ndarray, lock_state: Dict):
        """Kilit göstergesi"""
        h, w = frame.shape[:2]
        
        progress = lock_state.get('progress', 0)
        is_valid = lock_state.get('is_valid', False)
        
        # Progress bar
        bar_width = 200
        bar_height = 10
        bar_x = (w - bar_width) // 2
        bar_y = h - 40
        
        # Arka plan
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
                     
        # İlerleme
        fill_width = int(bar_width * progress)
        color = (0, 0, 255) if is_valid else (0, 200, 255)
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
                     
        # Metin
        text = "KİLİTLİ!" if is_valid else f"Kilitleniyor... {progress*100:.0f}%"
        cv2.putText(frame, text, (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                   
    @staticmethod
    def draw_telemetry(frame: np.ndarray, uav_state: Dict) -> np.ndarray:
        """Telemetri bilgilerini çiz"""
        output = frame.copy()
        
        texts = [
            f"Hiz: {uav_state.get('speed', 0):.1f} m/s",
            f"Irtifa: {uav_state.get('altitude', 0):.0f} m",
            f"Yon: {uav_state.get('heading', 0):.0f}°"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(output, text, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        return output
