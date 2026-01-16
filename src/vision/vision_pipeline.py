"""
Vision Pipeline

End-to-end görüntü tabanlı kilitlenme pipeline.
Render → İşleme → Tespit → Takip → Kilitlenme
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class VisionResult:
    """Vision pipeline sonucu"""
    frame: np.ndarray                     # İşlenmiş görüntü
    raw_frame: np.ndarray = None          # Ham render
    detections: List = field(default_factory=list)  # Tespitler
    tracks: List = field(default_factory=list)      # Takip edilen hedefler
    lock_status: Dict = field(default_factory=dict)  # Kilitlenme durumu
    processing_time_ms: float = 0.0       # İşlem süresi
    frame_number: int = 0


class VisionPipeline:
    """
    End-to-end görüntü tabanlı kilitlenme pipeline
    
    Akış:
    1. 3D Render (Panda3D) → RGB Frame
    2. Post-processing → Gerçekçi efektler
    3. Image Detection → Hedef tespiti (CV tabanlı)
    4. Tracking → ID takibi
    5. Lock-on Validation → Kilitlenme kontrolü
    
    Usage:
        pipeline = VisionPipeline(config)
        result = pipeline.process(camera_pos, camera_orient, uav_states, sim_time)
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        self._frame_count = 0
        self._initialized = False
        
        # Çözünürlük
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        
        # Alt modüller (lazy init)
        self._renderer = None
        self._post_processor = None
        self._detector = None
        self._tracker = None
        self._lock_validator = None
        
        # Konfigürasyonlar
        self._renderer_config = config.get('renderer', {})
        self._post_config = config.get('post_processing', {})
        self._detector_config = config.get('detector', {})
        self._tracker_config = config.get('tracker', {})
        self._lock_config = config.get('lock_on', {})
        
        # Debug
        self._debug_mode = config.get('debug', False)
        
    def initialize(self) -> bool:
        """Pipeline'ı başlat"""
        if self._initialized:
            return True
            
        try:
            # 1. Renderer
            from src.render.offscreen_renderer import create_renderer
            self._renderer = create_renderer(
                self.width, self.height,
                use_panda3d=self._renderer_config.get('use_panda3d', True)
            )
            
            # 2. Post-processor
            from src.render.post_processing import PostProcessing, PostProcessingConfig
            pp_config = PostProcessingConfig(
                lens_distortion=self._post_config.get('lens_distortion', True),
                sensor_noise=self._post_config.get('sensor_noise', True),
                vignette=self._post_config.get('vignette', True),
                motion_blur=self._post_config.get('motion_blur', False)
            )
            self._post_processor = PostProcessing(pp_config)
            
            # 3. Detector (görüntü tabanlı)
            from src.vision.image_detector import ImageBasedDetector, ImageDetectorConfig
            det_config = ImageDetectorConfig(
                min_area=self._detector_config.get('min_area', 50),
                max_area=self._detector_config.get('max_area', 50000),
                min_confidence=self._detector_config.get('min_confidence', 0.3)
            )
            self._detector = ImageBasedDetector(det_config)
            
            # 4. Tracker
            from src.vision.tracker import SimpleTracker
            self._tracker = SimpleTracker(
                max_age=self._tracker_config.get('max_age', 30),
                min_hits=self._tracker_config.get('min_hits', 3)
            )
            
            # 5. Lock validator
            from src.vision.lock_on import CompetitionLockValidator
            self._lock_validator = CompetitionLockValidator()
            
            self._initialized = True
            print(f"✅ VisionPipeline initialized ({self.width}x{self.height})")
            return True
            
        except Exception as e:
            print(f"❌ VisionPipeline init failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def setup_scene(self, scenario: dict = None):
        """
        Sahneyi hazırla
        
        Args:
            scenario: Senaryo konfigürasyonu
        """
        if self._renderer and hasattr(self._renderer, 'setup_scene'):
            terrain_config = scenario.get('world', {}) if scenario else {}
            self._renderer.setup_scene(terrain_config)
            
    def process(self,
                camera_pos: np.ndarray,
                camera_orient: np.ndarray,
                uav_states: List[dict],
                sim_time: float,
                dt: float = 0.016,
                camera_velocity: np.ndarray = None) -> VisionResult:
        """
        Tam pipeline işle
        
        Args:
            camera_pos: Kamera pozisyonu [x, y, z]
            camera_orient: Kamera oryantasyonu [roll, pitch, yaw]
            uav_states: Hedef İHA durumları
            sim_time: Simülasyon zamanı
            dt: Zaman adımı
            camera_velocity: Kamera hız vektörü
            
        Returns:
            VisionResult
        """
        start_time = time.perf_counter()
        
        if not self._initialized:
            if not self.initialize():
                return VisionResult(
                    frame=np.zeros((self.height, self.width, 3), dtype=np.uint8),
                    frame_number=self._frame_count
                )
                
        # 1. 3D Render
        raw_frame = self._renderer.render_frame(
            camera_pos, camera_orient, uav_states
        )
        
        # 2. Post-processing
        processed_frame = self._post_processor.process(
            raw_frame,
            velocity=camera_velocity
        )
        
        # 3. Görüntü tabanlı tespit
        detections = self._detector.detect(processed_frame, use_motion=True)
        
        # 4. Takip
        tracks = self._tracker.update([
            {
                'bbox': d.bbox,
                'center': d.center,
                'confidence': d.confidence,
                'class': d.color_class
            }
            for d in detections
        ])
        
        # 5. Kilitlenme durumu
        lock_status = self._update_lock_status(tracks, sim_time)
        
        # İşlem süresi
        processing_time = (time.perf_counter() - start_time) * 1000
        
        self._frame_count += 1
        
        return VisionResult(
            frame=processed_frame,
            raw_frame=raw_frame if self._debug_mode else None,
            detections=detections,
            tracks=tracks,
            lock_status=lock_status,
            processing_time_ms=processing_time,
            frame_number=self._frame_count
        )
        
    def _update_lock_status(self, tracks: List, sim_time: float) -> Dict:
        """Kilitlenme durumunu güncelle"""
        if not self._lock_validator or not tracks:
            return {
                'is_locked': False,
                'target_id': None,
                'lock_progress': 0.0,
                'valid_for_scoring': False
            }
            
        # En iyi hedefi seç (en yüksek güvenli track)
        best_track = max(tracks, key=lambda t: t.get('confidence', 0))
        
        # Frame validasyonu
        frame_result = self._lock_validator.validate_frame(
            {
                'bbox': best_track.get('bbox'),
                'center': best_track.get('center'),
                'id': best_track.get('id'),
                'confidence': best_track.get('confidence', 0.5)
            },
            (self.width, self.height),
            sim_time
        )
        
        # History'e ekle
        self._lock_validator.add_frame_to_history(
            frame_result, 
            best_track.get('id'), 
            sim_time
        )
        
        # Continuous lock kontrolü
        lock_result = self._lock_validator.check_continuous_lock(
            best_track.get('id'),
            sim_time
        )
        
        return {
            'is_locked': lock_result.get('is_locked', False),
            'target_id': best_track.get('id'),
            'lock_progress': lock_result.get('continuous_duration', 0) / 4.0,  # 4s hedef
            'valid_for_scoring': lock_result.get('valid_for_scoring', False),
            'frame_valid': frame_result.get('valid_frame', False)
        }
        
    def get_debug_frame(self, result: VisionResult) -> np.ndarray:
        """
        Debug görüntüsü oluştur
        
        Tespitleri, takipleri ve durumu gösterir.
        """
        import cv2
        
        debug = result.frame.copy()
        
        # Tespitler
        for det in result.detections:
            cv2.rectangle(debug,
                         (det.x, det.y),
                         (det.x + det.w, det.y + det.h),
                         (0, 255, 0), 1)
                         
        # Takipler
        for track in result.tracks:
            bbox = track.get('bbox', (0, 0, 0, 0))
            track_id = track.get('id', -1)
            
            cv2.rectangle(debug,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                         (255, 0, 0), 2)
                         
            cv2.putText(debug, f"ID:{track_id}",
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                       
        # Lock durumu
        lock = result.lock_status
        if lock.get('is_locked'):
            cv2.putText(debug, "LOCKED",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            progress = lock.get('lock_progress', 0)
            cv2.putText(debug, f"Locking: {progress*100:.0f}%",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                       
        # İşlem süresi
        cv2.putText(debug, f"{result.processing_time_ms:.1f}ms",
                   (debug.shape[1] - 80, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                   
        return debug
        
    def cleanup(self):
        """Kaynakları temizle"""
        if self._renderer:
            self._renderer.cleanup()
        if self._detector:
            self._detector.reset()
        self._initialized = False
        
    @property
    def is_initialized(self) -> bool:
        return self._initialized
        
    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)
