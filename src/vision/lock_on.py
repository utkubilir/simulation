"""
Lock-On State Machine (vNext)

Competition-aligned 4-second lock-on mechanic with:
- IDLE -> LOCKING -> SUCCESS state transitions
- Configurable tolerance and confidence thresholds
- Deterministic timing using simulation time
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque
from src.vision.validator import GeometryValidator, ValidationResult

class LockState(Enum):
    """Lock-on state machine states"""
    IDLE = "idle"           # No valid target or lost track
    LOCKING = "locking"     # Valid target, accumulating/tracking valid segments
    SUCCESS = "success"     # Lock requirement met (4s continuous in 5s window)

@dataclass
class LockConfig:
    """Lock-on system configuration"""
    # 2025 Rule constraints
    window_seconds: float = 5.0           # Success evaluation window
    required_continuous_seconds: float = 4.0 # Required continuous valid lock
    tolerance_below: float = 1.0          # Pre-window tolerance (implied 5.0 window logic)
    tolerance_above: float = 0.5          # Post-window tolerance (not used in main logic but config placeholder)
    
    # Validation rules
    size_threshold: float = 0.01          # Minimum coverage ratio (Safe margin > 0.5%)
    margin_horizontal: float = 0.5        # Horizontal tolerance (fraction of bbox width)
    margin_vertical: float = 0.5          # Vertical tolerance (fraction of bbox height)
    min_confidence: float = 0.5           # Minimum detection confidence
    
    frame_width: int = 640                # Camera frame width
    frame_height: int = 480               # Camera frame height
    
    # Legacy/Test Support
    success_duration: Optional[float] = None 

    def __post_init__(self):
        if self.success_duration is not None:
            self.required_continuous_seconds = self.success_duration

    @property
    def crosshair(self) -> Tuple[int, int]:
        """Crosshair position (center of frame)"""
        return (self.frame_width // 2, self.frame_height // 2)

@dataclass
class LockStatus:
    """Current lock status snapshot with audit details"""
    state: LockState = LockState.IDLE
    target_id: Optional[int] = None
    lock_time: float = 0.0              # Current continuous segment duration
    is_valid: bool = False
    
    # Audit fields
    dx: float = 0.0
    dy: float = 0.0
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    coverage_w: float = 0.0
    coverage_h: float = 0.0
    
    # Verification Booleans
    size_ok: bool = False
    center_ok: bool = False
    confidence_ok: bool = False
    id_stable: bool = True
    
    reason_invalid: Optional[str] = None
    progress: float = 0.0
    lock_end_time: Optional[float] = None # For success events

@dataclass
class ValidSegment:
    """A continuous duration of valid lock"""
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass
class ScoreEvent:
    """Scoring event emitted on state transitions"""
    event_type: str                     # "success_lock" or "false_lock"
    target_id: int
    timestamp: float
    lock_duration: float

class LockOnStateMachine:
    """
    Competition-aligned lock-on state machine (2025 Rules).
    
    Key Logic:
    - Validates frames using GeometryValidator (Size >= 6%, Center alignment).
    - Tracks 'Valid Segments' of continuous lock.
    - Success Condition: Identify a 4.0s continuous valid segment fully contained 
      within a 5.0s window ending at 'lock_end_time'.
    """
    
    def __init__(self, config: LockConfig = None):
        self.config = config or LockConfig()
        
        # Enforce safe margin warning if exactly 0.05
        if abs(self.config.size_threshold - 0.05) < 1e-6:
             self.config.size_threshold = 0.06 # Clamp to safe margin
        
        self.validator = GeometryValidator(
            size_threshold=self.config.size_threshold,
            margin_h=self.config.margin_horizontal,
            margin_v=self.config.margin_vertical
        )
        
        # State
        self._state = LockState.IDLE
        self._target_id: Optional[int] = None
        
        # Current continuously tracking segment
        self._current_segment_start: Optional[float] = None
        self._last_valid_time: float = 0.0
        self._lock_time: float = 0.0 # Accumulated continuous duration
        
        # Lock History for Window Check
        # List of ValidSegment (completed or ongoing)
        self._valid_segments: List[ValidSegment] = []
        
        # Frame-by-frame status
        self._current_val_result: Optional[ValidationResult] = None
        self._current_valid: bool = False
        
        # Scoring & Metrics
        self._score_events: List[ScoreEvent] = []
        self._correct_locks: int = 0
        self._incorrect_locks: int = 0
        
        self._first_valid_lock_time: Optional[float] = None
        self._first_success_lock_time: Optional[float] = None
        self._valid_lock_time_total: float = 0.0
        self._invalid_reason_counts: Dict[str, int] = {}

    @property
    def state(self) -> LockState:
        return self._state
        
    @property
    def target_id(self) -> Optional[int]:
        return self._target_id
        
    @property
    def current_continuous_duration(self) -> float:
        """Duration of the current ongoing valid segment"""
        # Return accumulated time which is robust to dt summing
        return self._lock_time

    @property
    def progress(self) -> float:
        """Progress towards successful lock based on current continuous segment"""
        return min(1.0, self.current_continuous_duration / self.config.required_continuous_seconds)
        
    def update(self, tracks: List, sim_time: float, dt: float) -> LockStatus:
        """
        Update lock state machine.
        Args:
            tracks: List of detection tracks
            sim_time: Current simulation time (authoritative)
            dt: Time step
        """
        # 1. Identify best target candidate strictly
        candidate, val_result = self._find_best_candidate(tracks)
        self._current_val_result = val_result
        self._current_valid = val_result.is_valid
        
        # Update metrics
        if self._current_valid and candidate:
            self._valid_lock_time_total += dt
            if self._first_valid_lock_time is None:
                self._first_valid_lock_time = sim_time
        
        if not self._current_valid and val_result.reason:
            count = self._invalid_reason_counts.get(val_result.reason, 0)
            self._invalid_reason_counts[val_result.reason] = count + 1

        # 2. State Machine Logic
        if self._state == LockState.IDLE:
            if self._current_valid and candidate:
                # Start locking
                self._state = LockState.LOCKING
                self._target_id = candidate['id']
                self._current_segment_start = sim_time
                self._last_valid_time = sim_time
                self._lock_time = dt # Start with first frame duration
                
        elif self._state == LockState.LOCKING:
            if not candidate:
                # Lost target completely
                self._close_current_segment(sim_time)
                self._state = LockState.IDLE
                self._target_id = None
                
            elif candidate['id'] != self._target_id:
                # Target switch - Close current segment first, then handle penalty
                self._close_current_segment(sim_time)
                self._handle_target_switch(candidate, sim_time)
                
            elif not self._current_valid:
                # Valid target became invalid (geometry/confidence)
                # Keep target_id but break continuity
                self._close_current_segment(sim_time)
                self._current_segment_start = None 
                
            else:
                # Continue valid lock
                if self._current_segment_start is None:
                    # Re-start valid segment
                    self._current_segment_start = sim_time
                    self._lock_time = 0.0 # Reset accumulator for new segment
                
                self._last_valid_time = sim_time
                self._lock_time += dt
                
                # Check Success Rule (lock_time updated)
                if self._check_success_condition(sim_time):
                    self._handle_success(sim_time)

        elif self._state == LockState.SUCCESS:
            self._state = LockState.IDLE
            self._target_id = None
            self._close_current_segment(sim_time)
            # Immediate re-lock possible next frame if valid
        
        # 3. Clean up old history (older than window + margin)
        cutoff = sim_time - (self.config.window_seconds * 2)
        self._valid_segments = [s for s in self._valid_segments if s.end_time > cutoff]

        return self.get_status()

    def _find_best_candidate(self, tracks) -> Tuple[Optional[Dict], ValidationResult]:
        """Find best target and validate geometry."""
        best_cand = None
        best_res = ValidationResult(is_valid=False, reason=GeometryValidator.REASON_TARGET_LOST)
        min_dist = float('inf')
        
        # Just pick closest to center for now, then validate
        cx, cy = self.config.crosshair
        
        for track in tracks:
            # Normalize track dict
            t_data = self._normalize_track(track)
            if not t_data: continue
            
            # Confidence check first
            if t_data['confidence'] < self.config.min_confidence:
                continue
                
            # Strict distance metric as requested (min sqrt(dx^2+dy^2))
            center = t_data['center']
            dist = np.sqrt((center[0]-cx)**2 + (center[1]-cy)**2)
            
            if dist < min_dist:
                # Candidate found, now strict validation
                res = self.validator.validate_lock_candidate(
                    bbox=t_data['bbox'],
                    frame_width=self.config.frame_width,
                    frame_height=self.config.frame_height,
                    center=center
                )
                
                # We prioritize valid candidates even if further away? 
                # Ideally yes, but typically we track one.
                # Let's verify strict validity.
                if res.is_valid:
                    min_dist = dist
                    best_cand = t_data
                    best_res = res
                elif best_cand is None:
                    # Keep invalid candidate if no valid one found yet (for reporting reason)
                    best_cand = t_data
                    best_res = res
                    # Update reason if it was just "valid"
                    if best_res.is_valid: 
                        best_res.is_valid = False
                        best_res.reason = GeometryValidator.REASON_CONFIDENCE_LOW # Should not happen given check above
        
        # If no tracks
        if not tracks:
            return None, ValidationResult(is_valid=False, reason=GeometryValidator.REASON_TARGET_LOST)
            
        return best_cand, best_res

    def _normalize_track(self, track) -> Optional[Dict]:
        """Extract standardized dict from track object/dict."""
        if hasattr(track, 'center'):
            return {
                'id': track.id,
                'center': track.center,
                'bbox': track.bbox,
                'confidence': track.confidence
            }
        elif isinstance(track, dict):
            return {
                'id': track.get('id') or track.get('world_id'),
                'center': track.get('center'),
                'bbox': track.get('bbox'),
                'confidence': track.get('confidence', 0)
            }
        return None

    def _close_current_segment(self, sim_time: float):
        """Finalize the current valid segment and store it."""
        if self._current_segment_start is not None:
            # End time is last valid time (sim_time or last_step)
            # Use last_valid_time for precision
            seg = ValidSegment(self._current_segment_start, self._last_valid_time)
            self._valid_segments.append(seg)
            self._current_segment_start = None
            self._lock_time = 0.0

    def _check_success_condition(self, sim_time: float) -> bool:
        """
        Check 2025 Rule: 4.0s continuous valid lock within 5.0s window.
        """
        # The simplest and most robust way to check "continuous duration" 
        # is to use the accumulated `_lock_time`.
        
        # Optimization: if current continuous is >= 4.0 (with epsilon), it's a success immediately
        if self._lock_time >= (self.config.required_continuous_seconds - 1e-6):
            return True
            
        return False

    def _handle_target_switch(self, new_cand: Dict, sim_time: float):
        self._incorrect_locks += 1
        self._close_current_segment(sim_time)
        self._score_events.append(ScoreEvent("false_lock", self._target_id, sim_time, 0)) # Duration?
        
        # Switch immediately
        self._target_id = new_cand['id']
        self._current_segment_start = sim_time
        self._last_valid_time = sim_time

    def _handle_success(self, sim_time: float):
        self._state = LockState.SUCCESS
        self._correct_locks += 1
        if self._first_success_lock_time is None:
            self._first_success_lock_time = sim_time
            
        lock_dur = self.current_continuous_duration
        self._score_events.append(ScoreEvent("success_lock", self._target_id, sim_time, lock_dur))

    def get_status(self) -> LockStatus:
        res = self._current_val_result or ValidationResult(is_valid=False)
        
        # Populate detailed status
        return LockStatus(
            state=self._state,
            target_id=self._target_id,
            lock_time=self.current_continuous_duration,
            is_valid=self._current_valid,
            dx=res.dx,
            dy=res.dy,
            bbox_w=res.coverage_w * self.config.frame_width, # Approx reconstruction
            bbox_h=res.coverage_h * self.config.frame_height,
            coverage_w=res.coverage_w,
            coverage_h=res.coverage_h,
            
            size_ok=(res.reason != GeometryValidator.REASON_SIZE_TOO_SMALL),
            center_ok=(res.reason != GeometryValidator.REASON_CENTER_OUTSIDE),
            confidence_ok=(res.reason != GeometryValidator.REASON_CONFIDENCE_LOW),
            id_stable=True, # Handled by reset logic
            
            reason_invalid=res.reason,
            progress=self.progress,
            lock_end_time=self._first_success_lock_time if self._state == LockState.SUCCESS else None
        )

    def get_score(self) -> Dict:
        return {
            'correct_locks': self._correct_locks,
            'incorrect_locks': self._incorrect_locks,
            'total_score': self._correct_locks * 100 - self._incorrect_locks * 50
        }

    def get_metrics(self) -> Dict:
        return {
            'correct_locks': self._correct_locks,
            'incorrect_locks': self._incorrect_locks,
            'total_score': self.get_score()['total_score'],
            'time_to_first_valid_lock': self._first_valid_lock_time,
            'time_to_first_success_lock': self._first_success_lock_time,
            'valid_lock_time_total': self._valid_lock_time_total,
            # 'valid_lock_ratio': ... (calculated externally or here)
            'invalid_reason_counts': self._invalid_reason_counts,
            'score_events': [
                {'type': e.event_type, 'target_id': e.target_id, 'timestamp': e.timestamp, 'duration': e.lock_duration}
                for e in self._score_events
            ]
        }
    
    def reset(self):
        self._state = LockState.IDLE
        self._target_id = None
        self._current_segment_start = None
        self._valid_segments.clear()
        self._correct_locks = 0
        self._incorrect_locks = 0
        self._score_events.clear()
        self._current_valid = False
        self._valid_lock_time_total = 0.0
        self._first_valid_lock_time = None
        self._first_success_lock_time = None
        self._invalid_reason_counts.clear()


class CompetitionLockValidator:
    """
    Şartname 6.1.1'e uygun kilitlenme doğrulama
    
    Yarışma kuralları:
    - 4 saniye kesintisiz kilitlenme (5 saniye pencere içinde)
    - 1 saniye tolerans (ileri veya geri dağıtılabilir)
    - Hedef ekranın minimum %5'ini kaplamalı
    - Hedef vuruş alanı: ekranın %30'u
    - Art arda aynı hedefe kilitlenemez
    """
    
    def __init__(self, config: LockConfig = None):
        self.config = config or LockConfig()
        
        # Şartname değerleri (override config if needed)
        self.required_lock_duration = 4.0    # 4 saniye
        self.tolerance_duration = 1.0         # 1 saniye tolerans
        self.window_duration = 5.0            # 5 saniye pencere
        self.min_screen_coverage = 0.05       # %5 minimum kaplama
        self.target_area_ratio = 0.30         # %30 vuruş alanı
        
        # Frame geçmişi
        self.lock_history: List[Dict] = []
        
        # Art arda kilitleme kontrolü
        self.last_locked_target_id = None
        
        # Metriks
        self.total_locks = 0
        self.valid_locks = 0
        self.invalid_locks = 0
        
    def validate_frame(self, detection: Dict, frame_size: Tuple[int, int],
                       sim_time: float) -> Dict:
        """
        Tek frame için kilitlenme doğrulaması
        
        Args:
            detection: {bbox: [x, y, w, h], center: [cx, cy], id, confidence}
            frame_size: (width, height)
            sim_time: Simülasyon zamanı
            
        Returns:
            {
                'in_target_area': bool,
                'size_sufficient': bool,
                'bbox_covers_target': bool,
                'valid_frame': bool,
                'reason': str or None
            }
        """
        result = {
            'in_target_area': False,
            'size_sufficient': False,
            'bbox_covers_target': True,
            'valid_frame': False,
            'reason': None
        }
        
        if not detection or 'bbox' not in detection:
            result['reason'] = 'no_detection'
            return result
            
        bbox = detection['bbox']
        center = detection.get('center', (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2))
        
        w, h = frame_size
        
        # 1. Hedef vuruş alanı kontrolü (ekran merkezi çevresinde %30)
        target_area_w = w * self.target_area_ratio
        target_area_h = h * self.target_area_ratio
        
        screen_center = (w / 2, h / 2)
        
        # Tespit merkezi hedef alanında mı?
        in_area_x = abs(center[0] - screen_center[0]) < target_area_w / 2
        in_area_y = abs(center[1] - screen_center[1]) < target_area_h / 2
        result['in_target_area'] = in_area_x and in_area_y
        
        if not result['in_target_area']:
            result['reason'] = 'outside_target_area'
            return result
        
        # 2. Boyut kontrolü (minimum %5)
        bbox_area = bbox[2] * bbox[3]  # w * h
        screen_area = w * h
        coverage_ratio = bbox_area / screen_area if screen_area > 0 else 0
        
        # Genişlik veya yükseklik bazlı kontrol
        width_ratio = bbox[2] / w if w > 0 else 0
        height_ratio = bbox[3] / h if h > 0 else 0
        
        result['size_sufficient'] = (coverage_ratio >= self.min_screen_coverage or
                                    width_ratio >= np.sqrt(self.min_screen_coverage) or
                                    height_ratio >= np.sqrt(self.min_screen_coverage))
        
        if not result['size_sufficient']:
            result['reason'] = 'size_too_small'
            return result
        
        # 3. Genel sonuç
        result['valid_frame'] = (
            result['in_target_area'] and 
            result['size_sufficient'] and
            result['bbox_covers_target']
        )
        
        return result
    
    def add_frame_to_history(self, frame_result: Dict, target_id: int, sim_time: float):
        """Frame sonucunu geçmişe ekle"""
        self.lock_history.append({
            'time': sim_time,
            'valid': frame_result['valid_frame'],
            'target_id': target_id,
            'result': frame_result
        })
        
        # Eski kayıtları temizle (pencere + tolerans dışındakiler)
        cutoff = sim_time - (self.window_duration + self.tolerance_duration + 1.0)
        self.lock_history = [f for f in self.lock_history if f['time'] > cutoff]
    
    def check_continuous_lock(self, target_id: int, end_time: float) -> Dict:
        """
        4 saniye kesintisiz kilitlenme kontrolü (1 sn toleranslı)
        
        Şartname: 5 saniyelik pencerede 4 saniye kesintisiz lock
        
        Returns:
            {
                'is_locked': bool,
                'continuous_duration': float,
                'lock_start_time': float or None,
                'lock_end_time': float or None,
                'valid_for_scoring': bool (art arda aynı hedef değilse)
            }
        """
        result = {
            'is_locked': False,
            'continuous_duration': 0.0,
            'lock_start_time': None,
            'lock_end_time': None,
            'valid_for_scoring': True
        }
        
        # Pencere başlangıcı
        window_start = end_time - self.window_duration
        
        # Penceredeki frame'leri al (bu hedef için)
        frames_in_window = [
            f for f in self.lock_history 
            if window_start <= f['time'] <= end_time and f['target_id'] == target_id
        ]
        
        if not frames_in_window:
            return result
            
        # Geçerli frame'ler
        valid_frames = [f for f in frames_in_window if f['valid']]
        
        if not valid_frames:
            return result
            
        # Kesintisiz süreyi hesapla
        # Frame'leri zamana göre sırala
        valid_frames.sort(key=lambda x: x['time'])
        
        max_continuous = 0.0
        current_streak_start = None
        current_streak = 0.0
        last_time = None
        
        # Tolerans: küçük boşlukları (<=100ms) görmezden gel
        gap_tolerance = 0.1  # 100ms
        
        for frame in valid_frames:
            if last_time is None:
                current_streak_start = frame['time']
                current_streak = 0.0
            elif frame['time'] - last_time <= gap_tolerance + 0.02:  # dt dahil
                current_streak = frame['time'] - current_streak_start
            else:
                # Boşluk var, streak sıfırla
                if current_streak > max_continuous:
                    max_continuous = current_streak
                    result['lock_start_time'] = current_streak_start
                current_streak_start = frame['time']
                current_streak = 0.0
                
            last_time = frame['time']
        
        # Son streak'i kontrol et
        if current_streak > max_continuous:
            max_continuous = current_streak
            result['lock_start_time'] = current_streak_start
            
        result['continuous_duration'] = max_continuous
        result['lock_end_time'] = last_time
        
        # 4 saniye kontrolü (biraz toleransla)
        lock_threshold = self.required_lock_duration - 0.2  # 200ms tolerans
        result['is_locked'] = max_continuous >= lock_threshold
        
        # Art arda aynı hedef kontrolü
        if result['is_locked']:
            if target_id == self.last_locked_target_id:
                result['valid_for_scoring'] = False
            else:
                result['valid_for_scoring'] = True
                
        return result
    
    def register_successful_lock(self, target_id: int, timestamp: float):
        """Başarılı kilitlenmeyi kaydet"""
        self.last_locked_target_id = target_id
        self.total_locks += 1
        self.valid_locks += 1
        
    def register_invalid_lock(self, target_id: int, timestamp: float, reason: str = None):
        """Geçersiz kilitlenmeyi kaydet"""
        self.total_locks += 1
        self.invalid_locks += 1
        
    def can_lock_target(self, target_id: int) -> bool:
        """Bu hedefe kilitlenebilir mi? (art arda kontrolü)"""
        return target_id != self.last_locked_target_id
    
    def get_metrics(self) -> Dict:
        """Validatör metrikleri"""
        return {
            'total_locks': self.total_locks,
            'valid_locks': self.valid_locks,
            'invalid_locks': self.invalid_locks,
            'last_locked_target': self.last_locked_target_id,
            'history_size': len(self.lock_history)
        }
    
    def reset(self):
        """Validatörü sıfırla"""
        self.lock_history.clear()
        self.last_locked_target_id = None
        self.total_locks = 0
        self.valid_locks = 0
        self.invalid_locks = 0


LockOnSystem = LockOnStateMachine

