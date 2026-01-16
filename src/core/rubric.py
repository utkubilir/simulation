"""
Evaluation Rubric - Phase 3

Competition-style metrics for simulation run evaluation.
All metrics are computed deterministically from frames.jsonl data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import math


@dataclass
class RubricMetrics:
    """Extended evaluation metrics for competition scoring."""
    
    # Existing (from Phase 0/1)
    time_to_first_lock: Optional[float] = None
    lock_ratio: float = 0.0
    correct_locks: int = 0
    incorrect_locks: int = 0
    total_detections: int = 0
    total_frames: int = 0
    duration: float = 0.0
    final_score: int = 0
    
    # Phase 3: New rubric metrics
    false_lock_rate: float = 0.0
    lock_stability_index: float = 0.0
    longest_continuous_lock: float = 0.0
    reacquire_time_mean: Optional[float] = None
    reacquire_time_median: Optional[float] = None
    reacquire_count: int = 0
    track_continuity_index: float = 0.0
    avg_track_age: float = 0.0
    angular_accuracy_mean_deg: Optional[float] = None
    angular_accuracy_max_deg: Optional[float] = None
    
    # Phase 3 Polish: Locked vs Valid metrics
    locked_time_total: float = 0.0
    locked_ratio: float = 0.0
    valid_lock_time_total: float = 0.0
    valid_lock_ratio: float = 0.0
    valid_longest_continuous_lock: float = 0.0
    time_to_first_valid_lock: Optional[float] = None
    valid_lock_count: int = 0
    invalid_lock_time_total: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            # Existing
            'time_to_first_lock': self.time_to_first_lock,
            'lock_ratio': self.lock_ratio,
            'correct_locks': self.correct_locks,
            'incorrect_locks': self.incorrect_locks,
            'total_detections': self.total_detections,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'final_score': self.final_score,
            # Phase 3 new
            'false_lock_rate': self.false_lock_rate,
            'lock_stability_index': self.lock_stability_index,
            'longest_continuous_lock': self.longest_continuous_lock,
            'reacquire_time_mean': self.reacquire_time_mean,
            'reacquire_time_median': self.reacquire_time_median,
            'reacquire_count': self.reacquire_count,
            'track_continuity_index': self.track_continuity_index,
            'avg_track_age': self.avg_track_age,
            'angular_accuracy_mean_deg': self.angular_accuracy_mean_deg,
            'angular_accuracy_max_deg': self.angular_accuracy_max_deg,
            # Phase 3 Polish
            'locked_time_total': self.locked_time_total,
            'locked_ratio': self.locked_ratio,
            'valid_lock_time_total': self.valid_lock_time_total,
            'valid_lock_ratio': self.valid_lock_ratio,
            'valid_longest_continuous_lock': self.valid_longest_continuous_lock,
            'time_to_first_valid_lock': self.time_to_first_valid_lock,
            'valid_lock_count': self.valid_lock_count,
            'invalid_lock_time_total': self.invalid_lock_time_total,
        }
    
    def csv_header(self) -> str:
        """CSV header row."""
        return ','.join(self.to_dict().keys())
    
    def csv_row(self) -> str:
        """CSV data row."""
        values = []
        for v in self.to_dict().values():
            if v is None:
                values.append('')
            else:
                values.append(str(v))
        return ','.join(values)


class RubricCalculator:
    """
    Calculates competition-style evaluation rubric from frame data.
    
    All calculations are deterministic and do not depend on wall-clock time.
    """
    
    def __init__(
        self,
        camera_fov_deg: float = 60.0,
        camera_resolution: tuple = (640, 480),
    ):
        self.camera_fov_deg = camera_fov_deg
        self.camera_resolution = camera_resolution
        
        # Derive vertical FOV from aspect ratio
        aspect_ratio = camera_resolution[0] / camera_resolution[1]
        self.horizontal_fov_deg = camera_fov_deg
        self.vertical_fov_deg = camera_fov_deg / aspect_ratio
        
        # Degrees per pixel
        self.deg_per_px_h = self.horizontal_fov_deg / camera_resolution[0]
        self.deg_per_px_v = self.vertical_fov_deg / camera_resolution[1]
        
    def calculate(self, frames: List[Dict]) -> RubricMetrics:
        """
        Calculate all rubric metrics from frame data.
        
        Args:
            frames: List of frame dicts (from frames.jsonl or in-memory)
            
        Returns:
            RubricMetrics with all fields populated
        """
        if not frames:
            return RubricMetrics()
            
        metrics = RubricMetrics()
        
        # Basic counts
        metrics.total_frames = len(frames)
        metrics.duration = frames[-1].get('t', 0.0) if frames else 0.0
        
        # Detection counting
        total_detections = 0
        for f in frames:
            dets = f.get('detections', [])
            total_detections += len(dets) if isinstance(dets, list) else 0
        metrics.total_detections = total_detections
        
        # Lock analysis vars
        lock_frames = []
        lock_durations = []
        valid_lock_durations = []  # For valid lock stability
        
        # Timers / State for lock analysis
        current_lock_start = None
        current_valid_lock_start = None
        
        prev_locked = False
        prev_valid_locked = False
        
        reacquire_times = []
        lock_lost_time = None
        
        # Valid lock counts (based on rising edge of valid)
        valid_locks_triggered = 0
        
        # Time accumulators
        total_locked_time = 0.0
        total_valid_locked_time = 0.0
        
        # Angular accuracy tracking
        angular_errors = []
        
        # Need dt for integration if timestamps not perfect, 
        # but better to use diff between frames
        
        for i, f in enumerate(frames):
            t = f['t']
            # Calculate dt from previous frame
            if i > 0:
                dt = t - frames[i-1]['t']
            else:
                dt = 0.0 # First frame has no duration contribution usually, or minimal
                
            lock = f.get('lock', {})
            is_locked = lock.get('locked', False)
            is_valid = lock.get('valid', False) and is_locked
            
            score = f.get('score', {})
            
            # Score tracking (max seen)
            metrics.correct_locks = max(metrics.correct_locks, score.get('correct_locks', 0))
            metrics.incorrect_locks = max(metrics.incorrect_locks, score.get('incorrect_locks', 0))
            metrics.final_score = max(metrics.final_score, score.get('total', 0))
            
            # --- General Lock Analysis ---
            if is_locked:
                lock_frames.append(i)
                total_locked_time += dt
                
                # Time to first lock (visual)
                if metrics.time_to_first_lock is None:
                    metrics.time_to_first_lock = t
                    
                # Reacquire logic
                if not prev_locked and lock_lost_time is not None:
                    reacquire_time = t - lock_lost_time
                    reacquire_times.append(reacquire_time)
                    
                # Continuous segment tracking
                if current_lock_start is None:
                    current_lock_start = t
                    
                # Angular accuracy (dx, dy in pixels)
                dx = lock.get('dx')
                dy = lock.get('dy')
                if dx is not None and dy is not None:
                    err_h_deg = abs(dx) * self.deg_per_px_h
                    err_v_deg = abs(dy) * self.deg_per_px_v
                    total_err_deg = math.sqrt(err_h_deg**2 + err_v_deg**2)
                    angular_errors.append(total_err_deg)
            else:
                # Lock lost
                if prev_locked and current_lock_start is not None:
                    lock_duration = t - current_lock_start
                    lock_durations.append(lock_duration)
                    current_lock_start = None
                    lock_lost_time = t
            
            # --- Valid Lock Analysis ---
            if is_valid:
                total_valid_locked_time += dt
                
                # Time to first VALID lock
                if metrics.time_to_first_valid_lock is None:
                    metrics.time_to_first_valid_lock = t
                    
                # Valid lock count (rising edge)
                if not prev_valid_locked:
                    valid_locks_triggered += 1
                    
                # Valid continuous segment
                if current_valid_lock_start is None:
                    current_valid_lock_start = t
            else:
                # Valid lock lost
                if prev_valid_locked and current_valid_lock_start is not None:
                    v_duration = t - current_valid_lock_start
                    valid_lock_durations.append(v_duration)
                    current_valid_lock_start = None
            
            # Update state for next frame
            prev_locked = is_locked
            prev_valid_locked = is_valid
            
        # Close final segments if active at end
        final_t = frames[-1]['t'] if frames else 0.0
        if current_lock_start is not None:
            lock_durations.append(final_t - current_lock_start)
        if current_valid_lock_start is not None:
            valid_lock_durations.append(final_t - current_valid_lock_start)
            
        # --- Metrics Population ---
        
        # 1. Existing Rubric
        if metrics.total_frames > 0:
            metrics.lock_ratio = len(lock_frames) / metrics.total_frames
            
        total_locks = metrics.correct_locks + metrics.incorrect_locks
        if total_locks > 0:
            metrics.false_lock_rate = metrics.incorrect_locks / total_locks
            
        if lock_durations:
            metrics.longest_continuous_lock = max(lock_durations)
            if metrics.duration > 0:
                metrics.lock_stability_index = metrics.longest_continuous_lock / metrics.duration
                
        metrics.reacquire_count = len(reacquire_times)
        if reacquire_times:
            metrics.reacquire_time_mean = sum(reacquire_times) / len(reacquire_times)
            sorted_times = sorted(reacquire_times)
            mid = len(sorted_times) // 2
            if len(sorted_times) % 2 == 0:
                metrics.reacquire_time_median = (sorted_times[mid-1] + sorted_times[mid]) / 2
            else:
                metrics.reacquire_time_median = sorted_times[mid]
                
        # 2. Track Clarity
        track_ages = []
        for f in frames:
            tracks = f.get('tracks', [])
            if isinstance(tracks, list):
                for tr in tracks:
                     # 'age' might not be in all track dicts depending on log level, checking safety
                    if isinstance(tr, dict) and 'age' in tr:
                        track_ages.append(tr['age'])
        if track_ages:
            metrics.avg_track_age = sum(track_ages) / len(track_ages)
            max_expected_age = 300
            metrics.track_continuity_index = min(1.0, metrics.avg_track_age / max_expected_age)
            
        if angular_errors:
            metrics.angular_accuracy_mean_deg = sum(angular_errors) / len(angular_errors)
            metrics.angular_accuracy_max_deg = max(angular_errors)
            
        # 3. Phase 3 Polish: Locked vs Valid
        metrics.locked_time_total = total_locked_time
        metrics.valid_lock_time_total = total_valid_locked_time
        metrics.valid_lock_count = valid_locks_triggered
        metrics.invalid_lock_time_total = total_locked_time - total_valid_locked_time
        
        if metrics.duration > 0:
            metrics.locked_ratio = total_locked_time / metrics.duration
            metrics.valid_lock_ratio = total_valid_locked_time / metrics.duration
            
        if valid_lock_durations:
            metrics.valid_longest_continuous_lock = max(valid_lock_durations)
            
        return metrics
    
    @classmethod
    def from_config(cls, config: Dict) -> 'RubricCalculator':
        """Create calculator from config dict."""
        return cls(
            camera_fov_deg=config.get('camera', {}).get('fov', 60.0),
            camera_resolution=tuple(config.get('camera', {}).get('resolution', [640, 480])),
        )


def calculate_rubric(frames: List[Dict], config: Dict = None) -> RubricMetrics:
    """Convenience function to calculate rubric metrics."""
    if config:
        calc = RubricCalculator.from_config(config)
    else:
        calc = RubricCalculator()
    return calc.calculate(frames)
