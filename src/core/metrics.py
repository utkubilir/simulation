"""
Metrics Calculator - Compute lock-on quality metrics.

Produces summary statistics for evaluation.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class MetricsSummary:
    """Summary metrics for a simulation run."""
    # Basic info
    total_frames: int
    duration: float
    
    # Lock metrics
    time_to_first_lock: Optional[float]  # Time when first valid lock achieved
    total_lock_time: float               # Total time in locked state
    lock_ratio: float                    # lock_time / duration
    correct_locks: int
    incorrect_locks: int
    
    # Track metrics
    total_detections: int
    total_tracks: int
    
    # Final score
    final_score: int


class MetricsCalculator:
    """
    Calculate metrics from simulation frames.
    """
    
    @staticmethod
    def calculate(frames: List[Dict]) -> MetricsSummary:
        """
        Calculate metrics from frame list.
        
        Args:
            frames: List of frame dicts (from JSONL)
            
        Returns:
            MetricsSummary with computed metrics
        """
        if not frames:
            return MetricsSummary(
                total_frames=0,
                duration=0.0,
                time_to_first_lock=None,
                total_lock_time=0.0,
                lock_ratio=0.0,
                correct_locks=0,
                incorrect_locks=0,
                total_detections=0,
                total_tracks=0,
                final_score=0
            )
            
        total_frames = len(frames)
        duration = frames[-1].get('t', 0.0) if frames else 0.0
        
        # Lock analysis
        time_to_first_lock = None
        total_lock_time = 0.0
        last_t = 0.0
        
        total_detections = 0
        total_tracks = 0
        
        for frame in frames:
            t = frame.get('t', 0.0)
            dt = t - last_t if last_t > 0 else 0.0
            last_t = t
            
            # Lock state
            lock = frame.get('lock', {})
            if lock.get('valid', False):
                if time_to_first_lock is None:
                    time_to_first_lock = t
                    
            if lock.get('locked', False):
                total_lock_time += dt
                
            # Counts
            total_detections += len(frame.get('detections', []))
            total_tracks += len(frame.get('tracks', []))
            
        # Final score from last frame
        final_frame = frames[-1] if frames else {}
        score = final_frame.get('score', {})
        correct_locks = score.get('correct_locks', 0)
        incorrect_locks = score.get('incorrect_locks', 0)
        final_score = score.get('total', 0)
        
        # Lock ratio
        lock_ratio = total_lock_time / duration if duration > 0 else 0.0
        
        return MetricsSummary(
            total_frames=total_frames,
            duration=duration,
            time_to_first_lock=time_to_first_lock,
            total_lock_time=total_lock_time,
            lock_ratio=lock_ratio,
            correct_locks=correct_locks,
            incorrect_locks=incorrect_locks,
            total_detections=total_detections,
            total_tracks=total_tracks,
            final_score=final_score
        )
        
    @staticmethod
    def to_dict(metrics: MetricsSummary) -> Dict:
        """Convert metrics to dict."""
        return {
            'total_frames': metrics.total_frames,
            'duration': metrics.duration,
            'time_to_first_lock': metrics.time_to_first_lock,
            'total_lock_time': metrics.total_lock_time,
            'lock_ratio': metrics.lock_ratio,
            'correct_locks': metrics.correct_locks,
            'incorrect_locks': metrics.incorrect_locks,
            'total_detections': metrics.total_detections,
            'total_tracks': metrics.total_tracks,
            'final_score': metrics.final_score
        }
        
    @staticmethod
    def save_json(metrics: MetricsSummary, path: Path):
        """Save metrics as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(MetricsCalculator.to_dict(metrics), f, indent=2)
            
    @staticmethod
    def save_csv(metrics: MetricsSummary, path: Path):
        """Save metrics as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = MetricsCalculator.to_dict(metrics)
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data.keys())
            writer.writerow(data.values())
            
    @staticmethod
    def load_json(path: Path) -> Dict:
        """Load metrics from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
