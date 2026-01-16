"""
Frame Logger and Metrics Aggregation (vNext)

Structured output logging for deterministic benchmarking:
- frames.jsonl: Per-frame simulation state
- metrics.json: Aggregated run metrics
- metrics.csv: Same metrics in tabular format
"""

import json
import csv
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib


@dataclass
class FrameData:
    """Single frame of simulation data"""
    t: float                            # Simulation time
    frame_id: int                       # Frame number
    own_state: Dict                     # Our UAV state
    enemies: List[Dict]                 # Enemy UAV states
    detections: List[Dict]              # Raw detections
    tracks: List[Dict]                  # Tracker output
    lock: Dict                          # Lock state
    score: Dict                         # Current score


@dataclass
class RunMetadata:
    """Run metadata"""
    run_id: str
    seed: int
    scenario: str
    start_time: str                     # ISO format, for reference only
    config_hash: str                    # Hash of config for reproducibility check


class FrameLogger:
    """
    Structured output logger.
    
    Writes to:
    - results/<run_id>/frames.jsonl
    - results/<run_id>/meta.json
    - results/<run_id>/config_snapshot.yaml
    """
    
    def __init__(self, output_dir: Path, run_id: str, seed: int, 
                 scenario: str, config: Dict):
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = run_id
        self.seed = seed
        self.scenario = scenario
        self.config = config
        
        # Frame storage
        self._frames: List[FrameData] = []
        self._frame_file = None
        
        # Write config snapshot
        self._write_config_snapshot()
        
        # Open frames file for streaming writes
        self._frame_file = open(self.output_dir / "frames.jsonl", "w")
        
    def _write_config_snapshot(self):
        """Write config snapshot"""
        config_path = self.output_dir / "config_snapshot.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)
            
        # Compute hash
        config_str = yaml.safe_dump(self.config, default_flow_style=False)
        self._config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        # Write metadata
        meta = RunMetadata(
            run_id=self.run_id,
            seed=self.seed,
            scenario=self.scenario,
            start_time=datetime.now().isoformat(),
            config_hash=self._config_hash
        )
        
        meta_path = self.output_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2)
            
    def log_frame(self, frame: FrameData):
        """Log a single frame"""
        self._frames.append(frame)
        
        # Stream write to file
        frame_dict = asdict(frame)
        self._frame_file.write(json.dumps(frame_dict) + "\n")
        self._frame_file.flush()
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure file is closed"""
        if self._frame_file and not self._frame_file.closed:
            self._frame_file.close()
        return False  # Don't suppress exceptions
        
    def finalize(self) -> 'Metrics':
        """Finalize logging and compute metrics"""
        if self._frame_file:
            self._frame_file.close()
            self._frame_file = None
            
        # Compute metrics
        metrics = Metrics.from_frames(self._frames)
        
        # Write metrics.json
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
            
        # Write metrics.csv
        csv_path = self.output_dir / "metrics.csv"
        metrics.to_csv(csv_path)
        
        # Compute frames hash for determinism check
        self._compute_frames_hash()
        
        return metrics
        
    def _compute_frames_hash(self):
        """Compute hash of frames.jsonl for determinism verification"""
        frames_path = self.output_dir / "frames.jsonl"
        with open(frames_path, "rb") as f:
            content = f.read()
            
        frames_hash = hashlib.sha256(content).hexdigest()
        
        # Update meta.json with hash
        meta_path = self.output_dir / "meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta['frames_hash'] = frames_hash
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        return frames_hash
        
    def __del__(self):
        """Cleanup"""
        if self._frame_file and not self._frame_file.closed:
            self._frame_file.close()


class Metrics:
    """
    Aggregated run metrics.
    
    Computes:
    - Lock success metrics
    - Timing metrics
    - Stability metrics
    """
    
    def __init__(self):
        # Basic
        self.duration: float = 0.0
        self.total_frames: int = 0
        
        # Lock metrics
        self.correct_locks: int = 0
        self.incorrect_locks: int = 0
        self.final_score: int = 0
        self.success_lock_count: int = 0
        
        # Timing
        self.time_to_first_valid_lock: Optional[float] = None
        self.time_to_first_success_lock: Optional[float] = None
        self.lock_success_rate: float = 0.0     # per minute
        
        # Stability
        self.valid_lock_time_total: float = 0.0
        self.longest_continuous_lock: float = 0.0
        self.valid_lock_ratio: float = 0.0      # time with valid lock / duration
        
        # Diagnostics
        self.invalid_reason_counts: Dict[str, int] = {}
        
        # Detection
        self.total_detections: int = 0
        self.detection_rate: float = 0.0        # per second
        
    @classmethod
    def from_frames(cls, frames: List[FrameData]) -> 'Metrics':
        """Compute metrics from frame data"""
        m = cls()
        
        if not frames:
            return m
            
        m.total_frames = len(frames)
        m.duration = frames[-1].t if frames else 0.0
        
        # Aggregate from last frame's score
        last_frame = frames[-1]
        m.correct_locks = last_frame.score.get('correct_locks', 0)
        m.incorrect_locks = last_frame.score.get('incorrect_locks', 0)
        m.final_score = last_frame.score.get('total_score', 0)
        m.success_lock_count = m.correct_locks # Alias
        
        # Time to first X
        for frame in frames:
            lock = frame.lock
            if lock.get('is_valid') and m.time_to_first_valid_lock is None:
                m.time_to_first_valid_lock = frame.t
                
            if lock.get('state') == 'success' and m.time_to_first_success_lock is None:
                m.time_to_first_success_lock = frame.t
                
        # Lock success rate (per minute)
        if m.duration > 0:
            m.lock_success_rate = (m.correct_locks / m.duration) * 60
            
        # Valid lock time & ratio & reasons
        prev_t = 0.0
        current_lock = 0.0
        
        for frame in frames:
            dt = frame.t - prev_t
            lock = frame.lock
            
            if lock.get('is_valid'):
                m.valid_lock_time_total += dt
                # Stability tracking
                current_lock += dt
                m.longest_continuous_lock = max(m.longest_continuous_lock, current_lock)
            else:
                current_lock = 0.0
                # Invalid reason tracking
                reason = lock.get('reason_invalid')
                if reason:
                    m.invalid_reason_counts[reason] = m.invalid_reason_counts.get(reason, 0) + 1
                    
            prev_t = frame.t

        if m.duration > 0:
            m.valid_lock_ratio = m.valid_lock_time_total / m.duration
            
        # Detection metrics
        m.total_detections = sum(len(f.detections) for f in frames)
        m.detection_rate = m.total_detections / m.duration if m.duration > 0 else 0.0
        
        return m
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'duration': self.duration,
            'total_frames': self.total_frames,
            'correct_locks': self.correct_locks,
            'incorrect_locks': self.incorrect_locks,
            'final_score': self.final_score,
            'success_lock_count': self.success_lock_count,
            'time_to_first_valid_lock': self.time_to_first_valid_lock,
            'time_to_first_success_lock': self.time_to_first_success_lock,
            'valid_lock_time_total': self.valid_lock_time_total,
            'valid_lock_ratio': self.valid_lock_ratio,
            'longest_continuous_lock': self.longest_continuous_lock,
            'invalid_reason_counts': self.invalid_reason_counts,
            'total_detections': self.total_detections,
            'detection_rate': self.detection_rate
        }
        
    def to_csv(self, path: Path):
        """Write metrics to CSV"""
        data = self.to_dict()
        # Flatten dicts for CSV (like invalid_reason_counts)
        flat_data = data.copy()
        reasons = flat_data.pop('invalid_reason_counts', {})
        for r, count in reasons.items():
            flat_data[f"reason_{r}"] = count
            
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(flat_data.keys())
            writer.writerow(flat_data.values())
