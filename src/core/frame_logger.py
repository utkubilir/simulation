"""
Frame Logger - JSONL frame-by-frame logging.

Produces standardized logs for post-analysis.
"""

import json
from pathlib import Path
from typing import List, Any
from dataclasses import asdict


class FrameLogger:
    """
    Logs simulation frames to JSONL format.
    
    Schema per line:
    {
        "t": float,
        "frame_id": int,
        "own_state": {...},
        "targets": [...],
        "detections": [...],
        "tracks": [...],
        "lock": {"locked": bool, "target_id": str|null, "dx": float|null, "dy": float|null},
        "score": {...}
    }
    """
    
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._frame_count = 0
        
    def __enter__(self):
        self._file = open(self.output_path, 'w', encoding='utf-8')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None
            
    def log_frame(self, state):
        """
        Log a single frame.
        
        Args:
            state: SimulationState dataclass instance
        """
        if self._file is None:
            raise RuntimeError("FrameLogger must be used as context manager")
            
        # Convert dataclass to dict
        if hasattr(state, '__dataclass_fields__'):
            data = asdict(state)
        else:
            data = state if isinstance(state, dict) else state.__dict__
            
        # Deterministic sorting of list fields
        if 'targets' in data and isinstance(data['targets'], list):
            data['targets'].sort(key=lambda x: x.get('id', ''))
            
        if 'detections' in data and isinstance(data['detections'], list):
            # Sort by class, then x1 coordinate (geometric stability)
            data['detections'].sort(key=lambda x: (
                x.get('class_id', 0), 
                x.get('bbox', [0,0,0,0])[0]
            ))
            
        if 'tracks' in data and isinstance(data['tracks'], list):
            data['tracks'].sort(key=lambda x: x.get('id', 0))
            
        # Write as JSON line (sort_keys=True for attribute order determinism)
        self._file.write(json.dumps(data, default=self._json_serializer, sort_keys=True) + '\n')
        self._frame_count += 1
        
    def log_frames(self, states: List):
        """Log multiple frames."""
        for state in states:
            self.log_frame(state)
            
    @property
    def frame_count(self) -> int:
        return self._frame_count
        
    @staticmethod
    def _json_serializer(obj):
        """Handle non-serializable types."""
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
        
    @staticmethod
    def load_frames(path: Path) -> List[dict]:
        """Load frames from JSONL file."""
        frames = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    frames.append(json.loads(line))
        return frames
