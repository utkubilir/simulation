"""
Deterministic Delay Buffer - Phase 2

Provides buffered delay for vision pipeline outputs.
Uses frame-based (not wall-clock) delay for determinism.
"""

from collections import deque
from typing import TypeVar, Generic, Optional, List

T = TypeVar('T')


class DelayedStream(Generic[T]):
    """
    Deterministic delay buffer using frame-based timing.
    
    Delays items by a fixed number of frames (not wall-clock time).
    """
    
    def __init__(self, delay_frames: int):
        """
        Args:
            delay_frames: Number of frames to delay output.
                         0 means no delay (pass-through).
        """
        self.delay_frames = max(0, delay_frames)
        self._buffer: deque = deque()
        
    def push(self, item: T) -> Optional[T]:
        """
        Push new item and return delayed item if available.
        
        Args:
            item: New item to buffer
            
        Returns:
            Delayed item if buffer is full, else None
        """
        if self.delay_frames == 0:
            return item
            
        self._buffer.append(item)
        
        if len(self._buffer) > self.delay_frames:
            return self._buffer.popleft()
        return None
        
    def flush(self) -> List[T]:
        """Flush all remaining items from buffer."""
        items = list(self._buffer)
        self._buffer.clear()
        return items
        
    def reset(self):
        """Clear the buffer."""
        self._buffer.clear()
        
    @property
    def pending_count(self) -> int:
        """Number of items waiting in buffer."""
        return len(self._buffer)


class LatencyConfig:
    """Configuration for pipeline latency."""
    
    def __init__(
        self,
        detection_delay_ms: float = 0.0,
        tracking_delay_ms: float = 0.0,
        lock_delay_ms: float = 0.0,
    ):
        self.detection_delay_ms = detection_delay_ms
        self.tracking_delay_ms = tracking_delay_ms
        self.lock_delay_ms = lock_delay_ms
        
    @classmethod
    def from_dict(cls, d: dict) -> 'LatencyConfig':
        """Create from dict (scenario YAML)."""
        return cls(
            detection_delay_ms=d.get('detection_delay_ms', 0.0),
            tracking_delay_ms=d.get('tracking_delay_ms', 0.0),
            lock_delay_ms=d.get('lock_delay_ms', 0.0),
        )
        
    def to_dict(self) -> dict:
        """Export for logging."""
        return {
            'detection_delay_ms': self.detection_delay_ms,
            'tracking_delay_ms': self.tracking_delay_ms,
            'lock_delay_ms': self.lock_delay_ms,
        }
        
    def compute_delay_frames(self, perception_dt_ms: float) -> dict:
        """
        Compute delay in frames given perception tick interval.
        
        Args:
            perception_dt_ms: Milliseconds per perception tick
            
        Returns:
            Dict with frame counts for each delay type
        """
        if perception_dt_ms <= 0:
            perception_dt_ms = 1000.0 / 30.0  # Default 30fps
            
        return {
            'detection_delay_frames': int(round(self.detection_delay_ms / perception_dt_ms)),
            'tracking_delay_frames': int(round(self.tracking_delay_ms / perception_dt_ms)),
            'lock_delay_frames': int(round(self.lock_delay_ms / perception_dt_ms)),
        }
