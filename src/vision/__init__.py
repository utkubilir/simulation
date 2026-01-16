"""
Görüntü işleme modülü
"""

from .detector import UAVDetector
from .tracker import TargetTracker
from .lock_on import LockOnSystem
from .model_loader import ModelLoader

# Optional Kalman tracker (requires filterpy)
try:
    from .kalman_tracker import KalmanTracker, KalmanTrack
    KALMAN_AVAILABLE = True
except ImportError:
    KalmanTracker = None
    KalmanTrack = None
    KALMAN_AVAILABLE = False

__all__ = ['UAVDetector', 'TargetTracker', 'LockOnSystem', 'ModelLoader', 
           'KalmanTracker', 'KalmanTrack', 'KALMAN_AVAILABLE']
