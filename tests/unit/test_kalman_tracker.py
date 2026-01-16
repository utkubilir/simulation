"""
Test Kalman Tracker - Unit tests for Kalman filter enhanced tracker.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if filterpy is available
try:
    from src.vision.kalman_tracker import (
        KalmanTracker, KalmanTrack, 
        create_kalman_filter, bbox_to_z, z_to_bbox,
        FILTERPY_AVAILABLE
    )
except ImportError:
    FILTERPY_AVAILABLE = False


# Skip all tests if filterpy is not installed
pytestmark = pytest.mark.skipif(
    not FILTERPY_AVAILABLE,
    reason="filterpy is required for Kalman tracker tests"
)


class TestBboxConversions:
    """Test bbox <-> z conversions."""
    
    def test_bbox_to_z(self):
        """Should convert bbox to Kalman measurement."""
        bbox = (100, 100, 200, 200)  # 100x100 box at (150, 150)
        z = bbox_to_z(bbox)
        
        assert z[0] == 150  # center x
        assert z[1] == 150  # center y
        assert z[2] == 10000  # area (100*100)
        assert z[3] == 1.0  # aspect ratio (100/100)
        
    def test_z_to_bbox(self):
        """Should convert Kalman state back to bbox."""
        z = np.array([150, 150, 10000, 1.0, 0, 0, 0])
        bbox = z_to_bbox(z)
        
        x1, y1, x2, y2 = bbox
        assert abs(x2 - x1 - 100) < 1  # width ~100
        assert abs(y2 - y1 - 100) < 1  # height ~100
        assert abs((x1 + x2) / 2 - 150) < 1  # center x
        assert abs((y1 + y2) / 2 - 150) < 1  # center y
        
    def test_roundtrip_conversion(self):
        """bbox -> z -> bbox should be approximately identity."""
        original = (50, 75, 150, 175)
        z = bbox_to_z(original)
        recovered = z_to_bbox(z)
        
        for orig, rec in zip(original, recovered):
            assert abs(orig - rec) < 1


class TestKalmanFilter:
    """Test Kalman filter creation."""
    
    def test_create_kalman_filter(self):
        """Should create properly configured Kalman filter."""
        kf = create_kalman_filter()
        
        assert kf.dim_x == 7  # State dimension
        assert kf.dim_z == 4  # Measurement dimension
        assert kf.F.shape == (7, 7)  # State transition
        assert kf.H.shape == (4, 7)  # Observation matrix
        
    def test_kalman_predict(self):
        """Kalman filter should predict next state."""
        kf = create_kalman_filter()
        
        # Initialize with some state
        kf.x = np.array([[100], [100], [1000], [1.0], [10], [5], [0]])
        
        kf.predict()
        
        # Position should have moved by velocity
        assert kf.x[0, 0] == 110  # x + vx
        assert kf.x[1, 0] == 105  # y + vy


class TestKalmanTracker:
    """Test KalmanTracker class."""
    
    def test_initialization(self):
        """Tracker should initialize with default config."""
        tracker = KalmanTracker()
        
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.tracks) == 0
        
    def test_update_creates_tracks(self):
        """Update should create new tracks from detections."""
        tracker = KalmanTracker()
        
        detections = [
            {'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9},
            {'bbox': (200, 200, 250, 250), 'center': (225, 225), 'confidence': 0.8},
        ]
        
        tracks = tracker.update(detections)
        
        assert len(tracks) == 2
        
    def test_update_matches_tracks(self):
        """Update should match detections to existing tracks."""
        tracker = KalmanTracker()
        
        det1 = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        tracker.update(det1)
        
        # Slightly moved detection
        det2 = [{'bbox': (105, 105, 155, 155), 'center': (130, 130), 'confidence': 0.85}]
        tracks = tracker.update(det2)
        
        assert len(tracks) == 1
        assert tracks[0].hits == 2
        
    def test_track_prediction(self):
        """Track should predict position when missing."""
        tracker = KalmanTracker()
        
        # Create track with consistent motion
        for i in range(5):
            det = [{'bbox': (100+i*10, 100, 150+i*10, 150), 
                    'center': (125+i*10, 125), 'confidence': 0.9}]
            tracker.update(det)
            
        # Get track position before missing
        track_before = list(tracker.tracks.values())[0]
        pos_before = track_before.center[0]
        
        # Miss a frame
        tracker.update([])
        
        # Track should have predicted forward motion
        track_after = list(tracker.tracks.values())[0]
        pos_after = track_after.center[0]
        
        assert pos_after > pos_before  # Should have moved forward
        
    def test_get_confirmed_tracks(self):
        """get_confirmed_tracks should filter by confirmation status."""
        config = {'min_hits': 2}
        tracker = KalmanTracker(config)
        
        det = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        
        tracker.update(det)
        assert len(tracker.get_confirmed_tracks()) == 0  # Not confirmed yet
        
        tracker.update(det)
        assert len(tracker.get_confirmed_tracks()) == 1  # Now confirmed
        
    def test_reset(self):
        """Reset should clear all tracks."""
        tracker = KalmanTracker()
        
        det = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        tracker.update(det)
        
        assert len(tracker.tracks) > 0
        
        tracker.reset()
        
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1


class TestKalmanTrack:
    """Test KalmanTrack dataclass."""
    
    def test_velocity_property(self):
        """velocity property should return Kalman state velocities."""
        kf = create_kalman_filter()
        kf.x = np.array([[100], [100], [1000], [1.0], [15], [10], [0]])
        
        track = KalmanTrack(
            id=1, kf=kf,
            bbox=(50, 50, 150, 150), center=(100, 100), confidence=0.9
        )
        
        vel = track.velocity
        assert vel[0] == 15  # vx
        assert vel[1] == 10  # vy
