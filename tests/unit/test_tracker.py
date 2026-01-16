"""
Test Tracker - Unit tests for target tracking system.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.tracker import TargetTracker, Track


class TestTrack:
    """Test Track dataclass."""
    
    def test_track_creation(self):
        """Track should be created with required fields."""
        track = Track(
            id=1,
            bbox=(100, 100, 200, 200),
            center=(150, 150),
            confidence=0.9
        )
        
        assert track.id == 1
        assert track.bbox == (100, 100, 200, 200)
        assert track.center == (150, 150)
        assert track.confidence == 0.9
        assert track.age == 0
        assert track.hits == 1
        assert track.misses == 0
        assert not track.is_confirmed
        

class TestTargetTracker:
    """Test TargetTracker class."""
    
    def test_initialization_default(self):
        """Tracker should initialize with default config."""
        tracker = TargetTracker()
        
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert len(tracker.tracks) == 0
        
    def test_initialization_custom_config(self):
        """Tracker should respect custom config."""
        config = {'max_age': 50, 'min_hits': 5, 'iou_threshold': 0.5}
        tracker = TargetTracker(config)
        
        assert tracker.max_age == 50
        assert tracker.min_hits == 5
        assert tracker.iou_threshold == 0.5
        
    def test_update_creates_tracks(self):
        """Update with detections should create new tracks."""
        tracker = TargetTracker()
        
        detections = [
            {'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9},
            {'bbox': (200, 200, 250, 250), 'center': (225, 225), 'confidence': 0.8},
        ]
        
        tracks = tracker.update(detections)
        
        assert len(tracks) == 2
        assert len(tracker.tracks) == 2
        
    def test_update_matches_tracks(self):
        """Update should match detections to existing tracks."""
        tracker = TargetTracker()
        
        # First frame - create tracks
        det1 = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        tracker.update(det1)
        
        # Second frame - slightly moved detection should match
        det2 = [{'bbox': (105, 105, 155, 155), 'center': (130, 130), 'confidence': 0.85}]
        tracks = tracker.update(det2)
        
        # Should still be just 1 track (matched, not new)
        assert len(tracks) == 1
        assert tracks[0].hits == 2
        
    def test_update_no_detections(self):
        """Update with empty detections should increment misses."""
        tracker = TargetTracker()
        
        # Create a track
        det = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        tracker.update(det)
        
        # Empty update
        tracks = tracker.update([])
        
        assert len(tracks) == 1
        assert tracks[0].misses == 1
        
    def test_track_becomes_confirmed(self):
        """Track should become confirmed after min_hits."""
        config = {'min_hits': 3}
        tracker = TargetTracker(config)
        
        det = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        
        # Update 3 times
        for _ in range(3):
            tracks = tracker.update(det)
            
        assert len(tracks) == 1
        assert tracks[0].is_confirmed
        
    def test_track_removed_after_max_age(self):
        """Track should be removed after max_age misses."""
        config = {'max_age': 3}
        tracker = TargetTracker(config)
        
        # Create track
        det = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        tracker.update(det)
        
        # Miss for max_age + 1 frames
        for _ in range(4):
            tracker.update([])
            
        assert len(tracker.tracks) == 0
        
    def test_velocity_prediction(self):
        """Track should predict position using velocity."""
        tracker = TargetTracker()
        
        # First detection
        det1 = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        tracker.update(det1)
        
        # Second detection - moved right
        det2 = [{'bbox': (110, 100, 160, 150), 'center': (135, 125), 'confidence': 0.9}]
        tracker.update(det2)
        
        # Check velocity was computed
        track = list(tracker.tracks.values())[0]
        assert track.velocity[0] == pytest.approx(10.0, abs=0.1)  # Moved 10 pixels right
        assert track.velocity[1] == 0
        
    def test_get_confirmed_tracks(self):
        """get_confirmed_tracks should filter by is_confirmed."""
        config = {'min_hits': 2}
        tracker = TargetTracker(config)
        
        det = [
            {'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9},
            {'bbox': (200, 200, 250, 250), 'center': (225, 225), 'confidence': 0.8},
        ]
        
        # First update - neither confirmed
        tracker.update(det)
        assert len(tracker.get_confirmed_tracks()) == 0
        
        # Second update - both confirmed
        tracker.update(det)
        assert len(tracker.get_confirmed_tracks()) == 2
        
    def test_reset(self):
        """Reset should clear all tracks."""
        tracker = TargetTracker()
        
        det = [{'bbox': (100, 100, 150, 150), 'center': (125, 125), 'confidence': 0.9}]
        tracker.update(det)
        
        assert len(tracker.tracks) > 0
        
        tracker.reset()
        
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1


class TestVectorizedIoU:
    """Test the vectorized IoU computation."""
    
    def test_iou_identical_boxes(self):
        """IoU of identical boxes should be 1.0."""
        from src.vision.tracker import IoUTracker
        tracker = IoUTracker()
        
        boxes1 = np.array([[100, 100, 200, 200]])
        boxes2 = np.array([[100, 100, 200, 200]])
        
        iou = tracker._compute_iou_vectorized(boxes1, boxes2)
        
        assert abs(iou[0, 0] - 1.0) < 1e-6
        
    def test_iou_no_overlap(self):
        """IoU of non-overlapping boxes should be 0.0."""
        tracker = TargetTracker()
        
        boxes1 = np.array([[0, 0, 50, 50]])
        boxes2 = np.array([[100, 100, 150, 150]])
        
        iou = tracker._compute_iou_vectorized(boxes1, boxes2)
        
        assert iou[0, 0] == 0.0
        
    def test_iou_partial_overlap(self):
        """IoU of partially overlapping boxes should be between 0 and 1."""
        tracker = TargetTracker()
        
        boxes1 = np.array([[0, 0, 100, 100]])  # 100x100 = 10000
        boxes2 = np.array([[50, 50, 150, 150]])  # 100x100 = 10000
        
        # Overlap is 50x50 = 2500
        # Union is 10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        
        iou = tracker._compute_iou_vectorized(boxes1, boxes2)
        
        expected = 2500 / 17500
        assert abs(iou[0, 0] - expected) < 1e-6
        
    def test_iou_batch_computation(self):
        """Vectorized IoU should handle multiple boxes."""
        tracker = TargetTracker()
        
        boxes1 = np.array([
            [0, 0, 50, 50],
            [100, 100, 150, 150],
        ])
        boxes2 = np.array([
            [0, 0, 50, 50],      # Identical to boxes1[0]
            [100, 100, 150, 150],  # Identical to boxes1[1]
            [200, 200, 250, 250],  # No overlap with either
        ])
        
        iou = tracker._compute_iou_vectorized(boxes1, boxes2)
        
        assert iou.shape == (2, 3)
        assert abs(iou[0, 0] - 1.0) < 1e-6  # boxes1[0] == boxes2[0]
        assert abs(iou[1, 1] - 1.0) < 1e-6  # boxes1[1] == boxes2[1]
        assert iou[0, 2] == 0.0  # No overlap
        assert iou[1, 2] == 0.0  # No overlap
        
    def test_iou_consistency_with_scalar(self):
        """Vectorized IoU should match scalar IoU."""
        from src.vision.tracker import IoUTracker
        tracker = IoUTracker()
        
        b1 = (25, 30, 120, 180)
        b2 = (50, 60, 140, 200)
        
        # Scalar computation
        scalar_iou = tracker._compute_iou(b1, b2)
        
        # Vectorized computation
        boxes1 = np.array([[b1[0], b1[1], b1[2], b1[3]]])
        boxes2 = np.array([[b2[0], b2[1], b2[2], b2[3]]])
        vec_iou = tracker._compute_iou_vectorized(boxes1, boxes2)[0, 0]
        
        assert abs(scalar_iou - vec_iou) < 1e-6
