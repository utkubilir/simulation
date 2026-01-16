"""
Unit Tests for Vision Components - Noise and Delay

Tests for:
- DetectionNoiseModel
- NoiseConfig
- DelayedStream
- LatencyConfig
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.noise import NoiseConfig, DetectionNoiseModel
from src.vision.delay import DelayedStream, LatencyConfig


class TestNoiseConfig:
    """Test NoiseConfig dataclass."""
    
    def test_default_values(self):
        """Default values should be zero/disabled."""
        config = NoiseConfig()
        
        assert config.bbox_sigma_px == 0.0
        assert config.conf_sigma == 0.0
        assert config.p_fn == 0.0
        assert config.p_fp == 0.0
        assert config.frame_width == 640
        assert config.frame_height == 480
        
    def test_from_dict(self):
        """Should create config from dictionary."""
        d = {
            'bbox_sigma_px': 5.0,
            'conf_sigma': 0.1,
            'p_fn': 0.05,
            'p_fp': 0.02,
            'frame_width': 1280,
            'frame_height': 720,
        }
        
        config = NoiseConfig.from_dict(d)
        
        assert config.bbox_sigma_px == 5.0
        assert config.conf_sigma == 0.1
        assert config.p_fn == 0.05
        assert config.p_fp == 0.02
        assert config.frame_width == 1280
        assert config.frame_height == 720
        
    def test_to_dict(self):
        """Should export to dictionary."""
        config = NoiseConfig(bbox_sigma_px=3.0, conf_sigma=0.05)
        
        d = config.to_dict()
        
        assert d['bbox_sigma_px'] == 3.0
        assert d['conf_sigma'] == 0.05
        assert 'p_fn' in d
        assert 'p_fp' in d
        
    def test_from_dict_with_missing_keys(self):
        """Should use defaults for missing keys."""
        d = {'bbox_sigma_px': 2.0}
        
        config = NoiseConfig.from_dict(d)
        
        assert config.bbox_sigma_px == 2.0
        assert config.conf_sigma == 0.0  # Default


class TestDetectionNoiseModel:
    """Test DetectionNoiseModel."""
    
    @pytest.fixture
    def rng(self):
        """Create seeded RNG for determinism."""
        return np.random.default_rng(42)
        
    @pytest.fixture
    def sample_detections(self):
        """Create sample detection list."""
        return [
            {
                'bbox': (100, 100, 150, 150),
                'confidence': 0.8,
                'center': (125, 125),
            },
            {
                'bbox': (200, 200, 250, 250),
                'confidence': 0.9,
                'center': (225, 225),
            },
        ]
        
    def test_no_noise_passthrough(self, rng, sample_detections):
        """With zero noise, detections should pass through unchanged."""
        config = NoiseConfig()  # All zeros
        model = DetectionNoiseModel(config, rng)
        
        result = model.apply(sample_detections)
        
        assert len(result) == 2
        assert result[0]['bbox'] == (100, 100, 150, 150)
        assert result[0]['confidence'] == 0.8
        
    def test_bbox_noise_applied(self, rng, sample_detections):
        """Bbox noise should modify bounding boxes."""
        config = NoiseConfig(bbox_sigma_px=10.0)
        model = DetectionNoiseModel(config, rng)
        
        result = model.apply(sample_detections)
        
        # Bbox should be modified
        assert result[0]['bbox'] != (100, 100, 150, 150)
        
        # But should still be valid (x1 < x2, y1 < y2)
        x1, y1, x2, y2 = result[0]['bbox']
        assert x1 < x2
        assert y1 < y2
        
    def test_confidence_noise_applied(self, rng, sample_detections):
        """Confidence noise should modify confidence values."""
        config = NoiseConfig(conf_sigma=0.1)
        model = DetectionNoiseModel(config, rng)
        
        result = model.apply(sample_detections)
        
        # Confidence should be modified
        assert result[0]['confidence'] != 0.8
        
        # But should be clamped to [0, 1]
        assert 0.0 <= result[0]['confidence'] <= 1.0
        
    def test_false_negatives(self, rng, sample_detections):
        """False negatives should drop detections."""
        config = NoiseConfig(p_fn=1.0)  # Drop all
        model = DetectionNoiseModel(config, rng)
        
        result = model.apply(sample_detections)
        
        assert len(result) == 0
        
    def test_false_positives(self, rng):
        """False positives should add spurious detections."""
        config = NoiseConfig(p_fp=1.0)  # Always add FP
        model = DetectionNoiseModel(config, rng)
        
        result = model.apply([])  # No input detections
        
        assert len(result) >= 1
        assert result[0].get('is_false_positive', False)
        
    def test_determinism(self, sample_detections):
        """Same seed should produce identical results."""
        config = NoiseConfig(bbox_sigma_px=5.0, conf_sigma=0.1)
        
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        model1 = DetectionNoiseModel(config, rng1)
        model2 = DetectionNoiseModel(config, rng2)
        
        result1 = model1.apply(sample_detections.copy())
        result2 = model2.apply(sample_detections.copy())
        
        assert result1[0]['bbox'] == result2[0]['bbox']
        assert result1[0]['confidence'] == result2[0]['confidence']
        
    def test_empty_input(self, rng):
        """Empty input with no FP should return empty."""
        config = NoiseConfig()
        model = DetectionNoiseModel(config, rng)
        
        result = model.apply([])
        
        assert result == []
        
    def test_get_config_dict(self, rng):
        """Should return config as dict."""
        config = NoiseConfig(bbox_sigma_px=3.0)
        model = DetectionNoiseModel(config, rng)
        
        d = model.get_config_dict()
        
        assert d['bbox_sigma_px'] == 3.0


class TestDelayedStream:
    """Test DelayedStream delay buffer."""
    
    def test_zero_delay_passthrough(self):
        """Zero delay should immediately return pushed item."""
        stream = DelayedStream[int](delay_frames=0)
        
        result = stream.push(42)
        
        assert result == 42
        
    def test_delay_one_frame(self):
        """Delay of 1 should return previous item."""
        stream = DelayedStream[int](delay_frames=1)
        
        # First push - nothing returned
        result1 = stream.push(1)
        assert result1 is None
        
        # Second push - first item returned
        result2 = stream.push(2)
        assert result2 == 1
        
        # Third push - second item returned
        result3 = stream.push(3)
        assert result3 == 2
        
    def test_delay_multiple_frames(self):
        """Delay of N should return item from N frames ago."""
        delay = 3
        stream = DelayedStream[int](delay_frames=delay)
        
        # First 3 pushes return None
        for i in range(delay):
            result = stream.push(i)
            assert result is None
            
        # 4th push returns first item
        result = stream.push(99)
        assert result == 0
        
    def test_flush(self):
        """Flush should return all buffered items."""
        stream = DelayedStream[int](delay_frames=3)
        
        stream.push(1)
        stream.push(2)
        stream.push(3)
        
        flushed = stream.flush()
        
        assert flushed == [1, 2, 3]
        assert stream.pending_count == 0
        
    def test_reset(self):
        """Reset should clear buffer."""
        stream = DelayedStream[int](delay_frames=3)
        
        stream.push(1)
        stream.push(2)
        
        stream.reset()
        
        assert stream.pending_count == 0
        
    def test_pending_count(self):
        """Pending count should track buffer size."""
        stream = DelayedStream[int](delay_frames=5)
        
        assert stream.pending_count == 0
        
        stream.push(1)
        assert stream.pending_count == 1
        
        stream.push(2)
        assert stream.pending_count == 2
        
    def test_negative_delay_treated_as_zero(self):
        """Negative delay should be treated as zero."""
        stream = DelayedStream[int](delay_frames=-5)
        
        result = stream.push(42)
        
        assert result == 42


class TestLatencyConfig:
    """Test LatencyConfig."""
    
    def test_default_values(self):
        """Default values should be zero."""
        config = LatencyConfig()
        
        assert config.detection_delay_ms == 0.0
        assert config.tracking_delay_ms == 0.0
        assert config.lock_delay_ms == 0.0
        
    def test_from_dict(self):
        """Should create from dictionary."""
        d = {
            'detection_delay_ms': 50.0,
            'tracking_delay_ms': 30.0,
            'lock_delay_ms': 10.0,
        }
        
        config = LatencyConfig.from_dict(d)
        
        assert config.detection_delay_ms == 50.0
        assert config.tracking_delay_ms == 30.0
        assert config.lock_delay_ms == 10.0
        
    def test_to_dict(self):
        """Should export to dictionary."""
        config = LatencyConfig(detection_delay_ms=100.0)
        
        d = config.to_dict()
        
        assert d['detection_delay_ms'] == 100.0
        assert 'tracking_delay_ms' in d
        
    def test_compute_delay_frames(self):
        """Should convert ms to frame count."""
        config = LatencyConfig(
            detection_delay_ms=100.0,  # 100ms
            tracking_delay_ms=50.0,    # 50ms
        )
        
        # At 30fps, each frame is ~33.33ms
        perception_dt_ms = 1000.0 / 30.0
        
        frames = config.compute_delay_frames(perception_dt_ms)
        
        # 100ms / 33.33ms ≈ 3 frames
        assert frames['detection_delay_frames'] == 3
        # 50ms / 33.33ms ≈ 2 frames
        assert frames['tracking_delay_frames'] == 2
        
    def test_compute_delay_frames_zero_dt(self):
        """Zero perception dt should use default 30fps."""
        config = LatencyConfig(detection_delay_ms=33.33)
        
        frames = config.compute_delay_frames(0.0)
        
        assert frames['detection_delay_frames'] == 1
