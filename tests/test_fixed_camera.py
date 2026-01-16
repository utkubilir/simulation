"""
FixedCamera Unit Tests

Comprehensive tests for the main camera module including:
- Synthetic frame generation
- Occlusion detection
- Effect methods
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestFixedCameraBasics:
    """Basic FixedCamera functionality tests"""
    
    def setup_method(self):
        from src.simulation.camera import FixedCamera
        self.config = {
            'resolution': (640, 480),
            'fov': 60,
            'distortion_enabled': True,
            'shake_enabled': False,  # Disable for deterministic tests
        }
        self.camera = FixedCamera(position=[0, 0, -100], config=self.config)
    
    def test_initialization(self):
        """Camera initializes with correct parameters"""
        assert self.camera.resolution == (640, 480)
        assert self.camera.fov == 60
        assert self.camera.K is not None
        assert self.camera.K.shape == (3, 3)
    
    def test_update_intrinsics(self):
        """Intrinsic matrix updates correctly"""
        old_K = self.camera.K.copy()
        self.camera.set_fov(90)
        
        assert not np.allclose(old_K, self.camera.K)
        
    def test_set_resolution(self):
        """Resolution change updates intrinsics"""
        self.camera.set_resolution(1280, 720)
        
        assert self.camera.resolution == (1280, 720)
        assert self.camera.K[0, 2] == 640  # cx = width/2


class TestProjection:
    """Projection and coordinate transformation tests"""
    
    def setup_method(self):
        from src.simulation.camera import FixedCamera
        self.camera = FixedCamera(position=[0, 0, 0], config={
            'resolution': (640, 480),
            'fov': 60,
            'distortion_enabled': False,
        })
    
    def test_project_point_in_front(self):
        """Point in front of camera projects correctly"""
        camera_pos = np.array([0, 0, 0])
        camera_orient = np.array([0, 0, 0])
        world_point = np.array([100, 0, 0])  # 100m forward
        
        result = self.camera.project_point(world_point, camera_pos, camera_orient)
        
        assert result is not None
        x, y = result
        # Should be near center
        assert abs(x - 320) < 10
        assert abs(y - 240) < 10
    
    def test_project_point_behind_camera(self):
        """Point behind camera returns None"""
        camera_pos = np.array([0, 0, 0])
        camera_orient = np.array([0, 0, 0])
        world_point = np.array([-100, 0, 0])  # Behind
        
        result = self.camera.project_point(world_point, camera_pos, camera_orient)
        
        assert result is None
    
    def test_calculate_apparent_size(self):
        """Apparent size calculation is correct"""
        size_100m = self.camera.calculate_apparent_size(100, 2.0)
        size_50m = self.camera.calculate_apparent_size(50, 2.0)
        
        # Closer object should appear larger
        assert size_50m > size_100m
        assert size_50m == pytest.approx(size_100m * 2, rel=0.1)


class TestOcclusion:
    """Occlusion detection tests"""
    
    def setup_method(self):
        from src.simulation.camera import FixedCamera
        self.camera = FixedCamera(position=[0, 0, 0], config={'resolution': (640, 480)})
    
    def test_no_occlusion_empty_scene(self):
        """No occlusion with empty scene"""
        target_pos = np.array([100, 0, 0])
        camera_pos = np.array([0, 0, 0])
        
        is_occluded, occluder = self.camera.check_occlusion(
            target_pos, [], camera_pos, target_id='target'
        )
        
        assert not is_occluded
        assert occluder is None
    
    def test_occlusion_detected(self):
        """Occlusion is detected when object blocks target"""
        target_pos = np.array([100, 0, 0])
        camera_pos = np.array([0, 0, 0])
        blocker = {'id': 'blocker', 'position': [50, 0, 0], 'size': 5.0}
        
        is_occluded, occluder = self.camera.check_occlusion(
            target_pos, [blocker], camera_pos, target_id='target'
        )
        
        assert is_occluded
        assert occluder == 'blocker'
    
    def test_no_self_occlusion(self):
        """Target does not occlude itself"""
        target_pos = np.array([100, 0, 0])
        camera_pos = np.array([0, 0, 0])
        target = {'id': 'target', 'position': [100, 0, 0], 'size': 5.0}
        
        is_occluded, occluder = self.camera.check_occlusion(
            target_pos, [target], camera_pos, target_id='target'
        )
        
        assert not is_occluded


class TestSyntheticFrameGeneration:
    """Synthetic frame generation tests"""
    
    def setup_method(self):
        from src.simulation.camera import FixedCamera
        self.camera = FixedCamera(position=[0, 0, -100], config={
            'resolution': (320, 240),  # Small for speed
            'fov': 60,
            'shake_enabled': False,
            'motion_blur_enabled': False,
            'sensor_noise_enabled': False,  # Disable for determinism
        })
        # Disable GPU for testing
        self.camera.renderer = None
    
    def test_generates_valid_frame(self):
        """Generates frame with correct dimensions"""
        uav_states = [
            {'id': 'uav1', 'position': [100, 0, -100], 'heading': 0, 'size': 2.0}
        ]
        camera_pos = np.array([0, 0, -100])
        camera_orient = np.array([0, 0, 0])
        
        frame = self.camera.generate_synthetic_frame(
            uav_states, camera_pos, camera_orient
        )
        
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
    
    def test_empty_scene(self):
        """Handles empty scene"""
        camera_pos = np.array([0, 0, -100])
        camera_orient = np.array([0, 0, 0])
        
        frame = self.camera.generate_synthetic_frame(
            [], camera_pos, camera_orient
        )
        
        assert frame.shape == (240, 320, 3)


class TestEffects:
    """Camera effect method tests"""
    
    def setup_method(self):
        from src.simulation.camera import FixedCamera
        self.camera = FixedCamera(position=[0, 0, 0], config={
            'resolution': (320, 240),
        })
        self.test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    def test_sensor_noise_adds_variation(self):
        """Sensor noise adds variation to frame"""
        uniform = np.ones((240, 320, 3), dtype=np.uint8) * 128
        
        result = self.camera._apply_sensor_noise(uniform)
        
        assert result.shape == uniform.shape
        assert result.std() > 0  # Has variation
    
    def test_vignette_darkens_corners(self):
        """Vignette effect darkens corners"""
        bright = np.ones((240, 320, 3), dtype=np.uint8) * 200
        
        result = self.camera._apply_vignette(bright)
        
        center_val = result[120, 160].mean()
        corner_val = result[0, 0].mean()
        
        assert center_val > corner_val
    
    def test_tonemapping_preserves_range(self):
        """Tonemapping keeps values in valid range"""
        result = self.camera._apply_tonemapping(self.test_frame)
        
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.dtype == np.uint8
    
    def test_rolling_shutter_no_effect_when_static(self):
        """Rolling shutter has no effect without motion"""
        result = self.camera.apply_rolling_shutter(
            self.test_frame, 
            angular_velocity=np.array([0, 0, 0])
        )
        
        np.testing.assert_array_equal(result, self.test_frame)


class TestNoisePoolOptimization:
    """Tests for the precomputed noise pool optimization"""
    
    def setup_method(self):
        from src.simulation.camera import FixedCamera
        self.camera = FixedCamera(position=[0, 0, 0], config={
            'resolution': (320, 240),
            'iso': 100,
        })
    
    def test_noise_pool_initialized(self):
        """Noise pool is initialized on camera creation"""
        assert self.camera._noise_pool is not None
        assert self.camera._noise_pool.shape[0] == self.camera._noise_pool_size
    
    def test_noise_pool_cycles(self):
        """Noise pool index cycles correctly"""
        test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        initial_index = self.camera._noise_pool_index
        
        for _ in range(self.camera._noise_pool_size + 5):
            self.camera._apply_sensor_noise(test_frame)
        
        # Should have wrapped around
        assert self.camera._noise_pool_index < self.camera._noise_pool_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
