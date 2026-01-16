
import pytest
import numpy as np
from src.vision.lock_on import LockOnStateMachine, LockConfig, LockState
from src.vision.validator import GeometryValidator

class TestLockOnCompliance2025:
    """
    Verification suite for 2025 TEKNOFEST Savaşan İHA Rules
    """
    
    @pytest.fixture
    def lock_sm(self):
        config = LockConfig(
            window_seconds=5.0,
            required_continuous_seconds=4.0,
            size_threshold=0.06,  # Safe margin
            margin_horizontal=0.5,
            margin_vertical=0.5,
            frame_width=1000,
            frame_height=1000
        )
        return LockOnStateMachine(config)

    def test_size_threshold_clamping(self):
        """Rule: Do not use exactly 5.0% threshold."""
        # Test intentional exact 0.05
        config = LockConfig(size_threshold=0.05)
        sm = LockOnStateMachine(config)
        # Should auto-clamp to 0.06
        assert abs(sm.config.size_threshold - 0.06) < 1e-6, "Should clamp 0.05 to 0.06 safe margin"
        
        # Test explicit safe value
        config = LockConfig(size_threshold=0.07)
        sm = LockOnStateMachine(config)
        assert abs(sm.config.size_threshold - 0.07) < 1e-6, "Should keep explicit safe values"

    def test_overlay_thickness_limit(self):
        """Rule: Max line thickness = 3px."""
        assert GeometryValidator.validate_thickness(1) == 1
        assert GeometryValidator.validate_thickness(3) == 3
        assert GeometryValidator.validate_thickness(5) == 3  # Clamp down
        assert GeometryValidator.validate_thickness(0) == 1  # Clamp up

    def test_geometry_validation_size(self):
        """Rule: Target width OR height >= 5% of frame."""
        validator = GeometryValidator(size_threshold=0.06)
        
        # Frame 100x100
        # Box 4x4 (4%) -> Invalid
        res = validator.validate_lock_candidate((48, 48, 52, 52), 100, 100)
        assert not res.is_valid
        assert res.reason == GeometryValidator.REASON_SIZE_TOO_SMALL
        
        # Box 6x6 (6%) -> Valid
        res = validator.validate_lock_candidate((47, 47, 53, 53), 100, 100)
        assert res.is_valid
        
        # Box 2x10 (2% W, 10% H) -> Valid (OR logic)
        res = validator.validate_lock_candidate((49, 45, 51, 55), 100, 100)
        assert res.is_valid

    def test_geometry_validation_center(self):
        """Rule: Center offset <= half width/height."""
        validator = GeometryValidator(margin_h=0.5, margin_v=0.5)
        w, h = 100, 100
        frame_w, frame_h = 200, 200 # Center at 100,100
        
        # Box centered at 100,100 (dx=0, dy=0) -> Valid
        bbox = (50, 50, 150, 150) # 100x100 box
        res = validator.validate_lock_candidate(bbox, frame_w, frame_h)
        assert res.is_valid
        
        # Box centered at 150, 100 (dx=50) -> Valid (dx <= 100/2)
        # 150 is exactly edge of tolerance (half width=50)
        bbox = (100, 50, 200, 150) 
        res = validator.validate_lock_candidate(bbox, frame_w, frame_h)
        assert res.is_valid
        
        # Box centered at 151, 100 (dx=51) -> Invalid (51 > 50)
        bbox = (101, 50, 201, 150)
        res = validator.validate_lock_candidate(bbox, frame_w, frame_h)
        assert not res.is_valid
        assert res.reason == GeometryValidator.REASON_CENTER_OUTSIDE

    def test_lock_window_rule_success(self, lock_sm):
        """
        Rule: 4.0s continuous valid lock within 5.0s window.
        Scenario: Consistent valid lock from t=1.0 to t=5.0 (Duration 4.0s)
        """
        sim_time = 0.0
        dt = 0.1
        
        # Perfect track
        track = {'id': 1, 'center': (500, 500), 'bbox': (400, 400, 600, 600), 'confidence': 1.0}
        
        # Run for 3.9s -> No success
        for _ in range(39):
            sim_time += dt
            status = lock_sm.update([track], sim_time, dt)
            assert status.state != LockState.SUCCESS
            
        # Run 0.1s more -> 4.0s total -> SUCCESS
        sim_time += dt
        status = lock_sm.update([track], sim_time, dt)
        assert status.state == LockState.SUCCESS
        assert lock_sm._correct_locks == 1

    def test_lock_window_rule_broken_continuity(self, lock_sm):
        """
        Rule: Lock MUST be continuous.
        Scenario: 2.0s lock, 0.1s break, 2.0s lock.
        Result: No success at 4.1s (max continuous is 2.0s).
        """
        sim_time = 0.0
        dt = 0.1
        track = {'id': 1, 'center': (500, 500), 'bbox': (400, 400, 600, 600), 'confidence': 1.0}
        
        # 2.0s valid
        for _ in range(20):
            sim_time += dt
            lock_sm.update([track], sim_time, dt)
            
        # 0.1s invalid (empty tracks)
        sim_time += dt
        lock_sm.update([], sim_time, dt)
        
        # 2.0s valid again
        for _ in range(20):
            sim_time += dt
            status = lock_sm.update([track], sim_time, dt)
        
        # Total valid time is 4.0s, but not continuous
        assert status.state != LockState.SUCCESS
        assert lock_sm.current_continuous_duration < 2.1
        
        # Need 2.0s MORE to achieve success (segment restart)
        for _ in range(20):
            sim_time += dt
            status = lock_sm.update([track], sim_time, dt)
            
        assert status.state == LockState.SUCCESS

    def test_tolerance_window(self, lock_sm):
        """
        Rule: "4.0 seconds ... within any 5.0s window".
        Scenario: Lock starts at t=0.5, ends at t=4.5.
        At t=4.5, window [(-0.5), 4.5] contains 4.0s valid lock.
        """
        sim_time = 0.0
        dt = 0.1
        track = {'id': 1, 'center': (500, 500), 'bbox': (400, 400, 600, 600), 'confidence': 1.0}
        
        # Pre-lock delay
        sim_time = 0.5
        
        # Lock for 4.0s
        for _ in range(40):
            sim_time += dt
            status = lock_sm.update([track], sim_time, dt)
            
        assert status.state == LockState.SUCCESS
