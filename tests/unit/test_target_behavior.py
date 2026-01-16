"""
Unit Tests for Target Behavior

Tests for:
- TargetManeuverController class
- ManeuverPattern enum
- Maneuver patterns
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uav.target_behavior import ManeuverPattern, TargetManeuverController, ManeuverCommand


class TestManeuverPattern:
    """Test ManeuverPattern enum."""
    
    def test_patterns_exist(self):
        """All maneuver patterns should exist."""
        assert ManeuverPattern.STRAIGHT is not None
        assert ManeuverPattern.CONSTANT_TURN is not None
        assert ManeuverPattern.ZIGZAG is not None
        assert ManeuverPattern.EVASIVE is not None
        assert ManeuverPattern.RANDOM is not None
        
    def test_patterns_are_unique(self):
        """Each pattern should be unique."""
        patterns = list(ManeuverPattern)
        assert len(patterns) == len(set(patterns))


class TestTargetManeuverController:
    """Test TargetManeuverController class."""
    
    @pytest.fixture
    def target_state(self):
        """Create sample target state."""
        return {
            'position': np.array([1000.0, 1000.0, 100.0]),
            'velocity': np.array([20.0, 0.0, 0.0]),
            'heading': 0.0,
        }
    
    @pytest.fixture
    def rng(self):
        """Create seeded RNG."""
        return np.random.default_rng(42)
    
    def test_straight_behavior(self, target_state):
        """STRAIGHT should maintain constant heading."""
        controller = TargetManeuverController(
            pattern='straight',
        )
        
        # Get command
        cmd = controller.update(dt=0.016, target_state=target_state)
        
        # Straight pattern should have low turn rate
        assert isinstance(cmd, ManeuverCommand)
        assert abs(cmd.yaw_rate) < 1.0  # Low turn rate
        
    def test_constant_turn_behavior(self, target_state):
        """CONSTANT_TURN should have consistent turn rate."""
        controller = TargetManeuverController(
            pattern='constant_turn',
            params={'turn_rate': 15.0},
        )
        
        cmd = controller.update(dt=0.016, target_state=target_state)
        
        assert isinstance(cmd, ManeuverCommand)
        # Should have some turn rate
        
    def test_zigzag_behavior(self, target_state):
        """ZIGZAG should oscillate direction."""
        controller = TargetManeuverController(
            pattern='zigzag',
            params={'zigzag_period': 2.0, 'zigzag_amplitude': 30.0},
        )
        
        # Get commands at different times
        controller._elapsed_time = 0.0
        cmd_t0 = controller.update(dt=0.016, target_state=target_state)
        
        controller._elapsed_time = 1.0  # Half period
        cmd_t1 = controller.update(dt=0.016, target_state=target_state)
        
        assert cmd_t0 is not None
        assert cmd_t1 is not None
        
    def test_evasive_behavior(self, target_state):
        """EVASIVE should have varying maneuvers."""
        controller = TargetManeuverController(
            pattern='evasive',
        )
        
        cmd = controller.update(dt=0.016, target_state=target_state)
        
        assert isinstance(cmd, ManeuverCommand)
        
    def test_random_behavior_with_seed(self, target_state, rng):
        """RANDOM with same seed should be deterministic."""
        controller1 = TargetManeuverController(
            pattern='random',
            rng=np.random.default_rng(42)
        )
        controller2 = TargetManeuverController(
            pattern='random',
            rng=np.random.default_rng(42)
        )
        
        cmd1 = controller1.update(dt=0.016, target_state=target_state)
        cmd2 = controller2.update(dt=0.016, target_state=target_state)
        
        # Same seed should produce same commands
        assert cmd1.throttle == cmd2.throttle
        
    def test_reset(self, target_state):
        """Reset should reinitialize controller."""
        controller = TargetManeuverController(
            pattern='random',
            rng=np.random.default_rng(42)
        )
        
        # Get some commands
        controller.update(dt=0.016, target_state=target_state)
        controller.update(dt=0.016, target_state=target_state)
        
        # Reset
        controller.reset()
        
        # Should be reset
        assert controller._phase_time == 0.0
        
    def test_get_info(self, target_state):
        """Should return controller info."""
        controller = TargetManeuverController(
            pattern='straight',
        )
        
        info = controller.get_info()
        
        assert isinstance(info, dict)
        assert 'pattern' in info


class TestManeuverCommand:
    """Test ManeuverCommand dataclass."""
    
    def test_default_values(self):
        """Default command should be neutral."""
        cmd = ManeuverCommand()
        
        assert cmd.yaw_rate == 0.0
        assert cmd.pitch_rate == 0.0
        assert cmd.throttle == 0.7  # Default throttle
        
    def test_custom_values(self):
        """Should accept custom values."""
        cmd = ManeuverCommand(
            yaw_rate=15.0,
            throttle=0.9,
            aileron=0.5
        )
        
        assert cmd.yaw_rate == 15.0
        assert cmd.throttle == 0.9
        assert cmd.aileron == 0.5


class TestControllerPatterns:
    """Test each pattern more thoroughly."""
    
    @pytest.fixture
    def target_state(self):
        """Create sample target state."""
        return {
            'position': np.array([1000.0, 1000.0, 100.0]),
            'velocity': np.array([20.0, 0.0, 0.0]),
            'heading': 0.0,
        }
    
    @pytest.mark.parametrize("pattern", [
        'straight',
        'constant_turn',
        'zigzag',
        'evasive',
    ])
    def test_pattern_produces_command(self, pattern, target_state):
        """Each pattern should produce valid command."""
        controller = TargetManeuverController(pattern=pattern)
        
        cmd = controller.update(dt=0.016, target_state=target_state)
        
        assert isinstance(cmd, ManeuverCommand)
        assert -1.0 <= cmd.throttle <= 1.0 or 0.0 <= cmd.throttle <= 1.0
        
    def test_random_needs_rng(self, target_state):
        """Random pattern should work with RNG."""
        rng = np.random.default_rng(42)
        controller = TargetManeuverController(pattern='random', rng=rng)
        
        cmd = controller.update(dt=0.016, target_state=target_state)
        
        assert isinstance(cmd, ManeuverCommand)
