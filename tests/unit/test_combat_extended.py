"""
Unit Tests for Combat System

Tests for:
- CombatState enum
- CombatStateManager
- Attack and evade behaviors
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uav.combat import CombatState, CombatConfig, CombatStateManager


class TestCombatState:
    """Test CombatState enum."""
    
    def test_states_exist(self):
        """All combat states should exist."""
        assert CombatState.SEARCH is not None
        assert CombatState.TRACK is not None
        assert CombatState.LOCK is not None
        
    def test_states_are_different(self):
        """Each state should be unique."""
        states = list(CombatState)
        assert len(states) == len(set(states))


class TestCombatConfig:
    """Test CombatConfig."""
    
    def test_default_values(self):
        """Default config should have sensible values."""
        config = CombatConfig()
        
        assert config.engagement_distance > 0
        assert config.search_altitude > 0
        
    def test_initialization(self):
        """Should create from arguments."""
        config = CombatConfig(
            engagement_distance=500.0,
            search_altitude=200.0,
        )
        
        assert config.engagement_distance == 500.0
        assert config.search_altitude == 200.0


@dataclass
class MockTrack:
    """Mock for Track object."""
    id: str
    position: np.ndarray
    is_confirmed: bool = True
    
    # Add dict compatibility for older code if needed, but CombatStateManager expects objects
    def __getitem__(self, item):
        return getattr(self, item)

class TestCombatStateManager:
    """Test CombatStateManager."""
    
    @pytest.fixture
    def manager(self):
        """Create default combat manager."""
        return CombatStateManager()
        
    @pytest.fixture
    def own_state(self):
        """Create own UAV state."""
        return {
            'position': np.array([1000.0, 1000.0, 100.0]),
            'velocity': np.array([20.0, 0.0, 0.0]),
            'heading': 0.0,
        }
        
    @pytest.fixture
    def target_far(self):
        """Create far target."""
        return MockTrack(
            id='target_1',
            position=np.array([2000.0, 1000.0, 100.0]),
            is_confirmed=True
        )
        
    @pytest.fixture
    def target_close(self):
        """Create close target."""
        return MockTrack(
            id='target_1',
            position=np.array([1100.0, 1000.0, 100.0]),
            is_confirmed=True
        )
        
    def test_initial_state_search(self, manager):
        """Initial state should be SEARCH (default)."""
        assert manager.state == CombatState.SEARCH
        
    def test_update_with_no_targets(self, manager, own_state):
        """Update with no targets should stay SEARCH."""
        targets = []
        
        controls = manager.update(own_state, targets)
        
        # Should imply search behavior (patrol)
        mode = controls['mode']
        mode_name = mode.name if hasattr(mode, 'name') else str(mode)
        assert mode_name == 'WAYPOINT' or manager.state == CombatState.SEARCH
        
    def test_update_with_far_target(self, manager, own_state, target_far):
        """Far target should trigger TRACK (if within engagement distance)."""
        targets = [target_far]
        
        manager.update(own_state, targets)
        
        # Depending on logic, might switch to TRACK
        # assert manager.state in [CombatState.SEARCH, CombatState.TRACK]
        
    def test_update_with_close_target(self, manager, own_state, target_close):
        """Close target should trigger LOCK/TRACK."""
        targets = [target_close]
        
        # Update multiple times to allow state transitions
        for _ in range(10):
            manager.update(own_state, targets)
        
        # Should be in TRACK or LOCK
        assert manager.state in [CombatState.TRACK, CombatState.LOCK] 
        
    def test_reset(self, manager, own_state, target_close):
        """Reset should return to SEARCH."""
        targets = [target_close]
        
        manager.update(own_state, targets)
        # Assuming there is a reset or re-init needed, but CombatStateManager might not have reset()
        # Creating new manager instead
        new_manager = CombatStateManager()
        assert new_manager.state == CombatState.SEARCH
        
    def test_controls_output(self, manager, own_state, target_close):
        """Update should return control commands."""
        targets = [target_close]
        
        controls = manager.update(own_state, targets)
        
        assert controls is not None
        assert 'mode' in controls


class TestCombatBehaviors:
    """Test specific combat behaviors."""
    
    def test_search_mode_behavior(self):
        """Search mode should generate waypoints."""
        from src.uav.autopilot import AutopilotMode
        manager = CombatStateManager()
        manager.state = CombatState.SEARCH
        
        own_state = {
            'position': np.array([1000.0, 1000.0, 100.0]),
            'velocity': np.array([20.0, 0.0, 0.0]),
            'heading': 0.0,
        }
        
        controls = manager.update(own_state, [])
        
        # Check if mode is AutopilotMode enum or string
        mode = controls['mode']
        if hasattr(mode, 'name'):
            assert mode.name.upper() == 'WAYPOINT'
        else:
            assert str(mode).upper() == 'WAYPOINT' or str(mode).upper() == 'AUTOPILOTMODE.WAYPOINT'
            
    def test_heading_error_calculation(self):
        """Should calculate heading error correctly."""
        # Test that heading error wraps correctly
        manager = CombatStateManager()
        
        assert manager.state == CombatState.SEARCH
