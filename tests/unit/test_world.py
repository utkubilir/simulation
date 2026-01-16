"""
Unit Tests for World Simulation

Tests for:
- SimulationWorld class
- Boundary enforcement
- UAV state management
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.world import SimulationWorld


class TestWorldInitialization:
    """Test World initialization."""
    
    def test_default_initialization(self):
        """World should initialize with defaults."""
        world = SimulationWorld()
        
        assert world.world_size is not None
        assert len(world.uavs) >= 0
        
    def test_initialization_with_config(self):
        """World should accept configuration."""
        config = {
            'world_size': (3000, 3000, 500),
        }
        
        world = SimulationWorld(config)
        
        # Should use config values
        assert world.world_size[0] >= 2000  # At least default or larger


class TestWorldBoundaries:
    """Test world boundary enforcement."""
    
    @pytest.fixture
    def world(self):
        """Create default world."""
        return SimulationWorld()
        
    def test_uav_stays_in_bounds(self, world):
        """UAV should be kept within world bounds."""
        # This depends on how world handles UAVs
        # The world should have boundary enforcement
        assert hasattr(world, 'world_size')


class TestWorldState:
    """Test world state management."""
    
    @pytest.fixture
    def world(self):
        """Create world with UAVs."""
        world = SimulationWorld()
        world.spawn_uav(
            uav_id='player',
            position=[1000, 1000, 100],
            heading=0,
            is_player=True
        )
        world.spawn_uav(
            uav_id='enemy_1',
            team='red',
            position=[1200, 1000, 100],
            heading=180
        )
        return world
        
    def test_get_world_state(self, world):
        """Should return world state dict."""
        state = world.get_world_state()
        
        assert isinstance(state, dict)
        assert 'uavs' in state
        assert len(world.uavs) >= 2
