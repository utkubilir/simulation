"""
Test Fixed Wing UAV - Unit tests for UAV physics model.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uav.fixed_wing import FixedWingUAV, UAVState, ControlInputs


class TestUAVState:
    """Test UAV state dataclass."""
    
    def test_default_state(self):
        """Default state should have reasonable values."""
        state = UAVState()
        
        assert np.allclose(state.position, [0, 0, 0])
        assert np.allclose(state.velocity, [20.0, 0.0, 0.0])
        assert np.allclose(state.orientation, [0, 0, 0])
        assert np.allclose(state.angular_velocity, [0, 0, 0])


class TestControlInputs:
    """Test control inputs dataclass."""
    
    def test_default_controls(self):
        """Default controls should be neutral except throttle."""
        controls = ControlInputs()
        
        assert controls.aileron == 0.0
        assert controls.elevator == 0.0
        assert controls.rudder == 0.0
        assert controls.throttle == 0.5


class TestFixedWingUAV:
    """Test fixed wing UAV physics model."""
    
    def test_initialization(self):
        """UAV should initialize with correct defaults."""
        uav = FixedWingUAV()
        
        assert uav.id is not None
        assert uav.team == "blue"
        assert not uav.is_crashed
        
    def test_initialization_with_config(self):
        """UAV should respect config parameters."""
        config = {
            'mass': 10.0,
            'wingspan': 3.0,
            'min_speed': 20.0,
            'max_speed': 50.0,
        }
        uav = FixedWingUAV(config=config, uav_id='test_uav', team='red')
        
        assert uav.id == 'test_uav'
        assert uav.team == 'red'
        assert uav.mass == 10.0
        assert uav.wingspan == 3.0
        assert uav.min_speed == 20.0
        assert uav.max_speed == 50.0
        
    def test_reset(self):
        """Reset should set position and heading."""
        uav = FixedWingUAV()
        position = np.array([100, 200, 150])
        heading = np.radians(45)
        
        uav.reset(position=position, heading=heading)
        
        assert np.allclose(uav.get_position(), position)
        assert abs(uav.get_heading() - heading) < 1e-6
        
    def test_set_controls(self):
        """Setting controls should update control state."""
        uav = FixedWingUAV()
        
        uav.set_controls(aileron=0.5, elevator=-0.3, rudder=0.2, throttle=0.8)
        
        # Access internal controls
        assert uav.controls.aileron == 0.5
        assert uav.controls.elevator == -0.3
        assert uav.controls.rudder == 0.2
        assert uav.controls.throttle == 0.8
        
    def test_update_moves_uav(self):
        """Update should change UAV position over time."""
        uav = FixedWingUAV()
        uav.reset(position=np.array([0, 0, 100]), heading=0)
        
        initial_pos = uav.get_position().copy()
        
        # Run for 1 second
        for _ in range(60):
            uav.update(dt=1/60)
            
        final_pos = uav.get_position()
        
        # Should have moved forward
        displacement = np.linalg.norm(final_pos - initial_pos)
        assert displacement > 10, f"UAV only moved {displacement}m in 1 second"
        
    def test_altitude_maintained_level_flight(self):
        """UAV should roughly maintain altitude in level flight."""
        uav = FixedWingUAV()
        initial_altitude = 100.0
        uav.reset(position=np.array([0, 0, initial_altitude]), heading=0)
        uav.set_controls(throttle=0.7, elevator=0, aileron=0, rudder=0)
        
        # Run for 2 seconds
        for _ in range(120):
            uav.update(dt=1/60)
            
        final_altitude = uav.get_altitude()
        altitude_change = abs(final_altitude - initial_altitude)
        
        # Should not have changed dramatically (within 50m)
        assert altitude_change < 50, f"Altitude changed by {altitude_change}m"
        
    def test_speed_getters(self):
        """Speed getters should return correct values."""
        uav = FixedWingUAV()
        uav.reset(position=np.array([0, 0, 100]), heading=0)
        
        speed = uav.get_speed()
        altitude = uav.get_altitude()
        heading_rad = uav.get_heading()
        heading_deg = uav.get_heading_degrees()
        
        assert speed > 0
        assert altitude == 100.0
        assert abs(heading_rad) < 1e-6  # Should be 0
        assert abs(heading_deg) < 1e-3  # Should be 0 degrees
        
    def test_get_orientation(self):
        """Orientation getters should return consistent values."""
        uav = FixedWingUAV()
        uav.reset(position=np.array([0, 0, 100]), heading=np.radians(90))
        
        orient_rad = uav.get_orientation()
        orient_deg = uav.get_orientation_degrees()
        
        # Roll, pitch should be near zero; yaw should be 90 deg
        assert abs(orient_rad[2] - np.radians(90)) < 1e-6
        assert abs(orient_deg[2] - 90) < 1e-3
        
    def test_get_forward_vector(self):
        """Forward vector should point in heading direction."""
        uav = FixedWingUAV()
        
        # Heading 0 (East in typical coord system)
        uav.reset(position=np.array([0, 0, 100]), heading=0)
        forward = uav.get_forward_vector()
        
        # Should be pointing mostly in positive X direction
        assert forward[0] > 0.9
        
        # Heading 90 degrees (North)
        uav.reset(position=np.array([0, 0, 100]), heading=np.radians(90))
        forward = uav.get_forward_vector()
        
        # Should be pointing mostly in positive Y direction
        assert forward[1] > 0.9
        
    def test_get_camera_position(self):
        """Camera position should be near UAV position."""
        uav = FixedWingUAV()
        uav.reset(position=np.array([100, 200, 150]), heading=0)
        
        cam_pos = uav.get_camera_position()
        uav_pos = uav.get_position()
        
        # Camera should be close to UAV position (within wingspan distance)
        distance = np.linalg.norm(cam_pos - uav_pos)
        assert distance < uav.wingspan
        
    def test_to_dict(self):
        """to_dict should return complete state dictionary."""
        uav = FixedWingUAV(uav_id='test', team='red')
        uav.reset(position=np.array([100, 200, 150]), heading=np.radians(45))
        
        state_dict = uav.to_dict()
        
        assert state_dict['id'] == 'test'
        assert state_dict['team'] == 'red'
        assert 'position' in state_dict
        assert 'velocity' in state_dict
        assert 'speed' in state_dict
        assert 'altitude' in state_dict
        assert 'heading' in state_dict
        assert 'orientation' in state_dict
        assert 'is_crashed' in state_dict
        

class TestUAVPhysics:
    """Test UAV physics behavior."""
    
    def test_throttle_affects_speed(self):
        """Higher throttle should result in higher speed."""
        uav_high = FixedWingUAV()
        uav_low = FixedWingUAV()
        
        uav_high.reset(position=np.array([0, 0, 100]), heading=0)
        uav_low.reset(position=np.array([0, 0, 100]), heading=0)
        
        uav_high.set_controls(throttle=1.0)
        uav_low.set_controls(throttle=0.3)
        
        # Run for 3 seconds
        for _ in range(180):
            uav_high.update(dt=1/60)
            uav_low.update(dt=1/60)
            
        # High throttle UAV should be faster or have traveled further
        dist_high = np.linalg.norm(uav_high.get_position()[:2])
        dist_low = np.linalg.norm(uav_low.get_position()[:2])
        
        assert dist_high > dist_low, "High throttle should result in greater distance"
        
    def test_aileron_causes_roll(self):
        """Aileron input should cause roll."""
        uav = FixedWingUAV()
        uav.reset(position=np.array([0, 0, 100]), heading=0)
        
        uav.set_controls(aileron=0.5, throttle=0.7)
        
        # Run for 0.5 seconds
        for _ in range(30):
            uav.update(dt=1/60)
            
        # Should have some roll
        roll = uav.get_orientation()[0]  # roll is first component
        assert abs(roll) > 0.01, "Aileron should cause roll"
        
    def test_elevator_affects_pitch(self):
        """Elevator input should affect pitch."""
        uav = FixedWingUAV()
        uav.reset(position=np.array([0, 0, 100]), heading=0)
        
        uav.set_controls(elevator=-0.3, throttle=0.7)  # Pull up
        
        # Run for 0.5 seconds
        for _ in range(30):
            uav.update(dt=1/60)
            
        # Should have some pitch change
        pitch = uav.get_orientation()[1]  # pitch is second component
        # Pitch direction depends on convention, just check it changed
        assert abs(pitch) > 0.005 or uav.get_altitude() > 100, "Elevator should affect pitch/altitude"
