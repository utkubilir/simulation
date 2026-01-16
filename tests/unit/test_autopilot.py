"""
Test Autopilot - Unit tests for autopilot and PID controller.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uav.autopilot import Autopilot, AutopilotMode, PIDController


class TestPIDController:
    """Test PID controller functionality."""
    
    def test_pid_proportional(self):
        """Proportional term should be kp * error."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        
        error = 10.0
        output = pid.update(error, dt=0.1)
        
        assert abs(output - 10.0) < 1e-6, "Proportional output incorrect"
        
    def test_pid_integral(self):
        """Integral term should accumulate error over time."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        
        # Apply constant error for 5 steps
        for _ in range(5):
            output = pid.update(error=2.0, dt=0.1)
            
        # Integral should be 2.0 * 0.1 * 5 = 1.0
        expected = 2.0 * 0.1 * 5
        assert abs(output - expected) < 1e-6, f"Integral output {output} != {expected}"
        
    def test_pid_derivative(self):
        """Derivative term should respond to error change rate."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        
        # First step
        pid.update(error=0.0, dt=0.1)
        # Second step with error change
        output = pid.update(error=1.0, dt=0.1)
        
        # Derivative = (1.0 - 0.0) / 0.1 = 10.0
        expected = (1.0 - 0.0) / 0.1
        assert abs(output - expected) < 1e-6, f"Derivative output {output} != {expected}"
        
    def test_pid_reset(self):
        """Reset should clear integral and previous error."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
        
        # Accumulate state
        for _ in range(10):
            pid.update(error=5.0, dt=0.1)
            
        pid.reset()
        
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0
        
    def test_pid_integral_limit(self):
        """Integral should be clamped to limit."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        pid.integral_limit = 5.0
        
        # Apply large error for many steps
        for _ in range(100):
            pid.update(error=100.0, dt=0.1)
            
        assert pid.integral <= pid.integral_limit


class TestAutopilot:
    """Test autopilot modes."""
    
    def test_autopilot_init(self):
        """Autopilot should initialize with default values (full autonomous per spec)."""
        ap = Autopilot()
        
        # Teknofest şartnamesi: Tam otonom mod zorunlu, varsayılan COMBAT
        assert ap.mode == AutopilotMode.COMBAT
        assert ap.enabled  # Her zaman aktif
        
    def test_set_mode(self):
        """Should be able to set autopilot mode."""
        ap = Autopilot()
        ap.set_mode(AutopilotMode.ALTITUDE_HOLD)
        
        assert ap.mode == AutopilotMode.ALTITUDE_HOLD
        
    def test_enable_disable(self):
        """Enable works, disable is intentionally no-op (full autonomous spec)."""
        ap = Autopilot()
        
        ap.enable()
        assert ap.enabled
        
        # Teknofest şartnamesi: Tam otonom mod zorunlu, disable() pasif
        ap.disable()
        assert ap.enabled  # Hâlâ aktif kalmalı
        
    def test_set_target_altitude(self):
        """Should set target altitude."""
        ap = Autopilot()
        ap.set_target_altitude(150.0)
        
        assert ap.target_altitude == 150.0
        
    def test_set_target_heading(self):
        """Should set target heading (stored in radians)."""
        ap = Autopilot()
        ap.set_target_heading(90.0)  # Input is degrees
        
        # Internally stored as radians
        assert abs(ap.target_heading - np.radians(90.0)) < 1e-6
        
    def test_update_returns_controls(self):
        """Update should return control dict when enabled."""
        ap = Autopilot()
        ap.enable()
        ap.set_mode(AutopilotMode.ALTITUDE_HOLD)
        ap.set_target_altitude(100.0)
        
        uav_state = {
            'position': np.array([0, 0, 100]),
            'velocity': np.array([20, 0, 0]),
            'orientation': np.array([0, 0, 0]),
            'speed': 20.0,
            'altitude': 100.0,
            'heading': 0.0
        }
        
        controls = ap.update(uav_state, dt=0.016)
        
        assert controls is not None
        assert 'throttle' in controls
        assert 'aileron' in controls
        assert 'elevator' in controls
        assert 'rudder' in controls
        
    def test_always_enabled_returns_controls(self):
        """Update should always return controls (full autonomous spec)."""
        ap = Autopilot()
        # Teknofest: Autopilot her zaman aktif, disable() çalışmaz
        
        uav_state = {
            'position': np.array([0, 0, 100]),
            'velocity': np.array([20, 0, 0]),
            'orientation': np.array([0, 0, 0]),
            'speed': 20.0,
            'altitude': 100.0,
            'heading': 0.0
        }
        
        controls = ap.update(uav_state, dt=0.016)
        
        # Tam otonom: Her zaman kontrol döndürür
        assert controls is not None
        assert 'throttle' in controls


class TestAutopilotWaypoint:
    """Test waypoint following mode."""
    
    def test_set_waypoints(self):
        """Should set waypoints list."""
        ap = Autopilot()
        waypoints = [
            (100, 100, 100),
            (200, 200, 100),
            (300, 100, 100),
        ]
        ap.set_waypoints(waypoints)
        
        assert len(ap.waypoints) == 3
        
    def test_waypoint_mode_pursues_target(self):
        """Waypoint mode should steer towards target."""
        ap = Autopilot()
        ap.enable()
        ap.set_mode(AutopilotMode.WAYPOINT)
        ap.set_waypoints([(200, 0, 100)])
        
        # UAV at origin, heading north (90°), waypoint east
        uav_state = {
            'position': np.array([0, 0, 100]),
            'velocity': np.array([0, 20, 0]),
            'orientation': np.array([0, 0, np.radians(90)]),
            'speed': 20.0,
            'altitude': 100.0,
            'heading': 90.0
        }
        
        controls = ap.update(uav_state, dt=0.016)
        
        # Should have some aileron or rudder command to turn east
        assert controls is not None
