"""
Pytest Configuration and Shared Fixtures

This module provides common fixtures used across all test categories:
- Unit tests
- Integration tests
- Regression tests
- Performance tests
"""

import pytest
import shutil
import tempfile
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Provide temp dir that cleans up after test."""
    d = tempfile.mkdtemp(prefix="sim_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def pygame_init():
    """Initialize pygame for UI tests (session scope for efficiency)."""
    import pygame
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    pygame.init()
    yield pygame
    pygame.quit()


# =============================================================================
# UAV FIXTURES
# =============================================================================

@pytest.fixture
def sample_uav():
    """Create a default FixedWingUAV for testing."""
    from src.uav.fixed_wing import FixedWingUAV
    
    uav = FixedWingUAV()
    # Set initial position
    uav.state.position = np.array([1000.0, 1000.0, 100.0])
    uav.state.heading = 0.0
    return uav


@pytest.fixture
def sample_uav_state() -> Dict[str, Any]:
    """Create a sample UAV state dict."""
    return {
        'position': np.array([1000.0, 1000.0, 100.0]),
        'velocity': np.array([20.0, 0.0, 0.0]),
        'orientation': np.array([0.0, 0.0, 0.0]),
        'heading': 0.0,
        'altitude': 100.0,
        'speed': 20.0,
        'throttle': 0.7,
        'battery': 1.0
    }


@pytest.fixture
def sample_controls() -> Dict[str, float]:
    """Create sample control inputs."""
    return {
        'throttle': 0.7,
        'aileron': 0.0,
        'elevator': 0.0,
        'rudder': 0.0
    }


# =============================================================================
# DETECTION & TRACKING FIXTURES
# =============================================================================

@pytest.fixture
def sample_detection() -> Dict[str, Any]:
    """Create a sample detection dict at frame center."""
    return {
        'bbox': (300, 220, 340, 260),
        'center': (320, 240),
        'confidence': 0.85,
        'world_id': 'target_1',
        'distance': 150.0
    }


@pytest.fixture
def sample_track():
    """Create a sample Track object."""
    from src.vision.tracker import Track
    
    return Track(
        id=1,
        bbox=(300, 220, 340, 260),
        center=(320, 240),
        confidence=0.9,
        is_confirmed=True
    )


@pytest.fixture
def tracker():
    """Create a default TargetTracker."""
    from src.vision.tracker import TargetTracker
    return TargetTracker()


@pytest.fixture
def kalman_tracker():
    """Create a KalmanTracker if available, else IoUTracker."""
    from src.vision.kalman_tracker import KalmanTracker
    return KalmanTracker()


# =============================================================================
# LOCK-ON FIXTURES
# =============================================================================

@pytest.fixture
def lock_config():
    """Create default LockConfig."""
    from src.vision.lock_on import LockConfig
    return LockConfig()


@pytest.fixture
def lock_sm(lock_config):
    """Create a default LockOnStateMachine."""
    from src.vision.lock_on import LockOnStateMachine
    return LockOnStateMachine(lock_config)


@pytest.fixture
def geometry_validator():
    """Create a GeometryValidator."""
    from src.vision.validator import GeometryValidator
    return GeometryValidator()


# =============================================================================
# AUTOPILOT FIXTURES
# =============================================================================

@pytest.fixture
def autopilot():
    """Create an Autopilot instance."""
    from src.uav.autopilot import Autopilot
    return Autopilot()


@pytest.fixture
def pid_controller():
    """Create a PID controller with default gains."""
    from src.uav.autopilot import PIDController
    return PIDController(kp=1.0, ki=0.1, kd=0.05)


# =============================================================================
# SIMULATION FIXTURES
# =============================================================================

@pytest.fixture
def simulation_config():
    """Create a default SimulationConfig."""
    from src.core.simulation_core import SimulationConfig
    return SimulationConfig(seed=42, duration=5.0)


@pytest.fixture
def headless_runner(temp_output_dir):
    """Create a headless SimulationRunner for integration tests."""
    from scripts.run import SimulationRunner, load_scenario
    
    scenario_config = load_scenario('easy_lock')
    config = {
        **scenario_config,
        'seed': 42,
        'scenario': 'easy_lock',
        'duration': 3.0,
        'output_dir': str(temp_output_dir),
        'run_id': 'test_run'
    }
    return SimulationRunner(config, mode='headless')


# =============================================================================
# SCENARIO FIXTURES
# =============================================================================

@pytest.fixture
def scenarios_dir() -> Path:
    """Return path to scenarios directory."""
    return PROJECT_ROOT / 'scenarios'


@pytest.fixture
def easy_lock_config():
    """Load easy_lock scenario configuration."""
    from scripts.run import load_scenario
    return load_scenario('easy_lock')


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def assert_approx():
    """Helper for approximate float comparisons."""
    def _assert_approx(actual, expected, rel=1e-6, abs=1e-9):
        if isinstance(expected, (list, tuple, np.ndarray)):
            for a, e in zip(actual, expected):
                assert pytest.approx(e, rel=rel, abs=abs) == a
        else:
            assert pytest.approx(expected, rel=rel, abs=abs) == actual
    return _assert_approx


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location."""
    for item in items:
        # Auto-add markers based on path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "regression" in str(item.fspath):
            item.add_marker(pytest.mark.regression)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "compliance" in str(item.fspath):
            item.add_marker(pytest.mark.compliance)
