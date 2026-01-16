"""
Core simulation module - headless-capable simulation engine.
"""

from .simulation_core import SimulationCore
from .frame_logger import FrameLogger
from .metrics import MetricsCalculator
from .exceptions import (
    SimulationError,
    ScenarioError,
    ScenarioLoadError,
    ScenarioValidationError,
    ConfigurationError,
    VisionPipelineError,
    DetectionError,
    TrackingError,
    LockOnError,
    UAVError,
    CrashError,
    CompetitionError,
    InvalidLockError,
)

__all__ = [
    'SimulationCore',
    'FrameLogger', 
    'MetricsCalculator',
    # Exceptions
    'SimulationError',
    'ScenarioError',
    'ScenarioLoadError',
    'ScenarioValidationError',
    'ConfigurationError',
    'VisionPipelineError',
    'DetectionError',
    'TrackingError',
    'LockOnError',
    'UAVError',
    'CrashError',
    'CompetitionError',
    'InvalidLockError',
]

