"""
Custom Exception Classes for TEKNOFEST Savaşan İHA Simulation.

Provides hierarchical exception types for better error handling and debugging.
"""


class SimulationError(Exception):
    """Base exception for all simulation-related errors."""
    pass


class ScenarioError(SimulationError):
    """Errors related to scenario loading and validation."""
    pass


class ScenarioLoadError(ScenarioError):
    """Raised when a scenario YAML file cannot be loaded."""
    
    def __init__(self, scenario_name: str, message: str = None):
        self.scenario_name = scenario_name
        msg = f"Failed to load scenario '{scenario_name}'"
        if message:
            msg += f": {message}"
        super().__init__(msg)


class ScenarioValidationError(ScenarioError):
    """Raised when scenario validation fails."""
    
    def __init__(self, scenario_name: str, issues: list):
        self.scenario_name = scenario_name
        self.issues = issues
        msg = f"Scenario '{scenario_name}' validation failed:\n"
        msg += "\n".join(f"  - {issue}" for issue in issues)
        super().__init__(msg)


class ConfigurationError(SimulationError):
    """Errors related to configuration files."""
    pass


class VisionPipelineError(SimulationError):
    """Errors in the vision processing pipeline."""
    pass


class DetectionError(VisionPipelineError):
    """Raised when detection fails."""
    pass


class TrackingError(VisionPipelineError):
    """Raised when tracking fails."""
    pass


class LockOnError(VisionPipelineError):
    """Raised when lock-on system encounters an error."""
    pass


class UAVError(SimulationError):
    """Errors related to UAV physics or control."""
    pass


class CrashError(UAVError):
    """Raised when a UAV crashes."""
    
    def __init__(self, uav_id: str, reason: str = None):
        self.uav_id = uav_id
        msg = f"UAV '{uav_id}' crashed"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class CompetitionError(SimulationError):
    """Errors related to competition server simulation."""
    pass


class InvalidLockError(CompetitionError):
    """Raised when an invalid lock is reported."""
    
    def __init__(self, team_id: str, target_id: str, reason: str = None):
        self.team_id = team_id
        self.target_id = target_id
        msg = f"Invalid lock by team '{team_id}' on target '{target_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
