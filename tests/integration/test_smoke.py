"""
Smoke Tests - Verify basic simulation functionality.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.simulation_core import SimulationCore, SimulationConfig
from src.core.frame_logger import FrameLogger
from src.core.metrics import MetricsCalculator


class TestSmoke:
    """Basic smoke tests for simulation."""
    
    def test_headless_run_completes(self):
        """Headless simulation should complete without errors."""
        config = SimulationConfig(
            seed=42,
            duration=2.0,  # Short run
            scenario="default"
        )
        
        sim = SimulationCore(config)
        states = sim.run()
        
        assert len(states) > 0, "No states produced"
        assert states[-1].t >= 1.9, "Simulation did not reach target duration"
        
    def test_output_artifacts_created(self, tmp_path):
        """All required output artifacts should be created."""
        config = SimulationConfig(seed=42, duration=2.0)
        sim = SimulationCore(config)
        states = sim.run()
        
        # Log frames
        frames_path = tmp_path / "frames.jsonl"
        with FrameLogger(frames_path) as logger:
            logger.log_frames(states)
            
        # Verify frames file exists and has content
        assert frames_path.exists()
        assert frames_path.stat().st_size > 0
        
        # Load and verify
        frames = FrameLogger.load_frames(frames_path)
        assert len(frames) == len(states)
        
        # Compute and save metrics
        metrics = MetricsCalculator.calculate(frames)
        
        metrics_json = tmp_path / "metrics.json"
        metrics_csv = tmp_path / "metrics.csv"
        
        MetricsCalculator.save_json(metrics, metrics_json)
        MetricsCalculator.save_csv(metrics, metrics_csv)
        
        assert metrics_json.exists()
        assert metrics_csv.exists()
        
    def test_frame_schema_valid(self):
        """Frame schema should have all required keys."""
        config = SimulationConfig(seed=42, duration=1.0)
        sim = SimulationCore(config)
        states = sim.run()
        
        # Check first frame
        state = states[0]
        
        assert hasattr(state, 't')
        assert hasattr(state, 'frame_id')
        assert hasattr(state, 'own_state')
        assert hasattr(state, 'targets')
        assert hasattr(state, 'detections')
        assert hasattr(state, 'tracks')
        assert hasattr(state, 'lock')
        assert hasattr(state, 'score')
        
        # Check lock schema
        assert 'locked' in state.lock
        assert 'target_id' in state.lock
        assert 'dx' in state.lock
        assert 'dy' in state.lock
        
    def test_metrics_values_reasonable(self):
        """Metrics values should be reasonable."""
        config = SimulationConfig(seed=42, duration=5.0)
        sim = SimulationCore(config)
        states = sim.run()
        
        # Convert to frames dict
        from dataclasses import asdict
        frames = [asdict(s) for s in states]
        
        metrics = MetricsCalculator.calculate(frames)
        
        assert metrics.total_frames > 0
        assert metrics.duration > 0
        assert 0 <= metrics.lock_ratio <= 1
        assert metrics.correct_locks >= 0
        assert metrics.incorrect_locks >= 0
        
    def test_own_uav_flies(self):
        """Own UAV should actually move during simulation."""
        config = SimulationConfig(seed=42, duration=3.0)
        sim = SimulationCore(config)
        states = sim.run()
        
        # Get first and last positions
        first_pos = states[0].own_state['position']
        last_pos = states[-1].own_state['position']
        
        # Calculate displacement
        import math
        displacement = math.sqrt(
            (last_pos[0] - first_pos[0])**2 +
            (last_pos[1] - first_pos[1])**2 +
            (last_pos[2] - first_pos[2])**2
        )
        
        # UAV should have moved at least somewhat (not stuck)
        assert displacement > 1.0, "UAV appears to be stationary"
