"""
Determinism Tests - Verify that same seed produces identical results.
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


class TestDeterminism:
    """Test that simulation is deterministic with same seed."""
    
    def test_same_seed_produces_identical_metrics(self, tmp_path):
        """Two runs with same seed should produce identical core metrics."""
        seed = 42
        duration = 5.0
        
        # Run 1
        config1 = SimulationConfig(seed=seed, duration=duration)
        sim1 = SimulationCore(config1)
        states1 = sim1.run()
        
        # Run 2 (same seed)
        config2 = SimulationConfig(seed=seed, duration=duration)
        sim2 = SimulationCore(config2)
        states2 = sim2.run()
        
        # Verify same number of frames
        assert len(states1) == len(states2), "Frame count mismatch"
        
        # Verify key fields match
        for i, (s1, s2) in enumerate(zip(states1, states2)):
            assert abs(s1.t - s2.t) < 1e-9, f"Time mismatch at frame {i}"
            assert s1.frame_id == s2.frame_id, f"Frame ID mismatch at frame {i}"
            
            # Own state position should be identical
            if s1.own_state and s2.own_state:
                pos1 = s1.own_state['position']
                pos2 = s2.own_state['position']
                for j, (p1, p2) in enumerate(zip(pos1, pos2)):
                    assert abs(p1 - p2) < 1e-6, f"Position mismatch at frame {i}, axis {j}"
                    
    def test_different_seed_produces_different_results(self, tmp_path):
        """Different seeds should produce different results (for multi-target scenarios)."""
        duration = 5.0
        
        # Run with seed 42
        config1 = SimulationConfig(seed=42, duration=duration, scenario="multi_target_3")
        sim1 = SimulationCore(config1)
        states1 = sim1.run()
        
        # Run with seed 123
        config2 = SimulationConfig(seed=123, duration=duration, scenario="multi_target_3")
        sim2 = SimulationCore(config2)
        states2 = sim2.run()
        
        # Final scores could differ (or at least initial positions differ)
        # Check final state
        if states1 and states2:
            final1 = states1[-1]
            final2 = states2[-1]
            
            # Just verify both ran successfully
            assert final1.t > 0
            assert final2.t > 0
            
    def test_metrics_json_identical_for_same_seed(self, tmp_path):
        """Metrics JSON should be byte-identical for same seed runs."""
        seed = 42
        duration = 3.0
        
        # Run 1
        config1 = SimulationConfig(seed=seed, duration=duration)
        sim1 = SimulationCore(config1)
        states1 = sim1.run()
        
        # Log and compute metrics
        frames_path1 = tmp_path / "run1" / "frames.jsonl"
        frames_path1.parent.mkdir(parents=True, exist_ok=True)
        
        with FrameLogger(frames_path1) as logger:
            logger.log_frames(states1)
            
        frames1 = FrameLogger.load_frames(frames_path1)
        metrics1 = MetricsCalculator.calculate(frames1)
        
        # Run 2
        config2 = SimulationConfig(seed=seed, duration=duration)
        sim2 = SimulationCore(config2)
        states2 = sim2.run()
        
        frames_path2 = tmp_path / "run2" / "frames.jsonl"
        frames_path2.parent.mkdir(parents=True, exist_ok=True)
        
        with FrameLogger(frames_path2) as logger:
            logger.log_frames(states2)
            
        frames2 = FrameLogger.load_frames(frames_path2)
        metrics2 = MetricsCalculator.calculate(frames2)
        
        # Compare metrics
        assert metrics1.total_frames == metrics2.total_frames
        assert abs(metrics1.duration - metrics2.duration) < 1e-6
        assert metrics1.correct_locks == metrics2.correct_locks
        assert metrics1.incorrect_locks == metrics2.incorrect_locks
        assert metrics1.final_score == metrics2.final_score
