"""
Phase 3 Default Scenario Test - Regression Guard

Ensures default scenario never regresses to 0 detections.
"""

import pytest


class TestDefaultScenarioRegression:
    """Guard against default scenario detection regression."""
    
    def test_default_produces_detections(self):
        """Default scenario MUST produce detections - regression guard."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        
        config = SimulationConfig(seed=42, duration=5.0, scenario='default')
        sim = SimulationCore(config)
        states = sim.run()
        
        total_det = sum(len(s.detections) for s in states)
        
        # CRITICAL: Default must produce detections
        assert total_det > 0, (
            f"REGRESSION: default scenario produced 0 detections! "
            f"Check config/scenarios/default.yaml FOV geometry."
        )
        
        # Should also produce locks
        # final_score = states[-1].score.get('total', 0) if states else 0
        # assert final_score >= 100, (
        #     f"default scenario should produce at least 1 lock (100 pts), got {final_score}"
        # )
        
    def test_default_scenario_yaml_exists(self):
        """Ensure default.yaml file exists."""
        from pathlib import Path
        
        # tests/regression/ -> root/scenarios
        yaml_path = Path(__file__).parents[2] / 'scenarios' / 'default.yaml'
        assert yaml_path.exists(), f"default.yaml must exist at {yaml_path}"
        
    def test_default_scenario_has_metadata(self):
        """Ensure default.yaml has Phase 3 metadata."""
        import yaml
        from pathlib import Path
        
        yaml_path = Path(__file__).parents[2] / 'scenarios' / 'default.yaml'
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        assert 'difficulty' in data, "default.yaml missing 'difficulty' metadata"
        assert 'expected_detection' in data, "default.yaml missing 'expected_detection' metadata"
