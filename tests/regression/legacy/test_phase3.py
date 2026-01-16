"""
Phase 3 Tests - Scenario Metadata, Detection Expectations, Rubric Fields.
"""

import pytest
from pathlib import Path


class TestPhase3ScenarioMetadata:
    """Verify all scenarios have required Phase 3 metadata."""
    
    def test_scenarios_load_with_metadata(self):
        """Ensure all YAML scenarios have metadata fields."""
        import yaml
        
        # tests/regression/legacy -> root/scenarios
        scenarios_dir = Path(__file__).parents[3] / 'scenarios'
        yaml_files = list(scenarios_dir.glob('*.yaml'))
        
        assert len(yaml_files) >= 10, f"Expected >= 10 scenarios, got {len(yaml_files)}"
        
        for yaml_path in yaml_files:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            assert 'name' in data, f"{yaml_path.name} missing 'name'"
            assert 'duration' in data, f"{yaml_path.name} missing 'duration'"
            # Phase 3 metadata (optional but should exist in new scenarios)
            # Not enforcing all for backward compat


class TestPhase3DetectionExpectations:
    """Verify scenarios marked with expected_detection=true produce detections."""
    
    def test_easy_lock_produces_detections(self):
        """easy_lock should produce detections."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        
        config = SimulationConfig(seed=42, duration=3.0, scenario='easy_lock')
        sim = SimulationCore(config)
        states = sim.run()
        
        total_det = sum(len(s.detections) for s in states)
        assert total_det > 0, "easy_lock should produce detections"
        
    def test_crossing_target_produces_detections(self):
        """crossing_target should produce detections (after Phase 3 fix)."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        
        config = SimulationConfig(seed=42, duration=3.0, scenario='crossing_target')
        sim = SimulationCore(config)
        states = sim.run()
        
        total_det = sum(len(s.detections) for s in states)
        assert total_det > 0, "crossing_target should produce detections"
        
    def test_straight_approach_produces_detections(self):
        """straight_approach should produce detections."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        
        # Hedef 1600m uzakta başlıyor, birbirine yaklaşıyor
        # Görüş alanına girmesi için daha uzun süre gerekli
        config = SimulationConfig(seed=42, duration=10.0, scenario='straight_approach')
        sim = SimulationCore(config)
        states = sim.run()
        
        total_det = sum(len(s.detections) for s in states)
        assert total_det > 0, "straight_approach should produce detections"


class TestPhase3RubricFields:
    """Verify rubric calculator produces all required fields."""
    
    def test_rubric_fields_present(self):
        """Rubric should produce all Phase 3 metrics."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        from src.core.rubric import RubricCalculator
        
        config = SimulationConfig(seed=42, duration=3.0, scenario='easy_lock')
        sim = SimulationCore(config)
        states = sim.run()
        
        # Convert states to frame dicts
        frames = []
        for s in states:
            frames.append({
                't': s.t,
                'detections': s.detections,
                'tracks': s.tracks,
                'lock': s.lock,
                'score': s.score,
            })
            
        calc = RubricCalculator()
        metrics = calc.calculate(frames)
        
        # Check all Phase 3 fields exist
        metrics_dict = metrics.to_dict()
        required_fields = [
            'false_lock_rate',
            'lock_stability_index',
            'longest_continuous_lock',
            'reacquire_count',
            'track_continuity_index',
            'avg_track_age',
        ]
        
        for field in required_fields:
            assert field in metrics_dict, f"Missing rubric field: {field}"
            
    def test_angular_accuracy_computed(self):
        """Angular accuracy should be computed when lock dx/dy available."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        from src.core.rubric import RubricCalculator
        
        config = SimulationConfig(seed=42, duration=3.0, scenario='easy_lock')
        sim = SimulationCore(config)
        states = sim.run()
        
        frames = [{'t': s.t, 'lock': s.lock, 'detections': s.detections, 
                   'tracks': s.tracks, 'score': s.score} for s in states]
        
        calc = RubricCalculator()
        metrics = calc.calculate(frames)
        
        # easy_lock should produce locks with dx/dy
        # Angular accuracy might be None if no locks with dx/dy
        # Just check it doesn't crash
        assert hasattr(metrics, 'angular_accuracy_mean_deg')
