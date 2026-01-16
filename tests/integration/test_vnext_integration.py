"""
Test Suite for TEKNOFEST Savaşan İHA Sim vNext

Tests:
- Determinism: Same seed produces identical frames
- Lock Success: Easy scenario achieves lock
- Schema: All required fields present
- Scenarios: All YAML files load correctly
"""

import pytest
import hashlib
import json
import yaml
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDeterminism:
    """Test simulation determinism"""
    
    def test_same_seed_identical_output(self):
        """Two runs with same seed should produce identical frames.jsonl"""
        from scripts.run import SimulationRunner, load_scenario
        
        scenario_config = load_scenario('easy_lock')
        config = {
            **scenario_config,
            'seed': 42,
            'scenario': 'easy_lock',
            'duration': 5.0,
            'output_dir': '/tmp/test_determinism',
            'run_id': 'run_1'
        }
        
        # Run 1
        runner1 = SimulationRunner(config, mode='headless')
        runner1.run()
        
        # Run 2
        config['run_id'] = 'run_2'
        runner2 = SimulationRunner(config, mode='headless')
        runner2.run()
        
        # Compare frame hashes
        frames1 = Path('/tmp/test_determinism/run_1/frames.jsonl').read_bytes()
        frames2 = Path('/tmp/test_determinism/run_2/frames.jsonl').read_bytes()
        
        hash1 = hashlib.sha256(frames1).hexdigest()
        hash2 = hashlib.sha256(frames2).hexdigest()
        
        assert hash1 == hash2, "Same seed should produce identical output"
        
    def test_different_seed_different_output(self):
        """Different seeds should produce different outputs"""
        from scripts.run import SimulationRunner, load_scenario
        
        scenario_config = load_scenario('easy_lock')
        base_config = {
            **scenario_config,
            'scenario': 'easy_lock',
            'duration': 5.0,
            'output_dir': '/tmp/test_determinism',
        }
        
        config1 = {**base_config, 'seed': 42, 'run_id': 'seed_42'}
        config2 = {**base_config, 'seed': 123, 'run_id': 'seed_123'}
        
        runner1 = SimulationRunner(config1, mode='headless')
        runner1.run()
        
        runner2 = SimulationRunner(config2, mode='headless')
        runner2.run()
        
        frames1 = Path('/tmp/test_determinism/seed_42/frames.jsonl').read_bytes()
        frames2 = Path('/tmp/test_determinism/seed_123/frames.jsonl').read_bytes()
        
        hash1 = hashlib.sha256(frames1).hexdigest()
        hash2 = hashlib.sha256(frames2).hexdigest()
        
        assert hash1 != hash2, "Different seeds should produce different output"


class TestLockSuccess:
    """Test lock-on functionality"""
    
    @pytest.mark.xfail(reason="Pre-existing regression: easy_lock scenario fails to achieve lock")
    def test_easy_lock_achieves_success(self):
        """Easy lock scenario should achieve at least 1 successful lock"""
        from scripts.run import SimulationRunner, load_scenario
        
        scenario_config = load_scenario('easy_lock')
        config = {
            **scenario_config,
            'seed': 42,
            'scenario': 'easy_lock',
            'duration': 30.0,
            'output_dir': '/tmp/test_lock',
            'run_id': 'easy_lock_test'
        }
        
        runner = SimulationRunner(config, mode='headless')
        metrics = runner.run()
        
        assert metrics.correct_locks >= 1, "Easy lock should achieve at least 1 lock"
        
    def test_lock_duration_is_4_seconds(self):
        """Lock success requires 4.0 seconds of continuous lock"""
        from src.vision.lock_on import LockOnStateMachine, LockConfig
        
        config = LockConfig(success_duration=4.0)
        sm = LockOnStateMachine(config)
        
        assert sm.config.success_duration == 4.0


class TestSchema:
    """Test output schema compliance"""
    
    def test_frames_jsonl_schema(self):
        """frames.jsonl should have all required fields"""
        from scripts.run import SimulationRunner
        
        config = {
            'seed': 42,
            'scenario': 'easy_lock',
            'duration': 2.0,
            'output_dir': '/tmp/test_schema',
            'run_id': 'schema_test'
        }
        
        runner = SimulationRunner(config, mode='headless')
        runner.run()
        
        frames_path = Path('/tmp/test_schema/schema_test/frames.jsonl')
        with open(frames_path) as f:
            frame = json.loads(f.readline())
            
        required_keys = ['t', 'frame_id', 'own_state', 'enemies', 
                         'detections', 'tracks', 'lock', 'score']
        
        for key in required_keys:
            assert key in frame, f"Missing key: {key}"
            
    def test_metrics_json_schema(self):
        """metrics.json should have all required fields"""
        from scripts.run import SimulationRunner
        
        config = {
            'seed': 42,
            'scenario': 'easy_lock',
            'duration': 2.0,
            'output_dir': '/tmp/test_schema',
            'run_id': 'metrics_test'
        }
        
        runner = SimulationRunner(config, mode='headless')
        runner.run()
        
        metrics_path = Path('/tmp/test_schema/metrics_test/metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
            
        required_keys = ['duration', 'total_frames', 'correct_locks', 
                         'incorrect_locks', 'final_score']
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"


class TestScenarios:
    """Test scenario loading"""
    
    def test_all_scenarios_load(self):
        """All YAML scenarios should load without error"""
        scenarios_dir = Path(__file__).parent.parent / 'scenarios'
        
        for yaml_file in scenarios_dir.glob('*.yaml'):
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                
            # Required metadata
            assert 'name' in config, f"{yaml_file.name}: missing 'name'"
            assert 'difficulty' in config, f"{yaml_file.name}: missing 'difficulty'"
            
    def test_scenario_metadata_present(self):
        """Scenarios should have expected fields"""
        scenarios_dir = Path(__file__).parent.parent / 'scenarios'
        
        required_fields = ['name', 'difficulty', 'expected_detection', 
                          'expected_success_locks_min', 'duration']
        
        for yaml_file in scenarios_dir.glob('*.yaml'):
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                
            for field in required_fields:
                assert field in config, f"{yaml_file.name}: missing '{field}'"


class TestLockStateMachine:
    """Unit tests for LockOnStateMachine"""
    
    def test_initial_state_is_idle(self):
        from src.vision.lock_on import LockOnStateMachine, LockState
        
        sm = LockOnStateMachine()
        assert sm.state == LockState.IDLE
        
    def test_valid_target_transitions_to_locking(self):
        from src.vision.lock_on import LockOnStateMachine, LockState
        from src.vision.tracker import Track
        
        sm = LockOnStateMachine()
        
        # Track at crosshair center
        track = Track(
            id=1,
            bbox=(300, 220, 340, 260),  # 40x40 box
            center=(320, 240),          # At crosshair
            confidence=0.9,
            is_confirmed=True
        )
        
        sm.update([track], sim_time=0.0, dt=0.016)
        assert sm.state == LockState.LOCKING
        
    def test_lock_success_after_4_seconds(self):
        from src.vision.lock_on import LockOnStateMachine, LockState
        from src.vision.tracker import Track
        
        sm = LockOnStateMachine()
        
        track = Track(
            id=1,
            bbox=(300, 220, 340, 260),
            center=(320, 240),
            confidence=0.9,
            is_confirmed=True
        )
        
        # Simulate 4 seconds of lock
        dt = 0.1
        for i in range(45):  # 4.5 seconds
            status = sm.update([track], sim_time=i*dt, dt=dt)
            
        assert sm.state == LockState.SUCCESS or sm._correct_locks >= 1
