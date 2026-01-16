
import pytest
import shutil
import json
import csv
from pathlib import Path
from tests.helpers import run_sim_headless, validate_jsonl_schema

class TestOutputContracts:
    """
    Contract tests to ensure simulation outputs adhere to strict schemas.
    These tests guarantee downstream tools (analytics, UI) won't break.
    """
    
    @classmethod
    def setup_class(cls):
        # Run a short simulation once to generate artifacts for inspection
        cls.metrics = run_sim_headless(
            scenario="default", 
            seed=999, 
            duration=2.0,
            output_temp_dir="/tmp/contract_test",
            run_id="contract_run"
        )

    @classmethod
    def teardown_class(cls):
        if cls.metrics.output_dir.exists():
            shutil.rmtree(cls.metrics.output_dir.parent, ignore_errors=True)

    @pytest.mark.compliance
    def test_frames_jsonl_schema(self):
        """
        Contract: frames.jsonl MUST contain specific keys on every line.
        Required: t, frame_id, own_state, enemies, detections, tracks, lock, score
        """
        required_keys = [
            't', 'frame_id', 'own_state', 'enemies', 
            'detections', 'tracks', 'lock', 'score'
        ]
        validate_jsonl_schema(self.metrics.frames_path, required_keys)

    @pytest.mark.compliance
    def test_metrics_json_schema(self):
        """
        Contract: metrics.json MUST contain summary statistics.
        """
        assert self.metrics.metrics_path.exists()
        with open(self.metrics.metrics_path) as f:
            data = json.load(f)
            
        required_keys = [
            'duration', 'total_frames', 'correct_locks', 
            'incorrect_locks', 'final_score', 'time_to_first_success_lock'
        ]
        for k in required_keys:
            assert k in data, f"metrics.json missing key: {k}"

    @pytest.mark.compliance
    def test_metrics_csv_structure(self):
        """
        Contract: metrics.csv MUST be present and have headers matching metrics.json keys (flat).
        """
        csv_path = self.metrics.output_dir / "metrics.csv"
        # Not all metrics.json keys are in CSV (e.g. nested lists usually aren't), 
        # but core scalar metrics should match.
        
        assert csv_path.exists()
        
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            
        required_headers = ['final_score', 'duration']
        for h in required_headers:
            assert h in headers, f"metrics.csv missing header: {h}"
