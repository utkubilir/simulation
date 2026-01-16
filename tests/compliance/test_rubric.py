"""
Phase 3 Rubric Integration Tests

Ensures rubric metrics are computed and persisted in metrics.json and benchmark_summary.csv.
"""

import pytest
import json
from pathlib import Path
import subprocess
import sys


@pytest.mark.skip(reason="Rubric calculation not yet integrated into run.py CLI output")
class TestRubricFieldsInMetricsJson:
    """Verify rubric fields are present in metrics.json after headless runs."""
    
    def test_rubric_fields_present_in_metrics_json(self, tmp_path):
        """Run easy_lock and verify metrics.json contains all rubric keys."""
        output_dir = tmp_path / "rubric_test"
        
        result = subprocess.run([
            sys.executable, "-m", "scripts.run",
            "--mode", "headless",
            "--scenario", "straight_approach",
            "--seed", "42",
            "--duration", "3",
            "--output", str(output_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parents[2])
        
        assert result.returncode == 0, f"Run failed: {result.stderr}"
        
        # Find metrics.json
        metrics_files = list(output_dir.rglob("metrics.json"))
        assert len(metrics_files) > 0, "metrics.json not found"
        
        with open(metrics_files[0]) as f:
            metrics = json.load(f)
            
        # Check all rubric keys present
        required_rubric_keys = [
            'false_lock_rate',
            'lock_stability_index', 
            'longest_continuous_lock',
            'reacquire_count',
            'track_continuity_index',
            'avg_track_age',
            'angular_accuracy_mean_deg',
            'angular_accuracy_max_deg',
        ]
        
        for key in required_rubric_keys:
            assert key in metrics, f"Rubric key '{key}' missing from metrics.json"
            
        # Verify some values are sensible
        assert metrics['false_lock_rate'] >= 0.0
        assert metrics['lock_stability_index'] >= 0.0
        assert metrics['reacquire_count'] >= 0


@pytest.mark.skip(reason="Rubric calculation not yet integrated into run.py CLI output")
class TestBenchmarkPopulatesRubricColumns:
    """Verify benchmark_summary.csv has populated rubric columns."""
    
    def test_benchmark_populates_rubric_columns(self, tmp_path):
        """Run minimal benchmark and verify CSV has rubric values."""
        import csv
        
        output_dir = tmp_path / "bench_test"
        
        # Run benchmark with just one scenario (use easy_lock for speed)
        result = subprocess.run([
            sys.executable, "-m", "scripts.run_all_scenarios",
            "--seeds", "42",
            "--output", str(output_dir)
        ], capture_output=True, text=True, cwd=Path(__file__).parents[2], timeout=180)
        
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
        
        # Check benchmark_summary.csv
        csv_path = output_dir / "benchmark_summary.csv"
        assert csv_path.exists(), "benchmark_summary.csv not found"
        
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) > 0, "No rows in benchmark_summary.csv"
        
        # Check easy_lock row has rubric columns populated
        easy_lock_row = [r for r in rows if r.get('scenario') == 'easy_lock']
        assert len(easy_lock_row) > 0, "easy_lock not in benchmark"
        
        row = easy_lock_row[0]
        
        # Rubric columns should exist and not be completely empty
        assert 'false_lock_rate' in row, "false_lock_rate column missing"
        assert 'lock_stability_index' in row, "lock_stability_index column missing"
        
        # Values should be non-empty (0.0 is valid)
        assert row['false_lock_rate'] != '', "false_lock_rate is blank"
        assert row['lock_stability_index'] != '', "lock_stability_index is blank"
