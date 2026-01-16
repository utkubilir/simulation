"""
Benchmark Runner - Run all scenarios and aggregate results.

Usage:
  python -m scripts.run_all_scenarios --seeds 42 123 --output results/benchmark_v1
"""

import argparse
import sys
import subprocess
import json
import csv
from pathlib import Path
from typing import List, Dict
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.scenarios import ScenarioLoader


def run_scenario(scenario: str, seed: int, output_dir: Path) -> Dict:
    """Run a single scenario and return metrics."""
    run_output = output_dir / f"{scenario}_seed{seed}"
    
    cmd = [
        sys.executable, "-m", "scripts.run",
        "--mode", "headless",
        "--scenario", scenario,
        "--seed", str(seed),
        "--output", str(run_output),
        "--duration", "10"  # Default duration for benchmark
    ]
    
    print(f"Running {scenario} (seed {seed})...", end="", flush=True)
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        duration = time.time() - start_time
        print(f" Done ({duration:.1f}s)")
        
        # Load metrics - find in subdirectory if needed
        metrics_file = run_output / "metrics.json"
        
        # scripts.run creates a timestamped subdirectory inside output_dir if headless
        if not metrics_file.exists():
            # Try to find a subdirectory
            subdirs = [d for d in run_output.iterdir() if d.is_dir()]
            if subdirs:
                # Use the most recent one if multiple (though clean run should have one)
                subdir = sorted(subdirs)[-1]
                metrics_file = subdir / "metrics.json"
                
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            metrics['scenario'] = scenario
            metrics['seed'] = seed
            metrics['status'] = 'success'
            return metrics
        else:
            print(f" Error: metrics.json not found in {run_output} or subdirs")
            return {
                'scenario': scenario, 
                'seed': seed, 
                'status': 'failed', 
                'error': 'no_metrics'
            }
            
    except subprocess.CalledProcessError as e:
        print(f" Failed!")
        print(f"Error output: {e.stderr}")
        return {
            'scenario': scenario, 
            'seed': seed, 
            'status': 'crashed',
            'error': str(e)
        }


def apply_gate_logic(row: Dict, scenario_conf) -> Dict:
    """Apply detection/lock gate logic to a metrics row."""
    exp_det = scenario_conf.expected_detection
    min_det = scenario_conf.expected_min_detections

    exp_lock = scenario_conf.expected_lock
    min_valid_lock = scenario_conf.expected_min_valid_locks

    total_det = row.get('total_detections', 0)
    valid_locks = row.get('valid_lock_count', 0)

    det_pass = (total_det >= min_det) if exp_det else True
    lock_pass = (valid_locks >= min_valid_lock) if exp_lock else True

    fail_reasons = []
    if not det_pass:
        fail_reasons.append("no_detections")
    if not lock_pass:
        fail_reasons.append("no_valid_locks")

    row['detection_gate_pass'] = det_pass
    row['lock_gate_pass'] = lock_pass
    row['overall_pass'] = det_pass and lock_pass
    row['fail_reason'] = ",".join(fail_reasons) if fail_reasons else "ok"
    return row


def main():
    parser = argparse.ArgumentParser(description="Run simulation benchmark")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="List of seeds to run")
    parser.add_argument("--output", type=str, default="results/benchmark", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get scenarios
    loader = ScenarioLoader()
    scenarios = loader.list_scenarios()
    
    print(f"Found {len(scenarios)} scenarios: {', '.join(scenarios)}")
    print(f"Running with {len(args.seeds)} seeds: {args.seeds}")
    
    results = []
    
    # Run loop
    for seed in args.seeds:
        for scenario in scenarios:
            metrics = run_scenario(scenario, seed, output_dir)
            results.append(metrics)
            
    # Save aggregate CSV
    csv_file = output_dir / "benchmark_summary.csv"
    if results:
        # Determine fields (union of all keys) - includes Phase 3 rubric fields
        fieldnames = ['scenario', 'seed', 'status', 'overall_pass', 'fail_reason', 
                      'detection_gate_pass', 'lock_gate_pass', 'final_score', 
                      'time_to_first_lock', 'lock_ratio', 'correct_locks', 
                      'incorrect_locks', 'total_detections', 'total_frames', 'duration',
                      # Phase 3 rubric fields
                      'false_lock_rate', 'lock_stability_index', 'longest_continuous_lock',
                      'reacquire_time_mean', 'reacquire_count', 'track_continuity_index',
                      'avg_track_age', 'angular_accuracy_mean_deg', 'angular_accuracy_max_deg',
                      # Phase 3 Polish fields
                      'locked_time_total', 'locked_ratio', 'valid_lock_time_total', 
                      'valid_lock_ratio', 'valid_longest_continuous_lock', 
                      'time_to_first_valid_lock', 'valid_lock_count', 'invalid_lock_time_total']
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in results:
                # Calculate gates if scenario is loaded
                if row.get('status') == 'success':
                    # Load scenario metadata using the existing loader
                    try:
                        scenario_conf = loader.load(row['scenario'])
                        apply_gate_logic(row, scenario_conf)
                        
                    except Exception as e:
                        print(f"Warning: Could not check gates for {row['scenario']}: {e}")
                        row['overall_pass'] = False
                        row['fail_reason'] = "gate_error"
                
                # Round floats to 4 decimal places for readability
                clean_row = {}
                for k, v in row.items():
                    if isinstance(v, float):
                        clean_row[k] = round(v, 4)
                    else:
                        clean_row[k] = v
                writer.writerow(clean_row)
                
    print(f"\nBenchmark complete. Summary saved to {csv_file}")
    
    # Print simple summary
    print("\nSummary:")
    print(f"{'Scenario':<20} {'Seed':<5} {'Score':<5} {'Lock%':<6} {'TimeToLock'}")
    print("-" * 50)
    for r in results:
        if r.get('status') == 'success':
            lock_ratio = r.get('lock_ratio', 0) * 100
            ttl = r.get('time_to_first_lock')
            ttl_str = f"{ttl:.2f}s" if ttl else "N/A"
            print(f"{r['scenario']:<20} {r['seed']:<5} {r.get('final_score',0):<5} {lock_ratio:>5.1f}% {ttl_str}")
        else:
            print(f"{r['scenario']:<20} {r['seed']:<5} FAILED ({r.get('error')})")


if __name__ == "__main__":
    main()
