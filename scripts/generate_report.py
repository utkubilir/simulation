#!/usr/bin/env python3
"""
Report Generator - Phase 3

Generates markdown reports from benchmark results.
- Per-run report: individual run analysis
- Aggregate report: benchmark summary with pass/fail gates
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# Pass/fail thresholds
THRESHOLDS = {
    'time_to_first_lock_max': 2.0,      # seconds
    'lock_ratio_min': 0.2,               # 20%
    'false_lock_rate_max': 0.05,         # 5%
    'angular_accuracy_mean_max_deg': 2.0 # degrees
}


def load_benchmark_summary(csv_path: Path) -> List[Dict]:
    """Load benchmark summary CSV into list of dicts."""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def evaluate_pass_fail(row: Dict) -> Dict[str, bool]:
    """Evaluate pass/fail for a single scenario result."""
    checks = {}
    
    # Time to first lock
    ttfl = row.get('time_to_first_lock', '')
    if ttfl and ttfl != '':
        try:
            checks['time_to_first_lock'] = float(ttfl) <= THRESHOLDS['time_to_first_lock_max']
        except ValueError:
            checks['time_to_first_lock'] = False
    else:
        checks['time_to_first_lock'] = None  # N/A
        
    # Lock ratio
    lr = row.get('lock_ratio', '0')
    try:
        checks['lock_ratio'] = float(lr) >= THRESHOLDS['lock_ratio_min']
    except ValueError:
        checks['lock_ratio'] = False
        
    # False lock rate
    correct = int(row.get('correct_locks', '0') or 0)
    incorrect = int(row.get('incorrect_locks', '0') or 0)
    total = correct + incorrect
    if total > 0:
        flr = incorrect / total
        checks['false_lock_rate'] = flr <= THRESHOLDS['false_lock_rate_max']
    else:
        checks['false_lock_rate'] = None  # N/A
        
    # Detection check
    det = int(row.get('total_detections', '0') or 0)
    checks['has_detections'] = det > 0
    
    return checks



def generate_aggregate_report(results: List[Dict], output_path: Path) -> str:
    """Generate aggregate markdown report."""
    
    # Load metadata for all scenarios to filter analysis
    from src.scenarios import ScenarioLoader
    loader = ScenarioLoader()
    scenario_meta = {}
    for r in results:
        scen = r.get('scenario')
        if scen and scen not in scenario_meta:
            try:
                conf = loader.load(scen)
                scenario_meta[scen] = {
                    'expected_lock': conf.expected_lock,
                    'expected_detection': conf.expected_detection
                }
            except:
                scenario_meta[scen] = {'expected_lock': True, 'expected_detection': True}

    lines = []
    lines.append("# Benchmark Report (Phase 3 Polish)")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Total Scenarios**: {len(results)}")
    
    # 1. Pass/Fail Summary
    passed = 0
    failed = 0
    fail_reasons_dist = {}
    
    for row in results:
        # Check overall_pass column if exists, otherwise assume True
        overall_pass = row.get('overall_pass')
        if isinstance(overall_pass, str):
            overall_pass = (overall_pass.lower() == 'true')
        
        if overall_pass:
            passed += 1
        else:
            failed += 1
            reason = row.get('fail_reason', 'unknown')
            fail_reasons_dist[reason] = fail_reasons_dist.get(reason, 0) + 1
            
    lines.append("\n## Pass/Fail Summary")
    lines.append(f"\n- **Passed**: {passed}")
    lines.append(f"- **Failed**: {failed}")
    if failed > 0:
        lines.append("\n**Fail Reasons**:")
        for r, count in fail_reasons_dist.items():
            lines.append(f"- {r}: {count}")
    
    # 2. Executive Summary - Top 5 Worst
    lines.append("\n## Worst Scenarios (Top 5)")
    
    # Filter for expected_lock=True
    lock_scenarios = [r for r in results if scenario_meta.get(r['scenario'], {}).get('expected_lock', True)]
    
    # Sort by Valid Lock Ratio (Lowest)
    lines.append("\n### Lowest Valid Lock Ratio (Expected Lock=True)")
    lock_scenarios.sort(key=lambda x: float(x.get('valid_lock_ratio', 0) or 0))
    lines.append("| Scenario | Valid Lock% | Valid Time | Locked% |")
    lines.append("|----------|-------------|------------|---------|")
    for r in lock_scenarios[:5]:
        vlr = float(r.get('valid_lock_ratio', 0) or 0)
        vlt = float(r.get('valid_lock_time_total', 0) or 0)
        raw_lr = float(r.get('locked_ratio', 0) or 0)
        lines.append(f"| {r['scenario']} | {vlr*100:.1f}% | {vlt:.2f}s | {raw_lr*100:.1f}% |")

    # Sort by Angular Accuracy (Highest/Worst Mean)
    # Filter those that have data
    ang_scenarios = [r for r in results if r.get('angular_accuracy_mean_deg') and r.get('angular_accuracy_mean_deg') != '']
    ang_scenarios.sort(key=lambda x: float(x.get('angular_accuracy_mean_deg', 0)), reverse=True)
    
    lines.append("\n### Worst Angular Accuracy (Mean Deg)")
    lines.append("| Scenario | Mean Deg | Max Deg | Valid Locks |")
    lines.append("|----------|----------|---------|-------------|")
    for r in ang_scenarios[:5]:
        mean_deg = float(r.get('angular_accuracy_mean_deg'))
        max_deg = float(r.get('angular_accuracy_max_deg'))
        vl_count = r.get('valid_lock_count', 0)
        lines.append(f"| {r['scenario']} | {mean_deg:.2f}° | {max_deg:.2f}° | {vl_count} |")
        
    # Sort by Track Continuity (Lowest)
    track_scenarios = [r for r in results if r.get('track_continuity_index') and r.get('track_continuity_index') != '']
    track_scenarios.sort(key=lambda x: float(x.get('track_continuity_index', 0)))
    
    lines.append("\n### Lowest Track Continuity")
    lines.append("| Scenario | Continuity Index | Avg Track Age |")
    lines.append("|----------|------------------|---------------|")
    for r in track_scenarios[:5]:
        cont = float(r.get('track_continuity_index'))
        age = float(r.get('avg_track_age'))
        lines.append(f"| {r['scenario']} | {cont:.3f} | {age:.1f} frames |")
        
    # 3. Full Results Table
    lines.append("\n## Detailed Scenario Results")
    lines.append("\n| Scenario | Pass/Fail | Score | Valid Lock% | Pass Reason |")
    lines.append("|----------|-----------|-------|-------------|-------------|")
    
    for row in results:
        overall_pass = row.get('overall_pass')
        if isinstance(overall_pass, str):
            overall_pass = (overall_pass.lower() == 'true')
        status_icon = "✅" if overall_pass else "❌"
        
        vlr = float(row.get('valid_lock_ratio', 0) or 0)
        reason = "OK" if overall_pass else row.get('fail_reason', 'unknown')
        
        lines.append(f"| {row['scenario']} | {status_icon} | {row.get('final_score')} | {vlr*100:.1f}% | {reason} |")
        
    # 4. Interpretation Notes
    lines.append("\n## Interpretation Notes (Phase 3 Polish)")
    lines.append("""
> [!NOTE]
> **Locked vs Valid Lock**:
> - **Locked Time**: Total duration where the system reported `locked=True`.
> - **Valid Lock Time**: Duration where `locked=True` AND the lock was considered valid (within scoring tolerances, sufficiently centered, etc.).
> - **Valid Lock Ratio**: The primary metric for evaluation. `valid_lock_time / duration`.
>
> **Gates**:
> - **Detection Gate**: Scenarios expecting detections must have ≥1 detection.
> - **Lock Gate**: Scenarios expecting locks must have ≥1 *valid* lock event.
""")
    
    report = '\n'.join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    return report


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark report')
    parser.add_argument('--runs-dir', '-d', type=str, required=True,
                       help='Path to benchmark results directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output report path (default: <runs-dir>/REPORT.md)')
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    
    # Find benchmark summary
    csv_path = runs_dir / 'benchmark_summary.csv'
    if not csv_path.exists():
        print(f"❌ benchmark_summary.csv not found at {csv_path}")
        return 1
        
    # Load results
    results = load_benchmark_summary(csv_path)
    print(f"📊 Loaded {len(results)} scenario results")
    
    # Generate report
    output_path = Path(args.output) if args.output else runs_dir / 'REPORT.md'
    report = generate_aggregate_report(results, output_path)
    
    print(f"✅ Report generated: {output_path}")
    print(f"\n{'-'*50}")
    # Print first 30 lines of report
    for line in report.split('\n')[:30]:
        print(line)
    print("...")
    
    return 0


if __name__ == '__main__':
    exit(main())
