"""
Diagnose Run Script
Analyzes frames.jsonl to pinpoint failure stages in the lock-on pipeline.
"""

import json
import argparse
from pathlib import Path
import numpy as np

def analyze_run(run_dir):
    run_path = Path(run_dir)
    frames_file = list(run_path.rglob('frames.jsonl'))[0]
    metrics_file = list(run_path.rglob('metrics.json'))[0]
    
    print(f"Analyzing: {run_path}")
    
    with open(metrics_file) as f:
        metrics = json.load(f)
        print("\n--- Metrics ---")
        print(json.dumps(metrics, indent=2))
        
    with open(frames_file) as f:
        frames = [json.loads(l) for l in f]
        
    total_frames = len(frames)
    det_frames = [f for f in frames if f['detections']]
    track_frames = [f for f in frames if f['tracks']]
    lock_frames = [f for f in frames if f['lock'].get('locked')]
    valid_lock_frames = [f for f in frames if f['lock'].get('valid')]
    
    print(f"\n--- Pipeline Stages (N={total_frames}) ---")
    print(f"1. Detection Rate: {len(det_frames)} frames ({100*len(det_frames)/total_frames:.1f}%)")
    print(f"2. Tracking Rate:  {len(track_frames)} frames ({100*len(track_frames)/total_frames:.1f}%)")
    print(f"3. Lock Active:    {len(lock_frames)} frames ({100*len(lock_frames)/total_frames:.1f}%)")
    print(f"4. Lock Valid:     {len(valid_lock_frames)} frames ({100*len(valid_lock_frames)/total_frames:.1f}%)")
    
    if det_frames:
        first_det = det_frames[0]
        print(f"\nFirst Detection at t={first_det['t']:.3f}s:")
        print(f"  Center: {first_det['detections'][0]['center']}")
        print(f"  BBox: {first_det['detections'][0]['bbox']}")
        
    # Analyze Lock Criteria on Tracking Frames
    print("\n--- Lock Criteria Analysis ---")
    criteria_met = 0
    crosshair = (320, 240) # assume default
    
    for f in track_frames:
        if not f['tracks']:
            continue
        # Assuming track 0 is target
        t = f['tracks'][0] 
        # Track structure in log is {'id', 'bbox', 'age'}
        # We assume bbox is [x1, y1, x2, y2]
        x1, y1, x2, y2 = t['bbox']
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        w = x2-x1
        h = y2-y1
        
        dx = abs(cx - crosshair[0])
        dy = abs(cy - crosshair[1])
        
        h_ok = dx <= w * 0.5
        v_ok = dy <= h * 0.5
        
        if h_ok and v_ok:
            criteria_met += 1
            
    print(f"Frames meeting geometric lock criteria: {criteria_met} ({100*criteria_met/len(track_frames) if track_frames else 0:.1f}% of tracked frames)")
    
    if metrics['lock_ratio'] == 0:
        print("\n!!! DIAGNOSIS !!!")
        if len(det_frames) < total_frames * 0.1:
            print("FAILURE: Detector is missing targets. Check FOV, range, or detection probability.")
        elif len(track_frames) < len(det_frames) * 0.8:
            print("FAILURE: Tracker is dropping detections. Check IoU threshold or track logic.")
        elif criteria_met == 0:
            print("FAILURE: Target is tracked but never centered. Check scenario alignment (heading/position) or control logic.")
        else:
            print("FAILURE: Lock criteria met but lock not forming. Check min_duration or lock logic timing.")
    else:
        print("\nSUCCESS: System is working.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Results directory")
    args = parser.parse_args()
    try:
        analyze_run(args.run_dir)
    except Exception as e:
        print(f"Error analyzing run: {e}")
