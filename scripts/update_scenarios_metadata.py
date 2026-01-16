
import yaml
from pathlib import Path

# Defaults
DEFAULT_METADATA = {
    'difficulty': 'medium',
    'expected_detection': True,
    'expected_lock': True,
    'tags': [],
    'notes': 'Phase 3 Polish'
}

# Specific overrides
OVERRIDES = {
    'easy_lock': {'difficulty': 'easy', 'tags': ['regression', 'baseline']},
    'straight_approach': {'difficulty': 'easy'},
    'default': {'difficulty': 'easy'},
    'noise_bbox_5px': {'difficulty': 'medium'},
    'noise_bbox_10px': {'difficulty': 'hard'},
    'latency_50ms': {'difficulty': 'medium'},
    'latency_100ms': {'difficulty': 'hard'},
    'fps_drop_15': {'difficulty': 'medium'},
    'evasive': {'difficulty': 'hard'},
    'zigzag': {'difficulty': 'hard'},
    'constant_turn': {'difficulty': 'medium'},
    'crossing_target': {'difficulty': 'medium', 'expected_lock': False}, # Often fails lock
    'high_speed_target': {'difficulty': 'hard', 'expected_lock': False}, # Often fails
    'low_contrast_target': {'difficulty': 'hard'},
    'occlusion_event': {'difficulty': 'hard', 'tags': ['occlusion']},
    'multi_target_3': {'difficulty': 'medium'},
    'turn_maneuver': {'difficulty': 'medium'}
}

def update_scenarios():
    scenarios_dir = Path("config/scenarios")
    for yaml_file in scenarios_dir.glob("*.yaml"):
        print(f"Processing {yaml_file.name}...")
        
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            
        name = data.get('name', yaml_file.stem)
        overrides = OVERRIDES.get(name, {})
        
        # Merge metadata
        # difficulty
        if 'difficulty' not in data:
            data['difficulty'] = overrides.get('difficulty', DEFAULT_METADATA['difficulty'])
            
        # expected_detection
        if 'expected_detection' not in data:
            data['expected_detection'] = overrides.get('expected_detection', DEFAULT_METADATA['expected_detection'])
            
        # expected_lock
        if 'expected_lock' not in data:
            data['expected_lock'] = overrides.get('expected_lock', DEFAULT_METADATA['expected_lock'])
            
        # expected_min_detections
        if 'expected_min_detections' not in data:
            data['expected_min_detections'] = 1 if data['expected_detection'] else 0
            
        # expected_min_valid_locks
        if 'expected_min_valid_locks' not in data:
            data['expected_min_valid_locks'] = 1 if data['expected_lock'] else 0
            
        # tags
        if 'tags' not in data:
            data['tags'] = overrides.get('tags', DEFAULT_METADATA['tags'])
            
        # notes
        if 'notes' not in data:
            data['notes'] = DEFAULT_METADATA['notes']
            
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False, indent=2)
            
    print("Done updating scenarios.")

if __name__ == "__main__":
    update_scenarios()
