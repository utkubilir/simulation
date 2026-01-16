"""
Scenario Loader - Load and validate scenario YAML files.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


class ScenarioLoadError(Exception):
    """Raised when a scenario cannot be loaded."""
    
    def __init__(self, scenario_name: str, message: str = None):
        self.scenario_name = scenario_name
        msg = f"Failed to load scenario '{scenario_name}'"
        if message:
            msg += f": {message}"
        super().__init__(msg)


@dataclass
class TargetDefinition:
    """Target UAV definition in scenario."""
    id: str
    position: List[float]
    heading: float = 0.0
    team: str = "red"
    behavior: str = "straight"
    behavior_params: Dict = field(default_factory=dict)


@dataclass 
class OwnUAVDefinition:
    """Own UAV definition in scenario."""
    position: List[float]
    heading: float = 0.0
    altitude: Optional[float] = None
    speed: Optional[float] = None


@dataclass
class ScenarioDefinition:
    """Complete scenario definition."""
    name: str
    description: str = ""
    duration: float = 10.0
    own_uav: OwnUAVDefinition = None
    targets: List[TargetDefinition] = field(default_factory=list)
    
    # Optional overrides
    camera_fov: Optional[float] = None
    camera_resolution: Optional[tuple] = None
    lock_min_duration: Optional[float] = None
    seed_overrides: Optional[int] = None
    world_overrides: Dict = field(default_factory=dict)
    
    # Phase 2: Noise configuration
    noise_config: Dict = field(default_factory=dict)
    
    # Phase 2: Latency configuration
    latency_config: Dict = field(default_factory=dict)
    
    # Phase 2: Perception FPS override
    perception_fps: Optional[float] = None
    
    # Phase 3 Polish: Metadata
    difficulty: str = "medium"
    expected_detection: bool = True
    expected_lock: bool = True
    expected_min_detections: int = 1
    expected_min_valid_locks: int = 1
    notes: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScenarioDefinition':
        """Create from dictionary (YAML parsed)."""
        # Parse own UAV
        own_data = data.get('own_uav', {})
        own_uav = OwnUAVDefinition(
            position=own_data.get('position', [500, 500, 100]),
            heading=own_data.get('heading', 0),
            altitude=own_data.get('altitude'),
            speed=own_data.get('speed')
        )
        
        # Parse targets
        targets = []
        for i, t in enumerate(data.get('targets', [])):
            target = TargetDefinition(
                id=t.get('id', f'target_{i:02d}'),
                position=t.get('position', [700, 500, 100]),
                heading=t.get('heading', 180),
                team=t.get('team', 'red'),
                behavior=t.get('behavior', 'straight'),
                behavior_params=t.get('behavior_params', {})
            )
            targets.append(target)
            
        # Build scenario
        return cls(
            name=data.get('name', 'unnamed'),
            description=data.get('description', ''),
            duration=data.get('duration', 10.0),
            own_uav=own_uav,
            targets=targets,
            camera_fov=data.get('camera', {}).get('fov') or data.get('camera_overrides', {}).get('fov'),
            camera_resolution=tuple(data.get('camera', {}).get('resolution', [])) or tuple(data.get('camera_overrides', {}).get('resolution', [])) or None,
            lock_min_duration=data.get('lock_on', {}).get('min_duration') or data.get('lock_overrides', {}).get('min_duration'),
            seed_overrides=data.get('seed_overrides'),
            world_overrides=data.get('world', {}),
            noise_config=data.get('noise', {}),
            latency_config=data.get('latency', {}),
            perception_fps=data.get('perception_fps'),
            # Phase 3 Polish
            difficulty=data.get('difficulty', 'medium'),
            expected_detection=data.get('expected_detection', True),
            expected_lock=data.get('expected_lock', True),
            expected_min_detections=data.get('expected_min_detections', 1),
            expected_min_valid_locks=data.get('expected_min_valid_locks', 1),
            notes=data.get('notes', ""),
        )


class ScenarioLoader:
    """Load scenarios from YAML files."""
    
    DEFAULT_SCENARIOS_DIR = Path(__file__).parent.parent.parent / 'config' / 'scenarios'
    
    # Built-in scenarios (fallback if no YAML)
    BUILTIN_SCENARIOS = {
        'default': {
            'name': 'default',
            'description': 'Default single target scenario',
            'duration': 10.0,
            'own_uav': {'position': [500, 500, 100], 'heading': 45},
            'targets': [
                {'id': 'target_01', 'position': [700, 500, 100], 'heading': 180, 'behavior': 'straight'}
            ]
        }
    }
    
    def __init__(self, scenarios_dir: Path = None):
        self.scenarios_dir = Path(scenarios_dir) if scenarios_dir else self.DEFAULT_SCENARIOS_DIR
        self._cache: Dict[str, ScenarioDefinition] = {}
        
    def list_scenarios(self) -> List[str]:
        """List available scenario names."""
        scenarios = list(self.BUILTIN_SCENARIOS.keys())
        
        if self.scenarios_dir.exists():
            for f in self.scenarios_dir.glob('*.yaml'):
                name = f.stem
                if name not in scenarios:
                    scenarios.append(name)
                    
        return sorted(scenarios)
        
    def load(self, name: str) -> ScenarioDefinition:
        """Load scenario by name."""
        if name in self._cache:
            return self._cache[name]
            
        # Try YAML file first
        yaml_path = self.scenarios_dir / f'{name}.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            scenario = ScenarioDefinition.from_dict(data)
        elif name in self.BUILTIN_SCENARIOS:
            scenario = ScenarioDefinition.from_dict(self.BUILTIN_SCENARIOS[name])
        else:
            raise ScenarioLoadError(name, f"Available scenarios: {self.list_scenarios()}")
            
        self._cache[name] = scenario
        return scenario
        
    def load_from_file(self, path: Path) -> ScenarioDefinition:
        """Load scenario from specific file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return ScenarioDefinition.from_dict(data)
        
    @staticmethod
    def validate(scenario: ScenarioDefinition) -> List[str]:
        """Validate scenario, return list of issues."""
        issues = []
        
        if not scenario.name:
            issues.append("Scenario must have a name")
            
        if not scenario.own_uav:
            issues.append("Scenario must define own_uav")
        elif not scenario.own_uav.position or len(scenario.own_uav.position) != 3:
            issues.append("own_uav.position must be [x, y, z]")
            
        if not scenario.targets:
            issues.append("Scenario should have at least one target")
            
        for t in scenario.targets:
            if not t.position or len(t.position) != 3:
                issues.append(f"Target {t.id} position must be [x, y, z]")
            if t.behavior not in ['straight', 'constant_turn', 'zigzag', 'evasive', 'random']:
                issues.append(f"Target {t.id} has invalid behavior: {t.behavior}")
                
        if scenario.duration <= 0:
            issues.append("Duration must be positive")
            
        return issues
