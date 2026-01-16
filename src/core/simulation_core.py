"""
SimulationCore - Headless-capable simulation engine.

Separates simulation logic from rendering for CI/benchmark runs.
Supports scenario YAML loading and target maneuver behaviors.
Phase 2: Noise models and latency simulation.
"""

import random
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..simulation.world import SimulationWorld
from ..simulation.camera import SimulatedCamera
from ..vision.detector import SimulationDetector
from ..vision.tracker import TargetTracker
from ..vision.lock_on import LockOnSystem, LockState
from ..vision.noise import DetectionNoiseModel, NoiseConfig
from ..vision.delay import DelayedStream, LatencyConfig
from ..competition.server_sim import CompetitionServerSimulator
from ..uav.target_behavior import TargetManeuverController
from ..scenarios import ScenarioLoader, ScenarioDefinition


@dataclass
class SimulationState:
    """Immutable snapshot of simulation state at a given time."""
    t: float
    frame_id: int
    scenario: str
    own_state: Dict
    targets: List[Dict]
    detections: List[Dict]
    tracks: List[Dict]
    lock: Dict
    score: Dict
    # Phase 2: Optional noise/latency metadata (backward-compatible)
    noise: Optional[Dict] = None
    latency: Optional[Dict] = None


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    seed: int = 42
    duration: float = 10.0
    physics_dt: float = 1.0 / 60.0
    perception_fps: float = 30.0
    scenario: str = "default"
    team_id: str = "team_001"
    
    # Scenario parameters (used if no YAML scenario found)
    own_position: List[float] = field(default_factory=lambda: [500.0, 500.0, 100.0])
    own_heading: float = 45.0
    target_position: List[float] = field(default_factory=lambda: [700.0, 500.0, 100.0])
    target_heading: float = 225.0
    enemy_count: int = 1
    
    # Flags
    override_scenario_duration: bool = False
    
    # Camera
    camera_fov: float = 60.0
    camera_resolution: tuple = (640, 480)


class SimulationCore:
    """
    Headless simulation engine.
    
    Runs physics at physics_dt (default 60Hz).
    Runs perception at perception_fps (default 30Hz).
    Produces frame-by-frame state for logging.
    
    Phase 1: Supports scenario YAML and target maneuver behaviors.
    Phase 2: Noise models and latency simulation.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Create RNG for determinism
        self.rng = np.random.default_rng(config.seed)
        self._seed_all(config.seed)
        
        # Scenario loader
        self.scenario_loader = ScenarioLoader()
        self.scenario_def: Optional[ScenarioDefinition] = None
        
        # Target behavior controllers
        self.target_behaviors: Dict[str, TargetManeuverController] = {}
        
        # Core components
        self.world = SimulationWorld({
            'physics_hz': int(1.0 / config.physics_dt),
            'time_scale': 1.0
        })
        
        self.camera = SimulatedCamera({
            'fov': config.camera_fov,
            'resolution': list(config.camera_resolution),
            'fps': config.perception_fps
        })
        
        # Pass seeded RNG to detector for determinism
        self.detector = SimulationDetector(rng=self.rng)
        
        self.tracker = TargetTracker({})
        
        # LockOnSystem requires LockConfig, not a dict
        from src.vision.lock_on import LockConfig
        lock_cfg = LockConfig(
            required_continuous_seconds=1.0,
            margin_horizontal=0.5,
            margin_vertical=0.5
        )
        self.lock_on = LockOnSystem(lock_cfg)
        # Note: frame_size is passed to validate_lock_candidate() at runtime, not set here
        
        self.server = CompetitionServerSimulator({})
        self.server.register_team(config.team_id)
        
        # State
        self.frame_id = 0
        self.perception_accumulator = 0.0
        self.perception_interval = 1.0 / config.perception_fps
        
        # Last perception results (persisted between perception ticks)
        self._last_detections: List[Dict] = []
        self._last_tracks: List[Dict] = []
        
        # Phase 2: Noise model (None = no noise)
        self.noise_model: Optional[DetectionNoiseModel] = None
        self.noise_config_dict: Dict = {}
        
        # Phase 2: Delay buffers (None = no delay)
        self.detection_delay: Optional[DelayedStream] = None
        self.tracking_delay: Optional[DelayedStream] = None
        self.latency_config_dict: Dict = {}
        
        # Setup scenario
        self._setup_scenario()
        
    def _seed_all(self, seed: int):
        """Set all random seeds for determinism."""
        random.seed(seed)
        np.random.seed(seed)
        
    def _setup_scenario(self):
        """Setup the simulation scenario from YAML or defaults."""
        self.world.reset()
        self.target_behaviors.clear()
        
        # Try to load scenario YAML
        try:
            self.scenario_def = self.scenario_loader.load(self.config.scenario)
            self._setup_from_definition(self.scenario_def)
        except ValueError:
            # Fall back to legacy hardcoded setup
            self._setup_legacy()
            
    def _setup_from_definition(self, scenario: ScenarioDefinition):
        """Setup from ScenarioDefinition (YAML loaded)."""
        # Override duration if specified in scenario AND not overridden by CLI
        if scenario.duration and not self.config.override_scenario_duration:
            self.config.duration = scenario.duration
            
        # Override camera if specified
        if scenario.camera_fov:
            self.config.camera_fov = scenario.camera_fov
            self.detector.fov = scenario.camera_fov
        if scenario.camera_resolution:
            self.config.camera_resolution = scenario.camera_resolution
            # Note: frame_size is passed to validate_lock_candidate() at runtime
            
        # Override perception FPS if specified
        if scenario.perception_fps:
            self.config.perception_fps = scenario.perception_fps
            self.perception_interval = 1.0 / scenario.perception_fps
            
        # Spawn own UAV
        own = scenario.own_uav
        self.world.spawn_uav(
            uav_id='player',
            team='blue',
            position=own.position,
            heading=own.heading,
            is_player=True
        )
        
        # Spawn targets with behaviors
        for target_def in scenario.targets:
            uav = self.world.spawn_uav(
                uav_id=target_def.id,
                team=target_def.team,
                position=target_def.position,
                heading=target_def.heading
            )
            
            # Create behavior controller with dedicated RNG from our seeded generator
            behavior_seed = int(self.rng.integers(0, 2**31))
            behavior_rng = np.random.default_rng(behavior_seed)
            
            controller = TargetManeuverController(
                pattern=target_def.behavior,
                params=target_def.behavior_params,
                rng=behavior_rng
            )
            self.target_behaviors[target_def.id] = controller
            
        # Phase 2: Setup noise model if configured
        if scenario.noise_config:
            noise_cfg = NoiseConfig.from_dict(scenario.noise_config)
            noise_cfg.frame_width = self.config.camera_resolution[0]
            noise_cfg.frame_height = self.config.camera_resolution[1]
            
            # Create dedicated RNG for noise (seeded from main rng)
            noise_seed = int(self.rng.integers(0, 2**31))
            noise_rng = np.random.default_rng(noise_seed)
            
            self.noise_model = DetectionNoiseModel(noise_cfg, noise_rng)
            self.noise_config_dict = noise_cfg.to_dict()
            
        # Phase 2: Setup latency buffers if configured
        if scenario.latency_config:
            latency_cfg = LatencyConfig.from_dict(scenario.latency_config)
            perception_dt_ms = self.perception_interval * 1000.0
            delay_frames = latency_cfg.compute_delay_frames(perception_dt_ms)
            
            if delay_frames['detection_delay_frames'] > 0:
                self.detection_delay = DelayedStream(delay_frames['detection_delay_frames'])
            if delay_frames['tracking_delay_frames'] > 0:
                self.tracking_delay = DelayedStream(delay_frames['tracking_delay_frames'])
                
            self.latency_config_dict = latency_cfg.to_dict()
            
    def _setup_legacy(self):
        """Legacy setup for backward compatibility."""
        # Spawn own UAV
        self.world.spawn_uav(
            uav_id='player',
            team='blue',
            position=self.config.own_position,
            heading=self.config.own_heading,
            is_player=True
        )
        
        # Spawn target(s) - no behaviors (static)
        if self.config.scenario == "default" or self.config.scenario == "single_target":
            self.world.spawn_uav(
                uav_id='target_01',
                team='red',
                position=self.config.target_position,
                heading=self.config.target_heading
            )
        elif self.config.scenario == "multi_target":
            for i in range(self.config.enemy_count):
                offset_x = 200 + i * 100
                self.world.spawn_uav(
                    uav_id=f'target_{i:02d}',
                    team='red',
                    position=[
                        self.config.own_position[0] + offset_x,
                        self.config.own_position[1] + (i % 2) * 50,
                        self.config.own_position[2] + (i % 3) * 20
                    ],
                    heading=180 + i * 10
                )
                
    def step(self, controls: Dict = None) -> SimulationState:
        """
        Advance simulation by one physics step.
        
        Args:
            controls: Optional control inputs for player UAV
                     {aileron, elevator, rudder, throttle}
        
        Returns:
            SimulationState snapshot
        """
        dt = self.config.physics_dt
        
        # Apply controls to player
        player = self.world.get_player_uav()
        if player and not player.is_crashed:
            # Pass detections to autopilot if in COMBAT mode
            # Assumes player has 'autopilot' attribute, typically handled in high-level main
            # or we need to ensure player object has access to it.
            # FixedWingUAV doesn't have autopilot instance by default, it's usually external.
            # However, if we are in headless mode, who runs the autopilot?
            # 'controls_fn' takes state and returns controls.
            # So the caller (run method) should handle this.
            
            # Let's check run() method below.
            if controls:
                player.set_controls(**controls, dt=dt)
            
        # Update target behaviors
        self._update_target_behaviors(dt)
            
        # Physics update
        self.world.update(dt)
        
        # Perception update (at perception_fps)
        self.perception_accumulator += dt
        run_perception = self.perception_accumulator >= self.perception_interval
        
        if run_perception:
            self.perception_accumulator -= self.perception_interval
            self._run_perception(player)
            
            
        # Lock-on update (uses tracks, not detections)
        # LockOnStateMachine.update signature: (tracks, sim_time, dt)
        lock_state = self.lock_on.update(
            self._last_tracks,
            self.world.time,
            dt
        )
        
        # Report valid locks at SUCCESS state (not just is_valid)
        if lock_state.state == LockState.SUCCESS and lock_state.is_valid:
            lock_data = {
                'target_id': lock_state.target_id,
                'lock_time': lock_state.lock_time,
                'end_time': lock_state.lock_end_time
            }
            self.server.report_lock_on(self.config.team_id, lock_data)
            self.lock_on.reset()
                
        # Update server with positions
        self._update_server_positions()
        
        # Build state snapshot
        state = self._build_state(player, lock_state)
        self.frame_id += 1
        
        return state
        
    def _update_target_behaviors(self, dt: float):
        """Update all target UAV behaviors."""
        player = self.world.get_player_uav()
        own_state = None
        if player:
            own_state = {
                'position': player.get_position().tolist(),
                'velocity': player.state.velocity.tolist(),
                'heading': player.get_heading_degrees()
            }
            
        for target_id, controller in self.target_behaviors.items():
            uav = self.world.get_uav(target_id)
            if uav is None or uav.is_crashed:
                continue
                
            target_state = {
                'position': uav.get_position().tolist(),
                'velocity': uav.state.velocity.tolist(),
                'heading': uav.get_heading_degrees(),
                'speed': uav.get_speed()
            }
            
            # Get maneuver command
            cmd = controller.update(dt, target_state, own_state)
            
            # Apply control surfaces
            uav.set_controls(
                aileron=cmd.aileron,
                elevator=cmd.elevator,
                rudder=cmd.rudder,
                throttle=cmd.throttle,
                dt=dt
            )
        
    def _run_perception(self, player):
        """Run perception pipeline with Phase 2 noise and latency."""
        if not player or player.is_crashed:
            self._last_detections = []
            self._last_tracks = []
            return
            
        # Get enemy states for detection
        enemy_states = self.world.get_uav_states_for_detection(player.id)
        
        # Set detector world state
        self.detector.set_world_state(
            uavs=enemy_states,
            camera_pos=player.get_camera_position(),
            camera_orient=player.get_orientation()
        )
        
        # Generate synthetic frame and detect
        frame = self.camera.generate_synthetic_frame(
            enemy_states,
            player.get_camera_position(),
            player.get_orientation(),
            player.state.velocity
        )
        
        raw_detections = self.detector.detect(frame)
        
        # Phase 2: Apply noise model
        if self.noise_model:
            raw_detections = self.noise_model.apply(raw_detections)
            
        # Phase 2: Apply detection delay
        if self.detection_delay:
            delayed_detections = self.detection_delay.push(raw_detections)
            raw_detections = delayed_detections if delayed_detections is not None else []
            
        self._last_detections = raw_detections
        
        # Update tracker
        raw_tracks = self.tracker.update(self._last_detections)
        
        # Phase 2: Apply tracking delay
        if self.tracking_delay:
            delayed_tracks = self.tracking_delay.push(raw_tracks)
            self._last_tracks = delayed_tracks if delayed_tracks is not None else []
        else:
            self._last_tracks = raw_tracks
        
    def _update_server_positions(self):
        """Update competition server with UAV positions."""
        positions = {}
        for uav_id, uav in self.world.uavs.items():
            if not uav.is_crashed:
                positions[uav_id] = {
                    'position': uav.get_position().tolist(),
                    'team': uav.team
                }
        self.server.update_uav_positions(positions)
        
    def _build_state(self, player, lock_state) -> SimulationState:
        """Build immutable state snapshot."""
        # Own state
        own_state = {}
        if player:
            own_state = {
                'position': player.get_position().tolist(),
                'velocity': player.state.velocity.tolist(),
                'orientation': player.get_orientation().tolist(),
                'speed': float(player.get_speed()),
                'altitude': float(player.get_altitude()),
                'heading': float(player.get_heading_degrees()),
                'is_crashed': player.is_crashed
            }
            
        # Targets with behavior info
        targets = []
        for uav_id, uav in self.world.uavs.items():
            if uav_id != self.world.player_uav_id:
                target_info = {
                    'id': uav_id,
                    'position': uav.get_position().tolist(),
                    'velocity': uav.state.velocity.tolist(),
                    'heading': float(uav.get_heading_degrees()),
                    'team': uav.team
                }
                # Add behavior info if available
                if uav_id in self.target_behaviors:
                    target_info['behavior'] = self.target_behaviors[uav_id].get_info()
                targets.append(target_info)
                
        # Lock state - use actual LockStatus fields
        is_locked = lock_state.state in (LockState.LOCKING, LockState.SUCCESS)
        lock_dict = {
            'locked': is_locked,
            'valid': lock_state.is_valid,
            'target_id': lock_state.target_id,
            'duration': lock_state.lock_time if is_locked else None,
            'dx': lock_state.dx if is_locked else None,
            'dy': lock_state.dy if is_locked else None
        }
            
        # Score
        score_data = self.server.get_score(self.config.team_id)
        score = {
            'total': score_data['total_score'] if score_data else 0,
            'correct_locks': score_data['correct_locks'] if score_data else 0,
            'incorrect_locks': score_data['incorrect_locks'] if score_data else 0
        }
        
        return SimulationState(
            t=self.world.time,
            frame_id=self.frame_id,
            scenario=self.config.scenario,
            own_state=own_state,
            targets=targets,
            detections=[d.copy() if isinstance(d, dict) else d for d in self._last_detections],
            tracks=[{'id': t.id, 'bbox': t.bbox, 'age': t.age} 
                    for t in self._last_tracks] if self._last_tracks else [],
            lock=lock_dict,
            score=score,
            noise=self.noise_config_dict if self.noise_config_dict else None,
            latency=self.latency_config_dict if self.latency_config_dict else None,
        )
        
    def run(self, controls_fn=None) -> List[SimulationState]:
        """
        Run complete simulation.
        
        Args:
            controls_fn: Optional callable(state, t) -> controls dict
                        For autopilot or scripted control
        
        Returns:
            List of all SimulationState snapshots
        """
        states = []
        
        while self.world.time < self.config.duration:
            # Get controls
            controls = None
            if controls_fn:
                controls = controls_fn(states[-1] if states else None, self.world.time)
            else:
                # Default: maintain straight flight
                controls = {'throttle': 0.7, 'aileron': 0, 'elevator': 0, 'rudder': 0}
                
            state = self.step(controls)
            states.append(state)
            
        return states
    
    def run_streaming(self, controls_fn=None):
        """
        Run simulation with generator-based streaming output.
        
        Memory-efficient alternative to run() for long simulations.
        Yields states one at a time instead of collecting all in memory.
        
        Args:
            controls_fn: Optional callable(state, t) -> controls dict
                        For autopilot or scripted control
        
        Yields:
            SimulationState snapshots one at a time
            
        Example:
            for state in sim.run_streaming():
                logger.log_frame(state)
                if state.lock['valid']:
                    print(f"Valid lock at t={state.t:.2f}s")
        """
        prev_state = None
        
        while self.world.time < self.config.duration:
            # Get controls
            controls = None
            if controls_fn:
                controls = controls_fn(prev_state, self.world.time)
            else:
                # Default: maintain straight flight
                controls = {'throttle': 0.7, 'aileron': 0, 'elevator': 0, 'rudder': 0}
                
            state = self.step(controls)
            prev_state = state
            yield state
        
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.world.time
        
    def get_score(self) -> Dict:
        """Get current score."""
        return self.server.get_score(self.config.team_id)
        
    def get_scenario_info(self) -> Dict:
        """Get loaded scenario information."""
        if self.scenario_def:
            return {
                'name': self.scenario_def.name,
                'description': self.scenario_def.description,
                'duration': self.scenario_def.duration,
                'target_count': len(self.scenario_def.targets),
                'behaviors': [t.behavior for t in self.scenario_def.targets]
            }
        return {'name': self.config.scenario, 'description': 'Legacy scenario'}
