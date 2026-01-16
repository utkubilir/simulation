"""
CLI Entrypoint for TEKNOFEST Savaşan İHA Sim vNext

Usage:
    python -m scripts.run --mode ui --scenario easy_lock --seed 42
    python -m scripts.run --mode headless --scenario easy_lock --duration 30
"""

import argparse
import sys
from pathlib import Path
import uuid
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.world import SimulationWorld
from src.simulation.renderer import Renderer
from src.simulation.camera import SimulatedCamera
from src.uav.controller import FlightController, KeyboardMapper
from src.uav.autopilot import Autopilot, AutopilotMode
from src.vision.detector import SimulationDetector
from src.vision.tracker import TargetTracker
from src.vision.lock_on import LockOnStateMachine, LockConfig, LockState
from src.logging.frame_logger import FrameLogger, FrameData
from src.simulation.air_defense import AirDefenseManager


def _available_scenarios() -> list[str]:
    scenarios_dir = Path(__file__).parent.parent / 'scenarios'
    return sorted([p.stem for p in scenarios_dir.glob('*.yaml')])


def _validate_scenario_config(scenario_name: str, config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError(f"Scenario '{scenario_name}' config must be a mapping.")

    for key in ('simulation', 'camera', 'player', 'lock', 'tracking', 'detection'):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"Scenario '{scenario_name}' key '{key}' must be a mapping.")

    enemies = config.get('enemies', [])
    if enemies and not isinstance(enemies, list):
        raise ValueError(f"Scenario '{scenario_name}' key 'enemies' must be a list.")

    if isinstance(enemies, list):
        for idx, enemy in enumerate(enemies):
            if not isinstance(enemy, dict):
                raise ValueError(
                    f"Scenario '{scenario_name}' enemy at index {idx} must be a mapping."
                )

    return config


def _normalize_camera_resolution(camera_config: dict) -> tuple[int, int]:
    resolution = camera_config.get('resolution')
    if resolution:
        return int(resolution[0]), int(resolution[1])
    return int(camera_config.get('width', 640)), int(camera_config.get('height', 480))


class SimulationRunner:
    """Unified simulation runner for UI and headless modes"""
    
    def __init__(self, config: dict, mode: str = 'ui'):
        self.config = config
        self.mode = mode
        self.seed = config.get('seed', 42)
        self.scenario = config.get('scenario', 'default')
        self.duration = config.get('duration', 60.0)
        
        # Deterministic RNG
        self.rng = np.random.default_rng(self.seed)
        
        # Run ID for outputs
        self.run_id = config.get('run_id') or f"{self.scenario}_{self.seed}_{uuid.uuid4().hex[:8]}"
        
        # Core components
        self.world = SimulationWorld(config.get('simulation', {}))
        det_config_path = config.get('detection', {}).get('config_path')
        self.detector = SimulationDetector(config_path=det_config_path, rng=self.rng)
        self.tracker = TargetTracker(config.get('tracking', {}))

        camera_config = config.get('camera', {})
        self.camera_resolution = _normalize_camera_resolution(camera_config)
        
        # Lock-on with competition settings
        lock_config = LockConfig(
            window_seconds=config.get('lock', {}).get('window_seconds', 5.0),
            required_continuous_seconds=config.get('lock', {}).get('success_duration', 4.0),
            size_threshold=config.get('lock', {}).get('size_threshold', 0.06),
            margin_horizontal=config.get('lock', {}).get('margin_h', 0.5),
            margin_vertical=config.get('lock', {}).get('margin_v', 0.5),
            min_confidence=config.get('lock', {}).get('min_confidence', 0.5),
            frame_width=self.camera_resolution[0],
            frame_height=self.camera_resolution[1]
        )
        self.lock_sm = LockOnStateMachine(lock_config)
        
        # Logger
        output_dir = Path(config.get('output_dir', 'results'))
        self.logger = FrameLogger(
            output_dir=output_dir,
            run_id=self.run_id,
            seed=self.seed,
            scenario=self.scenario,
            config=config
        )
        
        # UI components (only for ui mode)
        self.renderer = None
        self.camera = None
        self.controller = None
        self.keyboard = None
        self.autopilot = None
        
        if mode == 'ui':
            radar_heading_mode = config.get('ui', {}).get('radar_heading_mode', 'heading_up')
            self.renderer = Renderer(
                width=config.get('simulation', {}).get('window_width', 1280),
                height=config.get('simulation', {}).get('window_height', 720),
                radar_heading_mode=radar_heading_mode
            )
            self.camera = SimulatedCamera(config.get('camera', {}))
            self.controller = FlightController()
            self.keyboard = KeyboardMapper()
            self.autopilot = Autopilot()
            
        # State
        self.running = False
        self.sim_time = 0.0
        self.frame_id = 0
        
        # Enable autopilot by default in ALL modes (tam otonom sistem)
        self.use_autopilot = True
        
        # Make sure autopilot is initialized and enabled
        if self.autopilot is None:
            self.autopilot = Autopilot()
        self.autopilot.set_camera_frame_size(self.camera_resolution)
        
        # Always start in COMBAT mode (otonom kilitlenme)
        self.autopilot.set_mode(AutopilotMode.COMBAT)
        self.autopilot.enable()
            
        # Observer Mode
        self.camera_target_id = None  # None = Player, otherwise ID string
        
        # Air Defense Manager (Şartname 6.3)
        self.air_defense = AirDefenseManager()
        
        # Load air defense zones from scenario
        if 'air_defense' in self.config:
            self.air_defense.load_from_scenario(self.config)
        
    def setup_scenario(self, scenario_config: dict = None):
        """Setup scenario from config"""
        self.world.reset()
        
        # 1. Spawn Player
        player_config = self.config.get('player', {})
        position = player_config.get('position', [500, 500, 100])
        heading = player_config.get('heading', 45)
        speed = player_config.get('speed', 25.0)
        
        player = self.world.spawn_uav(
            uav_id='player',
            team='blue',
            position=position,
            heading=heading,
            is_player=True
        )
        # Set initial speed if supported (FixedWingUAV initializes with velocity based on heading * speed)
        # We might need to manually set velocity if spawn_uav doesn't take speed.
        # FixedWingUAV.reset takes position and heading, but maybe not speed?
        # Let's check FixedWingUAV later. For now, assume speed needs setting.
        # Assuming simple physics:
        heading_rad = np.radians(heading)
        velocity = np.array([
            np.cos(heading_rad) * speed,
            np.sin(heading_rad) * speed,
            0.0
        ])
        player.state.velocity = velocity
        
        # 2. Spawn Enemies
        enemies = self.config.get('enemies', [])
        if not enemies:
            # Fallback for empty scenarios
            enemy_count = self.config.get('enemy_count', 0)
            if enemy_count > 0:
                self.world.spawn_enemy_uavs(count=enemy_count)
        else:
            for i, enemy_conf in enumerate(enemies):
                eid = enemy_conf.get('id', f"enemy_{i}")
                pos = enemy_conf.get('position', [1500, 1500, 100])
                hdg = enemy_conf.get('heading', 225)
                spd = enemy_conf.get('speed', 20.0)
                behavior = enemy_conf.get('behavior', 'straight')
                
                enemy = self.world.spawn_uav(
                    uav_id=eid,
                    team='red',
                    position=pos,
                    heading=hdg,
                    is_player=False,
                    config={'behavior': behavior} # Pass behavior to config
                )
                
                # Set velocity
                hdg_rad = np.radians(hdg)
                vel = np.array([
                    np.cos(hdg_rad) * spd,
                    np.sin(hdg_rad) * spd,
                    0.0
                ])
                enemy.state.velocity = vel
                
                # Store extra behavior params if needed (FixedWingUAV might need update to handle 'behavior')
                enemy.behavior = behavior
                for k, v in enemy_conf.items():
                    if k not in ['id', 'position', 'heading', 'speed', 'behavior']:
                        setattr(enemy, k, v)
        
    def run(self):
        """Run simulation"""
        self.running = True
        self.setup_scenario()
        
        if self.mode == 'ui':
            self._run_ui()
        else:
            self._run_headless()
            
        # Finalize and write metrics
        metrics = self.logger.finalize()
        
        print(f"\n{'='*50}")
        print(f"📊 Run Complete: {self.run_id}")
        print(f"{'='*50}")
        print(f"  Duration: {metrics.duration:.1f}s")
        print(f"  Frames: {metrics.total_frames}")
        print(f"  Correct Locks: {metrics.correct_locks}")
        print(f"  Incorrect Locks: {metrics.incorrect_locks}")
        print(f"  Final Score: {metrics.final_score}")
        if metrics.time_to_first_success_lock:
            print(f"  Time to First Lock: {metrics.time_to_first_success_lock:.1f}s")
        print(f"{'='*50}\n")
        
        return metrics
        
    def _run_headless(self):
        """Run in headless mode (no GUI)"""
        dt = 1.0 / 60.0
        
        print(f"Running headless: {self.scenario} (seed={self.seed}, duration={self.duration}s)")
        
        while self.running and self.sim_time < self.duration:
            self._step(dt)
            self.sim_time += dt
            self.frame_id += 1
            
    def _run_ui(self):
        """Run in UI mode"""
        self.renderer.init()
        dt = 1.0 / 60.0
        
        print(f"\n{'='*50}")
        print(f"🎮 TEKNOFEST Savaşan İHA Sim vNext")
        print(f"{'='*50}")
        print(f"  Scenario: {self.scenario} | Seed: {self.seed}")
        print(f"  P: Toggle Autopilot | ESC: Exit")
        print(f"{'='*50}\n")
        
        try:
            while self.running:
                self._handle_events()
                self._step(dt)
                self._render()
                self.sim_time += dt
                self.frame_id += 1
                
        except KeyboardInterrupt:
            pass
        finally:
            self.renderer.close()
            
    def _cycle_camera_target(self, step: int):
        """Cycle through UVA targets for observer mode"""
        uavs = self.world.get_all_uavs()
        if not uavs: return
        
        # Sort by ID
        uavs.sort(key=lambda u: u.id)
        
        current_id = self.camera_target_id
        if current_id is None:
            # Default to player if exists
            player = self.world.get_player_uav()
            if player:
                current_id = player.id
        
        # Find index
        idx = 0
        for i, u in enumerate(uavs):
            if u.id == current_id:
                idx = i
                break
                
        # Cycle
        new_idx = (idx + step) % len(uavs)
        self.camera_target_id = uavs[new_idx].id
        print(f"👀 Observer: Watching {self.camera_target_id}")

    def restart_simulation(self):
        """Restart scenario and reset state"""
        print("🔄 Restarting Simulation...")
        self.setup_scenario()
        
        # Reset generic state
        self.sim_time = 0.0
        self.frame_id = 0
        self.world.is_paused = False
        
        # Reset specific components
        self.lock_sm.reset()
        if self.tracker:
            self.tracker.reset()
        
        # Re-enable autopilot if needed
        if self.use_autopilot and self.autopilot:
             self.autopilot.set_mode(AutopilotMode.COMBAT)
             self.autopilot.enable()

    def _handle_events(self):
        """Handle UI events"""
        events = self.renderer.handle_events()
        
        for event in events:
            # Observer commands
            if event.get('cmd') == 'next_uav':
                self._cycle_camera_target(1)
            elif event.get('cmd') == 'prev_uav':
                self._cycle_camera_target(-1)
                
            if event['type'] == 'quit':
                self.running = False
            elif event['type'] == 'keydown':
                key = event['key']
                
                # Global controls
                if key == 'escape':
                    self.running = False
                elif key == 'space':
                    if self.world.is_paused:
                        self.world.resume()
                        print("▶️ Resumed")
                    else:
                        self.world.pause()
                        print("⏸️ Paused")
                elif key == 'r':
                    self.restart_simulation()
                   
                # 'P' tuşu artık mod bilgisi gösterir (manuel geçiş kaldırıldı)
                elif key == 'p':
                    print(f"🤖 Autopilot: {self.autopilot.mode.value.upper()} Mode")
                    print("   ℹ️ Tam otonom mod aktif - manuel kontrol devre dışı")
                # M tuşu - mod bilgisi
                elif key == 'm':
                    print(f"📊 Mod Bilgisi:")
                    print(f"   Autopilot: {self.autopilot.mode.value}")
                    print(f"   Enabled: {self.autopilot.enabled}")
                    if hasattr(self.autopilot, 'combat_manager'):
                        print(f"   Combat State: {self.autopilot.combat_manager.state.value}")
                # NOT: W/A/S/D/Q/E uçuş kontrol tuşları kaldırıldı
                # İHA tamamen otonom kontrol altında
                        
            elif event['type'] == 'keyup':
                pass  # Manuel kontrol kaldırıldı
                
    def _step(self, dt: float):
        """Single simulation step"""
        player = self.world.get_player_uav()
        
        if player and not player.is_crashed:
            # Update controls
            if self.mode == 'ui':
                # Map keyboard inputs in UI mode
                inputs = self.keyboard.get_inputs() if self.keyboard else {}
                
            if self.use_autopilot:
                controls = self.autopilot.update(player.to_dict(), dt)
                if controls:
                    player.set_controls(**controls)
            elif self.mode == 'ui':
                # Manual control only in UI mode
                controls = self.controller.update(dt, inputs) if self.controller else {}
                player.set_controls(**controls)
                    
            # World update
            self.world.update(dt)
            
            # Determine view target (Player or Observer target)
            view_uav = player
            if self.camera_target_id:
                target = self.world.get_uav(self.camera_target_id)
                if target and not target.is_crashed:
                    view_uav = target
            
            # Vision pipeline (from viewpoint of view_uav)
            enemy_states = self.world.get_uav_states_for_detection(exclude_id=view_uav.id)
            
            # Update camera
            if self.camera:
                self.camera.update(dt, view_uav.state.velocity)
                camera_pos, camera_orient = self.camera.get_camera_pose(
                    view_uav.get_position(),
                    view_uav.get_orientation()
                )
            else:
                # Headless
                camera_pos = view_uav.get_camera_position()
                camera_orient = view_uav.get_orientation()
                
            self.detector.set_world_state(
                uavs=enemy_states,
                camera_pos=camera_pos,
                camera_orient=camera_orient
            )
            
            # Generate synthetic camera frame
            if self.camera and self.mode == 'ui':
                frame = self.camera.generate_synthetic_frame(
                    enemy_states, camera_pos, camera_orient, player.state.velocity
                )
            else:
                # Headless mod için minimal frame
                frame = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8)
            
            # Detect and track
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
            
            # Update lock state machine
            lock_status = self.lock_sm.update(tracks, self.sim_time, dt)
            
            # Air Defense Update (Şartname 6.3)
            ad_result = self.air_defense.update(
                self.sim_time,
                {'player': player.get_position()},
                dt
            )
            
            # Handle air defense warnings
            for warning in ad_result.get('warnings', []):
                print(warning)
                
            # Check if landing is required (30s limit exceeded)
            if ad_result.get('landing_required'):
                print("⚠️ 30 saniye sınırı aşıldı - İNİŞ GEREKLİ!")
                if self.autopilot:
                    self.autopilot.set_mode(AutopilotMode.RETURN_HOME)
                    
            # Provide avoidance heading to autopilot if in danger
            if ad_result.get('active_zones') and self.autopilot:
                avoidance_heading = self.air_defense.get_avoidance_heading(
                    player.get_position(),
                    np.degrees(player.state.heading)
                )
                if avoidance_heading is not None:
                    self.autopilot.set_avoidance_heading(avoidance_heading)
            
            # Pass tracks to autopilot
            if self.autopilot:
                self.autopilot.set_combat_detections(tracks)
            
            # Store for rendering
            self._last_frame = frame
            self._last_detections = detections
            self._last_tracks = tracks
            self._last_ad_result = ad_result
            
            # Log frame
            self._log_frame(player, enemy_states, detections, tracks, lock_status)
            
    def _log_frame(self, player, enemies, detections, tracks, lock_status):
        """Log frame data"""
        frame = FrameData(
            t=self.sim_time,
            frame_id=self.frame_id,
            own_state={
                'position': player.get_position().tolist(),
                'velocity': player.state.velocity.tolist(),
                'heading': player.get_orientation()[2],
                'altitude': player.get_position()[2],
                'speed': np.linalg.norm(player.state.velocity)
            },
            enemies=[
                {
                    'id': e.get('id'),
                    'position': e.get('position'),
                    'heading': e.get('heading', 0)
                }
                for e in enemies
            ],
            detections=[
                {
                    'id': d.get('world_id'),
                    'bbox': d.get('bbox'),
                    'center': d.get('center'),
                    'confidence': d.get('confidence')
                }
                for d in detections
            ],
            tracks=[
                {
                    'id': int(t.id),
                    'bbox': [float(x) for x in t.bbox],
                    'center': [float(x) for x in t.center],
                    'confidence': float(t.confidence),
                    'is_confirmed': bool(t.is_confirmed)
                }
                for t in tracks
            ],
            lock={
                'state': lock_status.state.value,
                'target_id': lock_status.target_id,
                'lock_time': lock_status.lock_time,
                'is_valid': lock_status.is_valid,
                'progress': lock_status.progress,
                'dx': lock_status.dx,
                'dy': lock_status.dy
            },
            score=self.lock_sm.get_score()
        )
        
        self.logger.log_frame(frame)
        
    def _render(self):
        """Render UI"""
        world_state = self.world.get_world_state()
        
        lock_status = self.lock_sm.get_status()
        lock_data = {
            'state': lock_status.state.value,
            'progress': lock_status.progress,
            'target_id': lock_status.target_id,
            'lock_time': lock_status.lock_time,
            'is_valid': lock_status.is_valid,
            'score': self.lock_sm.get_score()
        }
        
        player = self.world.get_player_uav()
        if player:
            pos = player.get_position()
            self.renderer.set_camera(pos[0], pos[1])
            
        # Get latest vision data from last step
        # Note: In a real threaded app, we'd need thread safety. 
        # Here step() and render() are sequential in the loop.
        
        # Use placeholders if _step hasn't run yet
        detections = getattr(self, '_last_detections', [])
        tracks = getattr(self, '_last_tracks', [])
        frame = getattr(self, '_last_frame', None)

        self.renderer.render(
            world_state=world_state, 
            lock_state=lock_data,
            sim_time=self.sim_time,
            scenario=self.scenario,
            seed=self.seed,
            camera_frame=frame,
            detections=detections,
            tracks=tracks,
            observer_target_id=self.camera_target_id,
            is_paused=self.world.is_paused
        )


def load_scenario(scenario_name: str) -> dict:
    """Load scenario configuration"""
    import yaml

    scenario_path = Path(__file__).parent.parent / 'scenarios' / f'{scenario_name}.yaml'
    if scenario_path.exists():
        with open(scenario_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
        return _validate_scenario_config(scenario_name, loaded)

    available = ", ".join(_available_scenarios()) or "none"
    raise FileNotFoundError(
        f"Scenario '{scenario_name}' not found. Available scenarios: {available}"
    )


def main():
    parser = argparse.ArgumentParser(description='TEKNOFEST Savaşan İHA Sim vNext')
    parser.add_argument('--mode', '-m', choices=['ui', 'headless'], default='ui',
                        help='Run mode')
    parser.add_argument('--scenario', '-s', type=str, default='default',
                        help='Scenario name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for determinism')
    parser.add_argument('--duration', '-d', type=float, default=60.0,
                        help='Simulation duration (seconds)')
    parser.add_argument('--output', '-o', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    try:
        scenario_config = load_scenario(args.scenario)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)
    
    # Merge with CLI args
    config = {
        **scenario_config,
        'seed': args.seed,
        'scenario': args.scenario,
        'duration': args.duration,
        'output_dir': args.output,
        'run_id': args.run_id
    }
    
    # Run simulation
    runner = SimulationRunner(config, mode=args.mode)
    metrics = runner.run()
    
    # Exit with success if at least one lock in easy scenarios
    if args.scenario == 'easy_lock' and metrics.correct_locks == 0:
        sys.exit(1)
        
    sys.exit(0)


if __name__ == '__main__':
    main()
