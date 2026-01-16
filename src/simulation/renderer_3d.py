"""
3D Renderer - Panda3D ile gerçekçi görselleştirme (v2)

Stabil versiyon - gerçekçi İHA modeli ile.
"""

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    Point3, Vec3, Vec4, 
    AmbientLight, DirectionalLight,
    WindowProperties, CardMaker,
    TextNode, LineSegs,
    ClockObject
)
from direct.gui.OnscreenText import OnscreenText
import numpy as np
import sys

# Detaylı İHA modeli
from .uav_model_3d import DetailedUAVModel
# Arazi
from .terrain import TerrainGenerator

# Autonomy & Vision
from src.uav.autopilot import Autopilot, AutopilotMode
from src.vision.detector import SimulationDetector
from src.vision.tracker import TargetTracker


class Renderer3D(ShowBase):
    """Panda3D tabanlı 3D İHA Simülasyonu"""
    
    def __init__(self, world=None):
        ShowBase.__init__(self)
        
        self.sim_world = world
        
        # Pencere ayarları
        props = WindowProperties()
        props.setTitle("Savaşan İHA 3D Simülasyonu")
        props.setSize(1280, 720)
        self.win.requestProperties(props)
        
        # Arka plan (gökyüzü)
        self.setBackgroundColor(0.4, 0.6, 0.9, 1)
        
        # FPS sınırı
        globalClock = ClockObject.getGlobalClock()
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(60)
        
        # İHA node'ları
        self.uav_nodes = {}
        self.player_node = None
        
        # Kamera
        self.camera_mode = "follow"
        self.camera_distance = 15
        self.camera_height = 4
        self.orbit_angle = 0
        
        # Kontrol girdileri
        self.inputs = {
            'pitch': 0, 'roll': 0, 'yaw': 0,
            'throttle_up': False, 'throttle_down': False
        }
        
        # Sahne kur
        self._setup_scene()
        self._setup_controls()
        self._setup_hud()
        
        # Autonomy & Vision Components
        self.autopilot = Autopilot()
        self.detector = SimulationDetector(rng=np.random.default_rng(42))
        self.tracker = TargetTracker()
        
        # Güncelleme görevi
        self.taskMgr.add(self._update, "MainUpdate")
        
    def _setup_scene(self):
        """Sahne kurulumu"""
        # Işıklar
        ambient = AmbientLight('ambient')
        ambient.setColor(Vec4(0.5, 0.5, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(ambient))
        
        sun = DirectionalLight('sun')
        sun.setColor(Vec4(1, 0.95, 0.8, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self.render.setLight(sun_np)
        
        # Arazi (dağlar, pist, binalar, ağaçlar)
        self.terrain = TerrainGenerator.create_terrain(self.render)
        
    def _setup_controls(self):
        """Kontroller"""
        # Uçuş
        self.accept('w', self._set, ['pitch', -1])
        self.accept('w-up', self._set, ['pitch', 0])
        self.accept('s', self._set, ['pitch', 1])
        self.accept('s-up', self._set, ['pitch', 0])
        self.accept('a', self._set, ['roll', -1])
        self.accept('a-up', self._set, ['roll', 0])
        self.accept('d', self._set, ['roll', 1])
        self.accept('d-up', self._set, ['roll', 0])
        self.accept('q', self._set, ['yaw', -1])
        self.accept('q-up', self._set, ['yaw', 0])
        self.accept('e', self._set, ['yaw', 1])
        self.accept('e-up', self._set, ['yaw', 0])
        
        # Gaz
        self.accept('shift', self._set, ['throttle_up', True])
        self.accept('shift-up', self._set, ['throttle_up', False])
        self.accept('control', self._set, ['throttle_down', True])
        self.accept('control-up', self._set, ['throttle_down', False])
        
        # Kamera
        self.accept('c', self._next_camera)
        self.accept('1', self._cam, ['follow'])
        self.accept('2', self._cam, ['cockpit'])
        self.accept('3', self._cam, ['orbit'])
        
        # Autopilot
        self.accept('p', self._toggle_autopilot)
        
        # Çıkış
        self.accept('escape', sys.exit)
        
    def _set(self, key, val):
        self.inputs[key] = val
        
    def _next_camera(self):
        modes = ['follow', 'cockpit', 'orbit']
        i = modes.index(self.camera_mode)
        self.camera_mode = modes[(i + 1) % 3]
        
    def _cam(self, mode):
        self.camera_mode = mode
        
    def _toggle_autopilot(self):
        if self.autopilot.enabled:
            self.autopilot.disable()
            print("Autopilot: DEVRE DIŞI")
        else:
            self.autopilot.enable()
            self.autopilot.set_mode(AutopilotMode.COMBAT)
            print("Autopilot: COMBAT MODU")
            
    def _setup_hud(self):
        """HUD - Enhanced Phase 3"""
        self.hud = {}
        self.hud['title'] = OnscreenText("Savaşan İHA 3D", pos=(-1.3, 0.9), 
                                          scale=0.06, fg=(1,1,1,1), align=TextNode.ALeft)
        self.hud['speed'] = OnscreenText("Hız: -- m/s", pos=(-1.3, 0.82),
                                          scale=0.05, fg=(1,1,1,1), align=TextNode.ALeft)
        self.hud['alt'] = OnscreenText("İrtifa: -- m", pos=(-1.3, 0.74),
                                        scale=0.05, fg=(1,1,1,1), align=TextNode.ALeft)
        self.hud['hdg'] = OnscreenText("Yön: --°", pos=(-1.3, 0.66),
                                        scale=0.05, fg=(1,1,1,1), align=TextNode.ALeft)
        # Score display (yeni)
        self.hud['score'] = OnscreenText("Skor: 0", pos=(-1.3, 0.58),
                                          scale=0.05, fg=(0, 1, 0.6, 1), align=TextNode.ALeft)
        # Detection/Track count (yeni)
        self.hud['det'] = OnscreenText("Det: 0 | Track: 0", pos=(-1.3, 0.50),
                                        scale=0.04, fg=(0.8, 0.8, 0.4, 1), align=TextNode.ALeft)
        # Lock state (yeni)
        self.hud['lock'] = OnscreenText("", pos=(0, -0.8),
                                         scale=0.06, fg=(1, 0, 0, 1), align=TextNode.ACenter)
        self.hud['lock_target'] = OnscreenText("", pos=(0, -0.87),
                                                scale=0.04, fg=(1, 0.8, 0, 1), align=TextNode.ACenter)
        
        self.hud['cam'] = OnscreenText("Kamera: follow", pos=(1.3, 0.9),
                                        scale=0.04, fg=(1,1,0.5,1), align=TextNode.ARight)
        self.hud['help'] = OnscreenText("W/S:Pitch A/D:Roll Q/E:Yaw Shift/Ctrl:Gaz C:Kamera ESC:Çıkış",
                                         pos=(0, -0.95), scale=0.04, fg=(1,1,1,0.7), align=TextNode.ACenter)
        
        # Lock state tracking
        self.current_lock_state = None
        self.current_score = 0
        self.detection_count = 0
        self.track_count = 0
    
    def update_lock_state(self, lock_state: dict = None, score: int = 0, 
                          det_count: int = 0, track_count: int = 0):
        """Update HUD with lock state and detection info"""
        self.current_score = score
        self.detection_count = det_count
        self.track_count = track_count
        
        # Update score
        self.hud['score'].setText(f"Skor: {score}")
        
        # Update detection count
        self.hud['det'].setText(f"Det: {det_count} | Track: {track_count}")
        
        # Update lock state
        if lock_state and lock_state.get('is_locked'):
            is_valid = lock_state.get('is_valid', False)
            target_id = lock_state.get('target_id', '?')
            duration = lock_state.get('duration', 0)
            
            if is_valid:
                self.hud['lock'].setText("KİLİTLİ!")
                self.hud['lock'].setFg((1, 0, 0, 1))
            else:
                self.hud['lock'].setText(f"Kilitlenme: {duration:.1f}s")
                self.hud['lock'].setFg((1, 0.8, 0, 1))
                
            dx = lock_state.get('dx', 0) or 0
            dy = lock_state.get('dy', 0) or 0
            self.hud['lock_target'].setText(f"Hedef: {target_id} | Δx:{dx:+.1f} Δy:{dy:+.1f}")
        else:
            self.hud['lock'].setText("")
            self.hud['lock_target'].setText("")
        
    def _create_uav(self, uav_id, team, is_player):
        """Gerçekçi İHA modeli oluştur"""
        node = DetailedUAVModel.create(
            self.render, 
            name=uav_id,
            team='green' if is_player else team,
            scale=1.5
        )
        self.uav_nodes[uav_id] = node
        if is_player:
            self.player_node = node
        return node
        
    def _update(self, task):
        """Ana güncelleme"""
        dt = globalClock.getDt()
        
        if self.sim_world:
            # Oyuncu kontrolü
            player = self.sim_world.get_player_uav()
            
            # Vision Pipeline (Detect -> Track)
            if player:
                # 1. Detection
                enemy_states = self.sim_world.get_uav_states_for_detection(player.id)
                self.detector.set_world_state(
                    uavs=enemy_states,
                    camera_pos=player.get_camera_position(),
                    camera_orient=player.get_orientation()
                )
                # Note: Renderer uses Panda3D logic, so we don't have a pixel frame easily.
                # SimulationDetector handles this by using world state directly if detect() is called with dummy or no frame.
                # Wait, SimulationDetector.detect() expects a frame but ignores it.
                detections = self.detector.detect(None)
                
                # 2. Tracking
                tracks = self.tracker.update(detections)
                self.autopilot.set_combat_detections(tracks)
                
                # HUD Update (for debug)
                self.update_lock_state(det_count=len(detections), track_count=len(tracks))

            if player and not player.is_crashed:
                # Autopilot Control
                controls = None
                if self.autopilot.enabled:
                    # Build state dict for autopilot
                    state = {
                        'position': player.get_position(),
                        'velocity': player.state.velocity,
                        'altitude': player.get_altitude(),
                        'speed': player.get_speed(),
                        'heading': np.degrees(player.get_heading()) # Autopilot expects degrees?
                        # Note: Autopilot._waypoint_follow converts heading to radians: np.radians(uav_state['heading'])
                        # So we must pass degrees.
                    }
                    # Update autopilot
                    controls = self.autopilot.update(state, dt)
                
                # Check manual override or fallback
                if not controls:
                    th = player.controls.throttle
                    if self.inputs['throttle_up']:
                         th = min(1.0, th + 0.5 * dt)
                    if self.inputs['throttle_down']:
                         th = max(0.0, th - 0.5 * dt)
                        
                    controls = {
                        'aileron': self.inputs['roll'],
                        'elevator': self.inputs['pitch'],
                        'rudder': self.inputs['yaw'],
                        'throttle': th
                    }
                    
                player.set_controls(**controls)
            
            # Fizik
            self.sim_world.update(dt)
            
            # Model güncelle
            for uid, uav in self.sim_world.uavs.items():
                if uid not in self.uav_nodes:
                    self._create_uav(uid, uav.team, uid == self.sim_world.player_uav_id)
                    
                pos = uav.get_position()
                orient = uav.get_orientation()
                
                node = self.uav_nodes[uid]
                node.setPos(pos[0], pos[1], pos[2])
                
                # HPR: Heading(yaw), Pitch, Roll
                node.setHpr(
                    np.degrees(orient[2]),   # yaw
                    np.degrees(orient[1]),   # pitch
                    np.degrees(orient[0])    # roll
                )
            
            # HUD
            if player:
                self.hud['speed'].setText(f"Hız: {player.get_speed():.1f} m/s")
                self.hud['alt'].setText(f"İrtifa: {player.get_altitude():.0f} m")
                self.hud['hdg'].setText(f"Yön: {player.get_heading_degrees():.0f}°")
                self.hud['cam'].setText(f"Kamera: {self.camera_mode}")
                
            # Kamera
            self._update_camera(player)
            
        return Task.cont
        
    def _update_camera(self, player):
        """Kamera güncellemesi"""
        if not player or not self.player_node:
            return
            
        pos = player.get_position()
        yaw = player.get_heading()
        
        if self.camera_mode == 'follow':
            cx = pos[0] - self.camera_distance * np.cos(yaw)
            cy = pos[1] - self.camera_distance * np.sin(yaw)
            cz = pos[2] + self.camera_height
            self.camera.setPos(cx, cy, cz)
            self.camera.lookAt(self.player_node)
            
        elif self.camera_mode == 'cockpit':
            fwd = player.get_forward_vector()
            cp = pos + fwd * 3
            self.camera.setPos(cp[0], cp[1], cp[2])
            look = pos + fwd * 50
            self.camera.lookAt(Point3(look[0], look[1], look[2]))
            
        elif self.camera_mode == 'orbit':
            self.orbit_angle += 30 * globalClock.getDt()
            r = 25
            cx = pos[0] + r * np.cos(np.radians(self.orbit_angle))
            cy = pos[1] + r * np.sin(np.radians(self.orbit_angle))
            cz = pos[2] + 8
            self.camera.setPos(cx, cy, cz)
            self.camera.lookAt(self.player_node)
