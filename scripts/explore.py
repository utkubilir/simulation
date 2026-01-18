"""
World Explorer Script

Bu script simülasyonu otopilot olmadan başlatır ve manuel uçuşa izin verir.
Dünyayı gezmek ve grafikleri test etmek için kullanılır.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import SimulationRunner, load_scenario
from src.uav.controller import FlightController, KeyboardMapper

class WorldExplorer(SimulationRunner):
    """
    Özel mod: Dünyayı gezmek için manuel kontrol.
    """
    def __init__(self, config):
        super().__init__(config, mode='ui')
        
        self.use_autopilot = False  # Disable autopilot
        self.controller = FlightController()
        self.keyboard = KeyboardMapper()
        
        # Kamera başlangıç pozisyonunu biraz yukarı al
        if self.camera:
            # Uçağın arkasında ve yukarısında
            pass 
        
    def _handle_events(self):
        """Klavye girdilerini manuel kontrolcüye yönlendir"""
        events = self.renderer.handle_events()
        
        for event in events:
            if event['type'] == 'quit':
                self.running = False
            
            # Klavye girdilerini Mapper'a ilet
            elif event['type'] == 'keydown':
                key = event['key']
                if key == 'escape':
                    self.running = False
                elif key == 'r':
                    self.restart_simulation()
                else:
                    self.keyboard.key_down(key)
                    
            elif event['type'] == 'keyup':
                key = event['key']
                self.keyboard.key_up(key)

    def _step(self, dt: float):
        """Simülasyon adımı - Manuel kontrol uygula"""
        # Debug heartbeat
        if self.frame_id % 60 == 0:
            print(f"🟢 Running... Frame: {self.frame_id} Time: {self.sim_time:.1f}s")
            
        player = self.world.get_player_uav()
        
        if player and not player.is_crashed:
            # 1. Klavye girdilerini al
            inputs = self.keyboard.get_inputs()
            
            # 2. Kontrolcüden uçuş komutlarını hesapla (Roll, Pitch, Yaw, Throttle)
            controls = self.controller.update(dt, inputs)
            
            # 3. Komutları uçağa uygula
            player.set_controls(**controls, dt=dt)

        # World update
        self.world.update(dt)
        
        # Kamera player'ı takip etsin
        if self.camera and player:
            # Generate synthetic frame for rendering
            # This handles camera update internally and returns the image
            enemy_states = self.world.get_uav_states_for_detection(player.id)
            
            self._last_frame = self.camera.generate_synthetic_frame(
                enemy_states,
                player.get_camera_position(),
                player.get_orientation(),
                player.state.velocity
            )
            
            # Debug: Frame içeriğini kontrol et
            if self.frame_id % 60 == 0:
                import numpy as np
                mean = np.mean(self._last_frame) if self._last_frame is not None else 0
                print(f"    📷 Frame Mean: {mean:.1f} | Shape: {self._last_frame.shape if self._last_frame is not None else 'None'}")
    
    def _run_ui(self):
        """Override: renderer.init() sonrası environment/arena bağla"""
        print(f"\n{'='*50}")
        print(f"🎮 TEKNOFEST Savaşan İHA Sim vNext")
        print(f"{'='*50}")
        print(f"  Scenario: {self.scenario} | Seed: {self.seed}")
        print(f"  P: Toggle Autopilot | ESC: Exit")
        print(f"{'='*50}\n")
        
        print("⏳ Initializing Graphics Engine...")
        self.renderer.init()
        print("✅ Graphics Initialized.")
        
        # --- KRITIK: renderer.init() SONRASI environment/arena bağla ---
        from src.simulation.arena import TeknofestArena
        
        if self.camera and hasattr(self.world, 'environment'):
            print("🌍 Initializing Environment (terrain, objects)...")
            self.camera.set_environment(self.world.environment)
        
        arena = TeknofestArena({
            'width': 500.0,
            'depth': 500.0,
            'min_altitude': 10.0,
            'max_altitude': 150.0,
            'safe_zone_size': 50.0
        })
        if self.camera:
            print("🏟️ Initializing Arena (markers, safe zones)...")
            self.camera.set_arena(arena)
        
        print("✅ World Ready. Starting loop...")
        
        dt = 1.0 / 60.0
        
        try:
            while self.running:
                self._handle_events()
                self._step(dt)
                self._render()
                self.sim_time += dt
                self.frame_id += 1
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            print(f"\n❌ Simulation Crashed: {e}")
            traceback.print_exc()
        finally:
            self.renderer.close()

if __name__ == "__main__":
    # Senaryo yükle
    scenario_name = 'camera_lock_test' # Basit bir senaryo
    try:
        scenario_config = load_scenario(scenario_name)
    except Exception as e:
        print(f"Senaryo yüklenemedi: {scenario_name}")
        # Fallback to defaults
        scenario_config = {}

    config = {
        **scenario_config,
        'scenario': scenario_name,
        'duration': 9999,  # Sınırsız süre
        'seed': 42
    }
    
    print("\n" + "="*50)
    print("🌎 WORLD EXPLORER - FREE ROAM MODE")
    print("="*50)
    print("🎮 KONTROLLER (MANUEL UÇUŞ):")
    print("   W / S      : Pitch (Burun Aşağı / Yukarı)")
    print("   A / D      : Roll (Sola / Sağa Yat)")
    print("   Q / E      : Yaw (Sola / Sağa Dön)")
    print("   SHIFT      : Gaz Artır (Yüksel/Hızlan)")
    print("   CTRL       : Gaz Azalt (Alçal/Yavaşla)")
    print("   R          : Restart")
    print("   ESC        : Çıkış")
    print("="*50 + "\n")
    
    explorer = WorldExplorer(config)
    explorer.setup_scenario()
    
    # Environment ve arena bağlantıları artık _run_ui() içinde yapılıyor
    # (renderer.init() sonrası)
         
    explorer.running = True  # Initialize running state
    explorer._run_ui()
