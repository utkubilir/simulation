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
    
    # Environment bağlantısını kesinleştir
    if explorer.camera and hasattr(explorer.world, 'environment'):
         explorer.camera.set_environment(explorer.world.environment)
         
    explorer.running = True  # Initialize running state
    explorer._run_ui()
