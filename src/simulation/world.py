"""
Simülasyon Dünyası

Tüm İHA'ları ve fizik simülasyonunu yöneten ana sınıf.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from ..uav.fixed_wing import FixedWingUAV



class SimulationWorld:
    """
    3D Simülasyon Dünyası
    
    Çoklu İHA yönetimi, fizik güncellemesi ve kamera simülasyonu sağlar.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Dünya parametreleri
        self.world_size = config.get('world_size', [2000, 2000, 500])
        self.ground_level = config.get('ground_level', 0)
        
        # Simülasyon parametreleri
        self.physics_hz = config.get('physics_hz', 100)
        self.dt = 1.0 / self.physics_hz
        self.time_scale = config.get('time_scale', 1.0)
        
        # Environment (Terrain + World Objects)
        from src.simulation.environment import Environment
        env_config = config.get('environment', {
            'terrain': {
                'type': 'hills',  # Creates varying terrain height for 3D appearance
                'size': self.world_size[:2],
                'max_height': 30.0
            },
            'auto_spawn': True,
            'num_buildings': 5,
            'num_trees': 15,
            'seed': config.get('seed', 42)
        })
        self.environment = Environment(env_config)
        
        # İHA'lar
        self.uavs: Dict[str, FixedWingUAV] = {}
        self.player_uav_id: str = None
        
        # Simülasyon durumu
        self.time = 0.0
        self.is_running = False
        self.is_paused = False
        
        # İstatistikler
        self.stats = {
            'total_updates': 0,
            'real_time': 0.0
        }
        
    def spawn_uav(self, uav_id: str = None, team: str = "blue",
                  position: List[float] = None, heading: float = 0.0,
                  is_player: bool = False, config: dict = None,
                  behavior: str = "normal") -> FixedWingUAV:
        """
        Yeni İHA ekle
        
        Args:
            uav_id: Benzersiz ID (None ise otomatik)
            team: Takım ('blue' veya 'red')
            position: Başlangıç pozisyonu [x, y, z]
            heading: Başlangıç yönü (derece)
            is_player: Oyuncu kontrolündeki İHA mı?
            config: İHA konfigürasyonu
            behavior: "normal" veya "stationary" (fizik güncellemesi yok)
        """
        if uav_id is None:
            uav_id = f"uav_{len(self.uavs):03d}"
            
        if position is None:
            position = [
                np.random.uniform(100, self.world_size[0] - 100),
                np.random.uniform(100, self.world_size[1] - 100),
                np.random.uniform(80, 150)
            ]
            
        uav = FixedWingUAV(config=config, uav_id=uav_id, team=team)
        uav.reset(position=np.array(position), heading=np.radians(heading))
        uav.behavior = behavior  # "stationary" ise update() çalışmaz
        
        self.uavs[uav_id] = uav
        
        if is_player:
            self.player_uav_id = uav_id
            
        return uav
        
    def spawn_enemy_uavs(self, count: int = 3, config: dict = None):
        """Düşman İHA'ları spawn et"""
        for i in range(count):
            x = np.random.uniform(200, self.world_size[0] - 200)
            y = np.random.uniform(200, self.world_size[1] - 200)
            z = np.random.uniform(80, 200)
            heading = np.random.uniform(0, 360)
            
            self.spawn_uav(
                uav_id=f"enemy_{i:02d}",
                team="red",
                position=[x, y, z],
                heading=heading,
                config=config
            )
            
    def get_player_uav(self) -> Optional[FixedWingUAV]:
        """Oyuncu İHA'sını al"""
        if self.player_uav_id:
            return self.uavs.get(self.player_uav_id)
        return None
        
    def get_uav(self, uav_id: str) -> Optional[FixedWingUAV]:
        """Belirli İHA'yı al"""
        return self.uavs.get(uav_id)
        
    def get_all_uavs(self) -> List[FixedWingUAV]:
        """Tüm İHA'ları al"""
        return list(self.uavs.values())
        
    def get_enemy_uavs(self, from_team: str = "blue") -> List[FixedWingUAV]:
        """Düşman İHA'ları al"""
        return [uav for uav in self.uavs.values() 
                if uav.team != from_team and not uav.is_crashed]
        
    def update(self, dt: float = None):
        """
        Dünya güncellemesi
        
        Args:
            dt: Zaman adımı (None ise varsayılan kullan)
        """
        if self.is_paused:
            return
            
        dt = dt or self.dt
        dt *= self.time_scale
        
        # Tüm İHA'ları güncelle
        for uav in self.uavs.values():
            if not uav.is_crashed:
                uav.update(dt)
                
                # Dünya sınırlarını ve çarpışmaları kontrol et
                self._check_collisions(uav)
                
        self.time += dt
        self.stats['total_updates'] += 1
        
    def _check_collisions(self, uav: FixedWingUAV):
        """
        Çarpışma kontrolleri (Zemin ve Tavan)
        Sonsuz dünyada X/Y sınırı yoktur.
        """
        pos = uav.state.position
        
        # 1. Tavan Kontrolü (Max Altitude)
        MAX_ALTITUDE = 1000.0
        if pos[2] > MAX_ALTITUDE:
            pos[2] = MAX_ALTITUDE
            if uav.state.velocity[2] > 0:
                uav.state.velocity[2] = 0
        
        # 2. Zemin Kontrolü (Terrain Collision)
        # Anlık konumdaki arazi yüksekliğini al
        from src.simulation.map_data import WorldMap
        ground_z = WorldMap.get_terrain_height(pos[0], pos[1])
        
        # Tolerans (İHA'nın yarıçapı kadar)
        COLLISION_THRESHOLD = 0.5 
        
        if pos[2] < ground_z + COLLISION_THRESHOLD:
            # Yere çarptı!
            if not uav.is_crashed:
                impact_speed = np.linalg.norm(uav.state.velocity)
                print(f"CRASH: {uav.id} hit terrain at {pos} speed={impact_speed:.1f}")
                uav.set_crashed(True)
                
                # Yerin altına girmesini engelle (görsel olarak)
                pos[2] = ground_z + COLLISION_THRESHOLD
                uav.state.velocity[:] = 0  # Durdur
            
    def get_uav_states_for_detection(self, exclude_id: str = None) -> List[Dict]:
        """
        Tespit sistemi için İHA durumlarını al
        
        Args:
            exclude_id: Hariç tutulacak İHA ID'si (genellikle oyuncu)
        """
        states = []
        for uav_id, uav in self.uavs.items():
            if uav_id == exclude_id or uav.is_crashed:
                continue
                
            states.append({
                'id': uav_id,
                'position': uav.get_position().tolist(),
                'team': uav.team,
                'size': uav.wingspan,
                'heading': uav.get_heading_degrees()  # 3D render için gerekli
            })
            
        return states
        
    def get_world_state(self) -> Dict:
        """Tüm dünya durumunu al"""
        return {
            'time': self.time,
            'uavs': {uid: uav.to_dict() for uid, uav in self.uavs.items()},
            'player_id': self.player_uav_id,
            'stats': self.stats.copy()
        }
        
    def pause(self):
        """Simülasyonu duraklat"""
        self.is_paused = True
        
    def resume(self):
        """Simülasyonu devam ettir"""
        self.is_paused = False
        
    def reset(self):
        """Dünyayı sıfırla"""
        self.uavs.clear()
        self.player_uav_id = None
        self.time = 0.0
        self.stats = {'total_updates': 0, 'real_time': 0.0}
