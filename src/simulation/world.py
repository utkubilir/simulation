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
                
                # Dünya sınırlarını kontrol et
                self._enforce_boundaries(uav)
                
        self.time += dt
        self.stats['total_updates'] += 1
        
    def _enforce_boundaries(self, uav: FixedWingUAV):
        """Dünya sınırlarını uygula (6DOF uyumlu)"""
        pos = uav.state.position
        
        # Velocity in Body Frame
        v_body = uav.state.velocity
        
        # Rotation Matrix to convert Body -> Inertial
        # We need this to check and reflect velocity in Inertial Frame
        phi, theta, psi = uav.state.orientation
        R_b2i = uav._rotation_matrix(phi, theta, psi)
        R_i2b = R_b2i.T
        
        # Convert Body Velocity to Inertial for reflection checks
        v_inertial = R_b2i @ v_body
        
        did_hit = False
        
        # X sınırları
        if pos[0] < 0:
            pos[0] = 0
            if v_inertial[0] < 0:
                v_inertial[0] *= -0.5
                did_hit = True
        elif pos[0] > self.world_size[0]:
            pos[0] = self.world_size[0]
            if v_inertial[0] > 0:
                v_inertial[0] *= -0.5
                did_hit = True
            
        # Y sınırları
        if pos[1] < 0:
            pos[1] = 0
            if v_inertial[1] < 0:
                v_inertial[1] *= -0.5
                did_hit = True
        elif pos[1] > self.world_size[1]:
            pos[1] = self.world_size[1]
            if v_inertial[1] > 0:
                v_inertial[1] *= -0.5
                did_hit = True
            
        # Z sınırları (tavan)
        if pos[2] > self.world_size[2]:
            pos[2] = self.world_size[2]
            if v_inertial[2] > 0:
                v_inertial[2] = 0 # Tavana çarpınca dikey hızı sıfırla
                did_hit = True
                
        # Zemin (yer) kontrolünü UAV sınıfı kendi yapıyor (update metodunda)
        
        if did_hit:
            # Convert modified inertial velocity back to body frame
            uav.state.velocity = R_i2b @ v_inertial
            
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
