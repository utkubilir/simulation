"""
Scene Manager

Sahne elemanlarını yönetir: arazi, İHA'lar, hava savunma bölgeleri.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SceneObject:
    """Sahne nesnesi"""
    id: str
    position: np.ndarray
    orientation: np.ndarray = None
    scale: float = 1.0
    visible: bool = True
    
    def __post_init__(self):
        if self.orientation is None:
            self.orientation = np.zeros(3)


class SceneManager:
    """
    Sahne yöneticisi
    
    Render edilecek tüm nesneleri takip eder.
    
    Usage:
        scene = SceneManager()
        scene.add_uav("enemy_1", pos, heading, team="red")
        scene.add_air_defense_zone("zone_1", center, radius)
        objects = scene.get_visible_objects()
    """
    
    def __init__(self):
        # Nesneler
        self.uavs: Dict[str, dict] = {}
        self.air_defense_zones: Dict[str, dict] = {}
        self.ground_targets: Dict[str, dict] = {}
        
        # Arazi parametreleri
        self.terrain_config = {
            'size': 2000,
            'resolution': 50,
            'seed': 42
        }
        
        # Genel ayarlar
        self.time_of_day = 12.0  # Saat (0-24)
        self.weather = 'clear'   # clear, cloudy, foggy
        
    def add_uav(self, uav_id: str, 
                position: np.ndarray,
                heading: float = 0,
                pitch: float = 0,
                roll: float = 0,
                team: str = 'red',
                size: float = 2.0):
        """
        İHA ekle veya güncelle
        
        Args:
            uav_id: Benzersiz ID
            position: [x, y, z] pozisyon
            heading: Yaw açısı (derece)
            pitch: Pitch açısı (derece)
            roll: Roll açısı (derece)
            team: Takım ('red', 'blue', 'green')
            size: Wingspan (metre)
        """
        self.uavs[uav_id] = {
            'id': uav_id,
            'position': np.array(position),
            'heading': heading,
            'pitch': pitch,
            'roll': roll,
            'team': team,
            'size': size,
            'visible': True
        }
        
    def remove_uav(self, uav_id: str):
        """İHA kaldır"""
        if uav_id in self.uavs:
            del self.uavs[uav_id]
            
    def update_uav(self, uav_id: str, **kwargs):
        """İHA özelliklerini güncelle"""
        if uav_id in self.uavs:
            for key, value in kwargs.items():
                if key == 'position':
                    self.uavs[uav_id][key] = np.array(value)
                else:
                    self.uavs[uav_id][key] = value
                    
    def add_air_defense_zone(self, zone_id: str,
                              center: Tuple[float, float],
                              radius: float,
                              zone_type: str = 'air_defense',
                              is_active: bool = False,
                              activation_time: float = 0,
                              deactivation_time: float = float('inf')):
        """
        Hava savunma bölgesi ekle
        
        Args:
            zone_id: Benzersiz ID
            center: (x, y) merkez koordinatları
            radius: Yarıçap (metre)
            zone_type: 'air_defense' veya 'signal_jamming'
            is_active: Aktif mi?
            activation_time: Aktifleşme zamanı (saniye)
            deactivation_time: Deaktivasyon zamanı (saniye)
        """
        self.air_defense_zones[zone_id] = {
            'id': zone_id,
            'center': center,
            'radius': radius,
            'zone_type': zone_type,
            'is_active': is_active,
            'activation_time': activation_time,
            'deactivation_time': deactivation_time,
            'warning_sent': False
        }
        
    def update_zone_status(self, zone_id: str, is_active: bool):
        """Bölge durumunu güncelle"""
        if zone_id in self.air_defense_zones:
            self.air_defense_zones[zone_id]['is_active'] = is_active
            
    def add_ground_target(self, target_id: str,
                          position: np.ndarray,
                          size: float = 2.0,
                          target_type: str = 'qr'):
        """Yer hedefi ekle"""
        self.ground_targets[target_id] = {
            'id': target_id,
            'position': np.array(position),
            'size': size,
            'type': target_type,
            'visible': True
        }
        
    def get_visible_uavs(self, camera_pos: np.ndarray = None,
                         max_distance: float = 1000) -> List[dict]:
        """
        Görünür İHA'ları getir
        
        Args:
            camera_pos: Kamera pozisyonu (opsiyonel, mesafe filtreleme için)
            max_distance: Maksimum render mesafesi
            
        Returns:
            İHA durumları listesi
        """
        visible = []
        
        for uav in self.uavs.values():
            if not uav['visible']:
                continue
                
            # Mesafe kontrolü
            if camera_pos is not None:
                dist = np.linalg.norm(uav['position'] - camera_pos)
                if dist > max_distance:
                    continue
                    
            visible.append(uav.copy())
            
        # Mesafeye göre sırala (uzaktan yakına - painter's algorithm)
        if camera_pos is not None:
            visible.sort(
                key=lambda u: np.linalg.norm(u['position'] - camera_pos),
                reverse=True
            )
            
        return visible
        
    def get_active_zones(self, sim_time: float = None) -> List[dict]:
        """
        Aktif hava savunma bölgelerini getir
        
        Args:
            sim_time: Simülasyon zamanı (zamanlama kontrolü için)
            
        Returns:
            Aktif bölge listesi
        """
        active = []
        
        for zone in self.air_defense_zones.values():
            # Zaman tabanlı aktivasyon kontrolü
            if sim_time is not None:
                if sim_time >= zone['activation_time'] and sim_time < zone['deactivation_time']:
                    zone['is_active'] = True
                elif sim_time >= zone['deactivation_time']:
                    zone['is_active'] = False
                    
            if zone['is_active']:
                active.append(zone.copy())
                
        return active
        
    def get_pending_warnings(self, sim_time: float, warning_advance: float = 60.0) -> List[str]:
        """
        Bekleyen uyarıları getir (1 dk önceden)
        
        Args:
            sim_time: Mevcut simülasyon zamanı
            warning_advance: Uyarı süresi (saniye)
            
        Returns:
            Uyarı mesajları
        """
        warnings = []
        
        for zone in self.air_defense_zones.values():
            if zone['warning_sent']:
                continue
                
            time_until_active = zone['activation_time'] - sim_time
            
            if 0 < time_until_active <= warning_advance:
                zone['warning_sent'] = True
                warnings.append(
                    f"⚠️ {zone['id']} ({zone['zone_type']}) "
                    f"{int(time_until_active)} saniye sonra aktif olacak!"
                )
                
        return warnings
        
    def check_zone_violation(self, position: np.ndarray) -> Optional[dict]:
        """
        Pozisyonun aktif bölge içinde olup olmadığını kontrol et
        
        Args:
            position: [x, y, z] pozisyon
            
        Returns:
            İhlal edilen bölge veya None
        """
        for zone in self.air_defense_zones.values():
            if not zone['is_active']:
                continue
                
            dx = position[0] - zone['center'][0]
            dy = position[1] - zone['center'][1]
            
            if (dx**2 + dy**2) <= zone['radius']**2:
                return zone
                
        return None
        
    def get_nearest_zone_distance(self, position: np.ndarray) -> Tuple[Optional[str], float]:
        """
        En yakın aktif bölgeye mesafe
        
        Returns:
            (zone_id, distance) veya (None, inf)
        """
        min_dist = float('inf')
        nearest_id = None
        
        for zone in self.air_defense_zones.values():
            if not zone['is_active']:
                continue
                
            dx = position[0] - zone['center'][0]
            dy = position[1] - zone['center'][1]
            dist = np.sqrt(dx**2 + dy**2) - zone['radius']
            
            if dist < min_dist:
                min_dist = dist
                nearest_id = zone['id']
                
        return nearest_id, min_dist
        
    def load_from_scenario(self, scenario: dict):
        """
        Senaryo dosyasından yükle
        
        Args:
            scenario: Senaryo konfigürasyonu (YAML'dan yüklenmiş)
        """
        # Arazi
        if 'world' in scenario:
            world = scenario['world']
            self.terrain_config.update({
                'size': world.get('size', 2000),
                'resolution': world.get('terrain_resolution', 50)
            })
            
        # Hava savunma bölgeleri
        if 'air_defense' in scenario:
            ad_config = scenario['air_defense']
            
            if ad_config.get('enabled', False):
                zones = ad_config.get('zones', [])
                
                for zone_data in zones:
                    self.add_air_defense_zone(
                        zone_id=zone_data['id'],
                        center=tuple(zone_data['center']),
                        radius=zone_data['radius'],
                        zone_type=zone_data.get('type', 'air_defense'),
                        activation_time=zone_data.get('activation_time', 0),
                        deactivation_time=zone_data.get('activation_time', 0) + 
                                         zone_data.get('duration', float('inf'))
                    )
                    
        # Yer hedefleri
        if 'ground_target' in scenario.get('world', {}):
            target = scenario['world']['ground_target']
            self.add_ground_target(
                target_id='kamikaze_target',
                position=np.array(target['position']),
                size=target.get('qr_size', 2.0)
            )
            
    def clear(self):
        """Tüm nesneleri temizle"""
        self.uavs.clear()
        self.air_defense_zones.clear()
        self.ground_targets.clear()
