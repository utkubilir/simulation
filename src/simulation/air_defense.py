"""
Hava Savunma Sistemi Yöneticisi

Şartname 6.3'e uygun hava savunma ve sinyal karıştırma bölgeleri.

Kurallar:
- Aktifleşmeden 1 dakika önce uyarı
- -5 puan/saniye ceza (kırmızı alanda)
- 30 saniye limit → iniş uyarısı
- Koordinatlar sunucudan veya senaryo dosyasından
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ZoneType(Enum):
    """Bölge tipi"""
    AIR_DEFENSE = "air_defense"
    SIGNAL_JAMMING = "signal_jamming"


@dataclass
class AirDefenseZone:
    """
    Hava Savunma Bölgesi
    
    Şartname 6.3:
    - Sanal daireler
    - Dikey eksende sonsuz yükseklik (isteğe bağlı irtifa sınırı)
    - Fiziksel müdahale yok (sadece ceza)
    """
    id: str
    center: Tuple[float, float]          # (x, y) merkez
    radius: float                         # Yarıçap (metre)
    zone_type: ZoneType = ZoneType.AIR_DEFENSE
    penalty_per_second: int = 0           # -puan/saniye (bölge tipine göre)
    min_alt: Optional[float] = None       # Minimum irtifa (metre)
    max_alt: Optional[float] = None       # Maksimum irtifa (metre)
    
    # Zamanlama
    activation_time: float = 0.0         # Aktifleşme zamanı (saniye)
    deactivation_time: float = float('inf')  # Deaktivasyon zamanı
    
    # Durum
    is_active: bool = False
    warning_sent: bool = False
    
    def contains(self, position: np.ndarray) -> bool:
        """
        Pozisyon bölge içinde mi?
        
        Not: Dikey eksende sonsuz yükseklik (min/max irtifa verilmediyse)
        """
        if not self.is_active:
            return False

        if self.min_alt is not None and position[2] < self.min_alt:
            return False
        if self.max_alt is not None and position[2] > self.max_alt:
            return False
            
        dx = position[0] - self.center[0]
        dy = position[1] - self.center[1]
        
        return (dx**2 + dy**2) <= self.radius**2
        
    def distance_to_boundary(self, position: np.ndarray) -> float:
        """
        Sınıra mesafe
        
        Returns:
            Negatif = içeride, Pozitif = dışarıda
        """
        dx = position[0] - self.center[0]
        dy = position[1] - self.center[1]
        
        return np.sqrt(dx**2 + dy**2) - self.radius


class AirDefenseManager:
    """
    Şartname 6.3 uyumlu hava savunma yönetimi
    
    Features:
    - Dinamik bölge aktivasyonu
    - 1 dakika önceden uyarı
    - İhlal takibi ve ceza hesaplama
    - 30 saniye limit kontrolü
    - Kaçınma heading önerisi
    
    Usage:
        manager = AirDefenseManager()
        manager.load_from_scenario(scenario)
        result = manager.update(sim_time, uav_positions, dt)
    """
    
    # Şartname sabitleri
    WARNING_ADVANCE_SEC = 60.0        # 1 dakika önceden uyarı
    PENALTY_PER_SECOND_BY_TYPE = {
        ZoneType.AIR_DEFENSE: -5,
        ZoneType.SIGNAL_JAMMING: -2
    }
    MAX_VIOLATION_SEC = 30.0          # 30 saniye limit
    SAFE_DISTANCE = 50.0              # Güvenli mesafe (kaçınma için)
    
    def __init__(self):
        # Bölgeler
        self.zones: Dict[str, AirDefenseZone] = {}
        
        # Her UAV ve bölge için ihlal takibi
        self.violation_time: Dict[str, Dict[str, float]] = {}
        self.total_penalty: int = 0
        
        # Uyarılar ve loglar
        self.warnings_log: List[Tuple[float, str]] = []
        self.violations_log: List[Dict] = []
        
    def add_zone(self, zone: AirDefenseZone):
        """Bölge ekle"""
        self.zones[zone.id] = zone
        
    def remove_zone(self, zone_id: str):
        """Bölge kaldır"""
        if zone_id in self.zones:
            del self.zones[zone_id]
            
    def load_from_scenario(self, scenario: dict):
        """
        Senaryo dosyasından bölgeleri yükle
        
        Beklenen format:
        air_defense:
          enabled: true
          zones:
            - id: zone_1
              center: [500, 500]
              radius: 100
              activation_time: 180
              duration: 90
              type: air_defense
        """
        ad_config = scenario.get('air_defense', {})
        
        if not ad_config.get('enabled', False):
            print("ℹ️ Air defense disabled in scenario")
            return
            
        zones = ad_config.get('zones', [])
        
        for zone_data in zones:
            zone_id = zone_data.get('id')
            center = zone_data.get('center')
            radius = zone_data.get('radius')
            activation = zone_data.get('activation_time', 0)
            duration = zone_data.get('duration', float('inf'))

            if not zone_id or not center or radius is None:
                print("⚠️ Air defense zone skipped (missing id/center/radius)")
                continue

            if not isinstance(center, (list, tuple)) or len(center) != 2:
                print(f"⚠️ Air defense zone {zone_id} skipped (center format)")
                continue

            if radius <= 0:
                print(f"⚠️ Air defense zone {zone_id} skipped (radius <= 0)")
                continue

            if activation < 0 or duration <= 0:
                print(f"⚠️ Air defense zone {zone_id} skipped (invalid timing)")
                continue

            zone_type = ZoneType.AIR_DEFENSE
            if zone_data.get('type') == 'signal_jamming':
                zone_type = ZoneType.SIGNAL_JAMMING

            min_alt = zone_data.get('min_alt')
            max_alt = zone_data.get('max_alt')

            penalty = zone_data.get(
                'penalty_per_second',
                self.PENALTY_PER_SECOND_BY_TYPE.get(zone_type, 0)
            )

            zone = AirDefenseZone(
                id=zone_id,
                center=tuple(center),
                radius=radius,
                zone_type=zone_type,
                penalty_per_second=penalty,
                min_alt=min_alt,
                max_alt=max_alt,
                activation_time=activation,
                deactivation_time=activation + duration
            )

            self.add_zone(zone)
            
        print(f"✅ Loaded {len(zones)} air defense zones")
        
    def update(self, 
               sim_time: float,
               uav_positions: Dict[str, np.ndarray],
               dt: float) -> Dict:
        """
        Her frame güncelle
        
        Args:
            sim_time: Simülasyon zamanı (saniye)
            uav_positions: {uav_id: [x, y, z]} pozisyonları
            dt: Zaman adımı
            
        Returns:
            {
                'active_zones': List[str],
                'warnings': List[str],
                'violations': Dict[str, Dict[str, float]],
                'penalty_this_frame': int,
                'total_penalty': int,
                'landing_required': List[str]
            }
        """
        result = {
            'active_zones': [],
            'warnings': [],
            'violations': {},
            'penalty_this_frame': 0,
            'total_penalty': self.total_penalty,
            'landing_required': []
        }
        
        # 1. Bölge durumlarını güncelle
        for zone_id, zone in self.zones.items():
            # Aktivasyon kontrolü
            if not zone.is_active:
                if sim_time >= zone.activation_time:
                    zone.is_active = True
                    msg = f"🔴 {zone_id} ({zone.zone_type.value}) AKTİF!"
                    result['warnings'].append(msg)
                    self.warnings_log.append((sim_time, msg))
                    
            # Deaktivasyon kontrolü
            if zone.is_active:
                if sim_time >= zone.deactivation_time:
                    zone.is_active = False
                    zone.warning_sent = False
                    msg = f"🟢 {zone_id} deaktif edildi"
                    result['warnings'].append(msg)
                    self.warnings_log.append((sim_time, msg))
                    
            # 1 dakika önceden uyarı
            if not zone.warning_sent:
                time_until = zone.activation_time - sim_time
                if 0 < time_until <= self.WARNING_ADVANCE_SEC:
                    zone.warning_sent = True
                    msg = f"⚠️ {zone_id} {int(time_until)} saniye sonra aktif olacak!"
                    result['warnings'].append(msg)
                    self.warnings_log.append((sim_time, msg))
                    
            if zone.is_active:
                result['active_zones'].append(zone_id)
                
        # 2. İhlal kontrolü
        for uav_id, position in uav_positions.items():
            violating_zones: List[AirDefenseZone] = []

            for zone in self.zones.values():
                if zone.contains(position):
                    violating_zones.append(zone)

            if violating_zones:
                if uav_id not in self.violation_time:
                    self.violation_time[uav_id] = {}

                for zone in violating_zones:
                    zone_time = self.violation_time[uav_id].get(zone.id, 0.0)
                    zone_time += dt
                    self.violation_time[uav_id][zone.id] = zone_time

                    # Ceza hesapla (her tam saniye için)
                    full_seconds = int(zone_time)
                    prev_seconds = int(zone_time - dt)

                    if full_seconds > prev_seconds:
                        penalty = zone.penalty_per_second
                        self.total_penalty += penalty
                        result['penalty_this_frame'] += penalty

                        self.violations_log.append({
                            'time': sim_time,
                            'uav_id': uav_id,
                            'zone_id': zone.id,
                            'zone_type': zone.zone_type.value,
                            'penalty': penalty
                        })

                    if zone_time >= self.MAX_VIOLATION_SEC:
                        if uav_id not in result['landing_required']:
                            result['landing_required'].append(uav_id)

                # İhlal olmayan bölgeleri sıfırla
                active_zone_ids = {zone.id for zone in violating_zones}
                for zone_id in list(self.violation_time[uav_id].keys()):
                    if zone_id not in active_zone_ids:
                        del self.violation_time[uav_id][zone_id]
            else:
                if uav_id in self.violation_time:
                    del self.violation_time[uav_id]

            if uav_id in self.violation_time:
                result['violations'][uav_id] = self.violation_time[uav_id].copy()
                
        result['total_penalty'] = self.total_penalty
        
        return result
        
    def get_avoidance_heading(self,
                               current_pos: np.ndarray,
                               current_heading: float,
                               velocity: np.ndarray = None) -> Optional[float]:
        """
        Kaçınma heading'i hesapla
        
        Args:
            current_pos: Mevcut pozisyon
            current_heading: Mevcut heading (derece)
            velocity: Hız vektörü
            
        Returns:
            Önerilen heading (derece) veya None
        """
        # En yakın tehlikeli bölgeyi bul
        min_dist = float('inf')
        closest_zone = None
        
        for zone in self.zones.values():
            if not zone.is_active:
                continue
            if zone.min_alt is not None and current_pos[2] < zone.min_alt:
                continue
            if zone.max_alt is not None and current_pos[2] > zone.max_alt:
                continue

            dist = zone.distance_to_boundary(current_pos)
            
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone
                
        # Tehlike yok
        if closest_zone is None or min_dist > self.SAFE_DISTANCE:
            return None
            
        # Bölge merkezinden kaçış yönü
        dx = current_pos[0] - closest_zone.center[0]
        dy = current_pos[1] - closest_zone.center[1]
        
        # Radyal yön (merkezden uzaklaşma)
        radial_heading = np.degrees(np.arctan2(dy, dx))
        
        # Teğet yönler
        tangent_cw = (radial_heading + 90) % 360
        tangent_ccw = (radial_heading - 90) % 360
        
        # Mevcut heading'e en yakın teğet yönü seç
        diff_cw = abs((tangent_cw - current_heading + 180) % 360 - 180)
        diff_ccw = abs((tangent_ccw - current_heading + 180) % 360 - 180)

        avoidance_heading = tangent_cw if diff_cw < diff_ccw else tangent_ccw

        # Hız vektörü varsa, akışa en yakın kaçınmayı tercih et
        if velocity is not None and np.linalg.norm(velocity[:2]) > 1e-6:
            vel_heading = np.degrees(np.arctan2(velocity[1], velocity[0])) % 360
            diff_vel_cw = abs((tangent_cw - vel_heading + 180) % 360 - 180)
            diff_vel_ccw = abs((tangent_ccw - vel_heading + 180) % 360 - 180)
            avoidance_heading = tangent_cw if diff_vel_cw < diff_vel_ccw else tangent_ccw
            
        # İçerideyse direkt dışarı çık
        if min_dist < 0:
            avoidance_heading = radial_heading
            
        return avoidance_heading % 360
        
    def get_safe_corridor(self,
                          start: np.ndarray,
                          goal: np.ndarray) -> List[np.ndarray]:
        """
        Güvenli waypoint rotası oluştur
        
        Args:
            start: Başlangıç pozisyonu
            goal: Hedef pozisyonu
            
        Returns:
            Waypoint listesi
        """
        # Basit implementasyon:
        # Aktif bölgelerin etrafından geç ve doğrusal çakışmaları kontrol et
        
        waypoints = [start.copy()]
        current = start.copy()
        
        # Direkt yol üzerinde bölge var mı kontrol et
        def segment_intersects_zone(a: np.ndarray, b: np.ndarray, zone: AirDefenseZone) -> bool:
            if not zone.is_active:
                return False
            if zone.min_alt is not None and a[2] < zone.min_alt and b[2] < zone.min_alt:
                return False
            if zone.max_alt is not None and a[2] > zone.max_alt and b[2] > zone.max_alt:
                return False
            center = np.array(zone.center)
            ab = b[:2] - a[:2]
            if np.allclose(ab, 0):
                return np.linalg.norm(a[:2] - center) <= zone.radius + self.SAFE_DISTANCE
            t = np.dot(center - a[:2], ab) / np.dot(ab, ab)
            t = np.clip(t, 0.0, 1.0)
            closest = a[:2] + t * ab
            return np.linalg.norm(closest - center) <= zone.radius + self.SAFE_DISTANCE

        direction = goal[:2] - start[:2]
        distance = np.linalg.norm(direction)

        if distance < 1:
            waypoints.append(goal.copy())
            return waypoints

        direction = direction / distance

        blocked_zones = []
        for zone in self.zones.values():
            if segment_intersects_zone(start, goal, zone):
                blocked_zones.append(zone)
                        
        # Bölgelerin etrafından geç
        for zone in blocked_zones:
            # Bölgenin teğet noktasını bul
            to_zone = np.array(zone.center) - current[:2]
            dist_to_zone = np.linalg.norm(to_zone)
            
            if dist_to_zone < 1:
                continue
                
            # Teğet nokta (bölge yarıçapı + güvenli mesafe)
            bypass_radius = zone.radius + self.SAFE_DISTANCE * 2
            
            # Sağdan mı soldan mı geç?
            direction = goal[:2] - current[:2]
            dir_len = np.linalg.norm(direction)
            if dir_len > 1e-6:
                direction = direction / dir_len
            cross = direction[0] * to_zone[1] - direction[1] * to_zone[0]
            
            if cross > 0:  # Soldan geç
                angle = np.arctan2(to_zone[1], to_zone[0]) - np.pi/2
            else:  # Sağdan geç
                angle = np.arctan2(to_zone[1], to_zone[0]) + np.pi/2
                
            bypass_point = np.array([
                zone.center[0] + bypass_radius * np.cos(angle),
                zone.center[1] + bypass_radius * np.sin(angle),
                current[2]  # Aynı irtifa
            ])
            
            waypoints.append(bypass_point)
            current = bypass_point
            
        waypoints.append(goal.copy())
        
        return waypoints
        
    def get_zone_info(self, zone_id: str) -> Optional[Dict]:
        """Bölge bilgisi getir"""
        if zone_id not in self.zones:
            return None
            
        zone = self.zones[zone_id]
        
        return {
            'id': zone_id,
            'center': zone.center,
            'radius': zone.radius,
            'type': zone.zone_type.value,
            'penalty_per_second': zone.penalty_per_second,
            'min_alt': zone.min_alt,
            'max_alt': zone.max_alt,
            'is_active': zone.is_active,
            'activation_time': zone.activation_time,
            'deactivation_time': zone.deactivation_time
        }
        
    def get_all_zones(self) -> List[Dict]:
        """Tüm bölgeleri getir"""
        return [self.get_zone_info(zid) for zid in self.zones]
        
    def get_violation_stats(self) -> Dict:
        """İhlal istatistikleri"""
        return {
            'total_penalty': self.total_penalty,
            'violation_times': self.violation_time.copy(),
            'violation_count': len(self.violations_log),
            'warnings_count': len(self.warnings_log)
        }
        
    def reset(self):
        """Durumu sıfırla"""
        self.violation_time.clear()
        self.total_penalty = 0
        self.warnings_log.clear()
        self.violations_log.clear()
        
        for zone in self.zones.values():
            zone.is_active = False
            zone.warning_sent = False
            
    def clear(self):
        """Tüm bölgeleri temizle"""
        self.zones.clear()
        self.reset()
