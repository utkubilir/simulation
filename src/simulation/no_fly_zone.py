"""
No-Fly Zones (NFZ) - Uçuşa yasaklı bölgeler

TEKNOFEST yarışmasında hakemler tarafından dinamik olarak kontrol edilen bölgeler.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class NFZShape(Enum):
    """NFZ şekil tipi."""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"


@dataclass
class NoFlyZone:
    """
    Uçuşa yasaklı bölge tanımı.
    
    Attributes:
        id: Benzersiz tanımlayıcı
        center: Merkez koordinatları [x, y]
        shape: Şekil tipi (circle/rectangle)
        radius: Yarıçap (circle için)
        width: Genişlik (rectangle için)
        height: Yükseklik (rectangle için)
        min_alt: Minimum irtifa (None = yerden)
        max_alt: Maksimum irtifa (None = sınırsız)
        active: Aktif mi?
        color: Görselleştirme rengi (R, G, B)
    """
    id: str
    center: Tuple[float, float]
    shape: NFZShape = NFZShape.CIRCLE
    radius: float = 100.0
    width: float = 100.0
    height: float = 100.0
    min_alt: Optional[float] = None
    max_alt: Optional[float] = None
    active: bool = True
    color: Tuple[int, int, int] = (255, 50, 50)
    
    def contains(self, position: np.ndarray) -> bool:
        """
        Pozisyonun NFZ içinde olup olmadığını kontrol et.
        
        Args:
            position: [x, y, z] koordinatları
            
        Returns:
            True eğer pozisyon NFZ içindeyse
        """
        if not self.active:
            return False
            
        x, y, z = position[0], position[1], position[2]
        
        # İrtifa kontrolü
        if self.min_alt is not None and z < self.min_alt:
            return False
        if self.max_alt is not None and z > self.max_alt:
            return False
            
        # Yatay mesafe kontrolü
        if self.shape == NFZShape.CIRCLE:
            dx = x - self.center[0]
            dy = y - self.center[1]
            return (dx**2 + dy**2) <= self.radius**2
            
        elif self.shape == NFZShape.RECTANGLE:
            half_w = self.width / 2
            half_h = self.height / 2
            in_x = abs(x - self.center[0]) <= half_w
            in_y = abs(y - self.center[1]) <= half_h
            return in_x and in_y
            
        return False
        
    def distance_to_boundary(self, position: np.ndarray) -> float:
        """
        NFZ sınırına olan mesafeyi hesapla.
        
        Returns:
            Negatif = içeride, Pozitif = dışarıda
        """
        x, y = position[0], position[1]
        
        if self.shape == NFZShape.CIRCLE:
            dx = x - self.center[0]
            dy = y - self.center[1]
            dist = np.sqrt(dx**2 + dy**2)
            return dist - self.radius
            
        elif self.shape == NFZShape.RECTANGLE:
            half_w = self.width / 2
            half_h = self.height / 2
            dx = abs(x - self.center[0]) - half_w
            dy = abs(y - self.center[1]) - half_h
            return max(dx, dy)
            
        return float('inf')


class NFZManager:
    """
    No-Fly Zone yönetici sınıfı.
    
    Tüm NFZ'leri yönetir, ihlal kontrolü yapar.
    """
    
    def __init__(self):
        self.zones: Dict[str, NoFlyZone] = {}
        self.violations: List[Dict] = []  # Son ihlaller
        
    def add_zone(self, zone: NoFlyZone):
        """NFZ ekle."""
        self.zones[zone.id] = zone
        
    def remove_zone(self, zone_id: str):
        """NFZ kaldır."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            
    def set_active(self, zone_id: str, active: bool):
        """NFZ aktifliğini ayarla."""
        if zone_id in self.zones:
            self.zones[zone_id].active = active
            
    def check_violation(self, position: np.ndarray, uav_id: str = None) -> Optional[NoFlyZone]:
        """
        Pozisyonun herhangi bir NFZ'yi ihlal edip etmediğini kontrol et.
        
        Args:
            position: [x, y, z] koordinatları
            uav_id: İhlal eden İHA ID'si
            
        Returns:
            İhlal edilen NFZ veya None
        """
        for zone in self.zones.values():
            if zone.contains(position):
                # İhlal kaydı
                self.violations.append({
                    'zone_id': zone.id,
                    'uav_id': uav_id,
                    'position': position.tolist(),
                })
                return zone
        return None
        
    def get_active_zones(self) -> List[NoFlyZone]:
        """Aktif NFZ'leri döndür."""
        return [z for z in self.zones.values() if z.active]
        
    def get_zone_warnings(self, position: np.ndarray, warning_distance: float = 50.0) -> List[Tuple[NoFlyZone, float]]:
        """
        Yaklaşılan NFZ'leri ve mesafelerini döndür.
        
        Args:
            position: Mevcut pozisyon
            warning_distance: Uyarı mesafesi
            
        Returns:
            [(zone, distance), ...] listesi
        """
        warnings = []
        for zone in self.get_active_zones():
            dist = zone.distance_to_boundary(position)
            if dist < warning_distance:
                warnings.append((zone, dist))
        return sorted(warnings, key=lambda x: x[1])
        
    def clear(self):
        """Tüm zone'ları temizle."""
        self.zones.clear()
        self.violations.clear()
        
    def to_render_data(self) -> List[Dict]:
        """Render için zone verilerini döndür."""
        data = []
        for zone in self.get_active_zones():
            data.append({
                'id': zone.id,
                'center': zone.center,
                'shape': zone.shape.value,
                'radius': zone.radius,
                'width': zone.width,
                'height': zone.height,
                'color': zone.color,
            })
        return data


# Default NFZ presets
def create_default_zones() -> NFZManager:
    """Varsayılan NFZ'leri oluştur."""
    manager = NFZManager()
    
    # Merkez koruma bölgesi
    manager.add_zone(NoFlyZone(
        id="nfz_center",
        center=(1000, 1000),
        shape=NFZShape.CIRCLE,
        radius=150,
        color=(255, 50, 50),
    ))
    
    # Köşe bölgeleri
    manager.add_zone(NoFlyZone(
        id="nfz_corner_ne",
        center=(1800, 1800),
        shape=NFZShape.RECTANGLE,
        width=200,
        height=200,
        color=(255, 100, 50),
    ))
    
    return manager
