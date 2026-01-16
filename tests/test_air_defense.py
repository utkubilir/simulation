"""
Hava Savunma Sistemi Test Modülü

AirDefenseManager ve ilgili sınıfların testleri.
"""

import pytest
import numpy as np
from src.simulation.air_defense import (
    AirDefenseManager, AirDefenseZone, ZoneType
)


class TestAirDefenseZone:
    """AirDefenseZone dataclass testleri"""
    
    def test_zone_creation(self):
        """Bölge oluşturma"""
        zone = AirDefenseZone(
            id="test_zone",
            center=(500, 500),
            radius=100
        )
        
        assert zone.id == "test_zone"
        assert zone.center == (500, 500)
        assert zone.radius == 100
        assert zone.zone_type == ZoneType.AIR_DEFENSE
        assert not zone.is_active
        
    def test_zone_contains_inside(self):
        """İçerideki pozisyon tespiti"""
        zone = AirDefenseZone("z1", (500, 500), 100)
        zone.is_active = True
        
        # Merkez
        assert zone.contains(np.array([500, 500, 100]))
        
        # Yarıçap içinde
        assert zone.contains(np.array([550, 550, 150]))
        
    def test_zone_contains_outside(self):
        """Dışarıdaki pozisyon tespiti"""
        zone = AirDefenseZone("z1", (500, 500), 100)
        zone.is_active = True
        
        # Yarıçap dışında
        assert not zone.contains(np.array([700, 700, 100]))
        
    def test_zone_inactive_not_contains(self):
        """İnaktif bölge hiçbir şey içermez"""
        zone = AirDefenseZone("z1", (500, 500), 100)
        zone.is_active = False
        
        # Merkez bile içermez
        assert not zone.contains(np.array([500, 500, 100]))
        
    def test_distance_to_boundary(self):
        """Sınıra mesafe hesaplaması"""
        zone = AirDefenseZone("z1", (500, 500), 100)
        
        # Merkez: -100 (içeride)
        assert zone.distance_to_boundary(np.array([500, 500, 0])) == -100
        
        # Sınırda: 0
        assert abs(zone.distance_to_boundary(np.array([600, 500, 0]))) < 0.01
        
        # Dışarıda: pozitif
        assert zone.distance_to_boundary(np.array([700, 500, 0])) == 100


class TestAirDefenseManager:
    """AirDefenseManager testleri"""
    
    def setup_method(self):
        """Her test için yeni manager"""
        self.manager = AirDefenseManager()
        
    def test_add_zone(self):
        """Bölge ekleme"""
        zone = AirDefenseZone("zone_1", (500, 500), 100)
        self.manager.add_zone(zone)
        
        assert "zone_1" in self.manager.zones
        assert len(self.manager.zones) == 1
        
    def test_zone_activation_by_time(self):
        """Zaman tabanlı aktivasyon"""
        zone = AirDefenseZone(
            id="zone_1",
            center=(500, 500),
            radius=100,
            activation_time=10.0,
            deactivation_time=20.0
        )
        self.manager.add_zone(zone)
        
        # t=5: henüz aktif değil
        result = self.manager.update(5.0, {}, 0.1)
        assert "zone_1" not in result['active_zones']
        
        # t=10: aktif olmalı
        result = self.manager.update(10.0, {}, 0.1)
        assert "zone_1" in result['active_zones']
        
        # t=15: hala aktif
        result = self.manager.update(15.0, {}, 0.1)
        assert "zone_1" in result['active_zones']
        
        # t=20: deaktif olmalı
        result = self.manager.update(20.0, {}, 0.1)
        assert "zone_1" not in result['active_zones']
        
    def test_warning_1_minute_before(self):
        """1 dakika önceden uyarı"""
        zone = AirDefenseZone(
            id="zone_1",
            center=(500, 500),
            radius=100,
            activation_time=120.0  # 2 dakikada aktif
        )
        self.manager.add_zone(zone)
        
        # t=50: henüz uyarı yok
        result = self.manager.update(50.0, {}, 0.1)
        assert len(result['warnings']) == 0
        
        # t=60: uyarı gelmeli (60 saniye önce)
        result = self.manager.update(60.0, {}, 0.1)
        warning_texts = ' '.join(result['warnings'])
        assert 'zone_1' in warning_texts or len(result['warnings']) > 0
        
    def test_violation_penalty(self):
        """İhlal cezası (-5 puan/saniye)"""
        zone = AirDefenseZone("zone_1", (500, 500), 100)
        zone.is_active = True
        self.manager.add_zone(zone)
        
        # UAV bölge içinde
        positions = {'uav_1': np.array([500, 500, 100])}
        
        # 2 saniye geç (simülasyon zamanı artarak)
        for i in range(20):  # 0.1 * 20 = 2s
            sim_time = i * 0.1
            result = self.manager.update(sim_time, positions, 0.1)
            
        # Toplam ceza kontrol (en az 1 saniye = -5)
        assert self.manager.total_penalty <= -5
        
    def test_30_second_limit(self):
        """30 saniye limit kontrolü"""
        zone = AirDefenseZone("zone_1", (500, 500), 100)
        zone.is_active = True
        self.manager.add_zone(zone)
        
        positions = {'uav_1': np.array([500, 500, 100])}
        
        # 30 saniye geç
        for i in range(300):  # 0.1 * 300 = 30s
            result = self.manager.update(i * 0.1, positions, 0.1)
            
        # İniş gerekli
        assert 'uav_1' in result['landing_required']
        
    def test_avoidance_heading(self):
        """Kaçınma heading hesaplaması"""
        zone = AirDefenseZone("zone_1", (500, 500), 100)
        zone.is_active = True
        self.manager.add_zone(zone)
        
        # Bölgeye yakın pozisyon
        position = np.array([480, 500, 100])  # 20m içeride
        
        heading = self.manager.get_avoidance_heading(position, 0)
        
        # Heading dönmeli (None değil)
        assert heading is not None
        
        # Merkezden uzağa yönelmeli (yaklaşık 180 derece)
        assert 90 < heading < 270
        
    def test_load_from_scenario(self):
        """Senaryo dosyasından yükleme"""
        scenario = {
            'air_defense': {
                'enabled': True,
                'zones': [
                    {
                        'id': 'zone_1',
                        'center': [500, 500],
                        'radius': 100,
                        'type': 'air_defense',
                        'activation_time': 60,
                        'duration': 30
                    },
                    {
                        'id': 'jamming_1',
                        'center': [1000, 1000],
                        'radius': 50,
                        'type': 'signal_jamming',
                        'activation_time': 120,
                        'duration': 45
                    }
                ]
            }
        }
        
        self.manager.load_from_scenario(scenario)
        
        assert len(self.manager.zones) == 2
        assert 'zone_1' in self.manager.zones
        assert 'jamming_1' in self.manager.zones
        
        # Tipler doğru mu?
        assert self.manager.zones['zone_1'].zone_type == ZoneType.AIR_DEFENSE
        assert self.manager.zones['jamming_1'].zone_type == ZoneType.SIGNAL_JAMMING


class TestSafeCorridor:
    """Güvenli koridor testi"""
    
    def setup_method(self):
        self.manager = AirDefenseManager()
        
    def test_direct_path_no_obstacle(self):
        """Engelsiz direkt yol"""
        start = np.array([100, 100, 100])
        goal = np.array([200, 200, 100])
        
        waypoints = self.manager.get_safe_corridor(start, goal)
        
        # Başlangıç ve hedef olmalı
        assert len(waypoints) >= 2
        assert np.allclose(waypoints[0], start)
        assert np.allclose(waypoints[-1], goal)
        
    def test_path_around_obstacle(self):
        """Engel etrafından geçiş"""
        zone = AirDefenseZone("z1", (500, 500), 100)
        zone.is_active = True
        self.manager.add_zone(zone)
        
        start = np.array([400, 400, 100])  # Sol alt
        goal = np.array([600, 600, 100])    # Sağ üst
        
        waypoints = self.manager.get_safe_corridor(start, goal)
        
        # Ekstra waypoint olmalı
        assert len(waypoints) >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
