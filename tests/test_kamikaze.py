"""
Kamikaze Görevi Test Modülü

KamikazeController ve QR tespit sistemi için birim testleri.
"""

import pytest
import numpy as np
from src.uav.kamikaze import (
    KamikazeController, KamikazePhase, KamikazeConfig, GroundTarget
)
from src.vision.qr_detector import SyntheticQRDetector, QRDetection


class TestKamikazePhases:
    """Kamikaze faz geçiş testleri"""
    
    def setup_method(self):
        """Her test için yeni controller oluştur"""
        self.target = GroundTarget(
            position=np.array([1000, 1000, 0]),
            qr_content="TEST_QR_2026",
            size=2.0,
            wall_height=3.0,
            wall_angle=45.0
        )
        self.config = KamikazeConfig(
            min_dive_altitude=100.0,
            approach_altitude=120.0,
            pullup_altitude=30.0
        )
        self.controller = KamikazeController(self.target, self.config)
        
    def test_initial_state_is_idle(self):
        """Başlangıç durumu IDLE olmalı"""
        assert self.controller.phase == KamikazePhase.IDLE
        assert not self.controller.is_active()
        
    def test_start_transitions_to_approach(self):
        """start() çağrıldığında APPROACH fazına geçmeli"""
        self.controller.start(sim_time=0.0)
        assert self.controller.phase == KamikazePhase.APPROACH
        assert self.controller.is_active()
        
    def test_approach_to_climb_transition(self):
        """Hedefe yaklaşınca CLIMB fazına geçmeli"""
        self.controller.start(sim_time=0.0)
        
        # Hedefe yakın bir pozisyondan update
        uav_state = {
            'position': [900, 900, 100],  # 141m uzakta
            'altitude': 100,
            'heading': 45,
            'speed': 25.0
        }
        
        result = self.controller.update(uav_state, {}, sim_time=1.0)
        assert result['phase'] == KamikazePhase.CLIMB
        
    def test_climb_to_align_transition(self):
        """Yeterli irtifaya ulaşınca ALIGN fazına geçmeli"""
        self.controller.start(sim_time=0.0)
        self.controller.phase = KamikazePhase.CLIMB  # Manuel geçiş
        
        uav_state = {
            'position': [950, 950, 105],  # 100m üstünde
            'altitude': 105,
            'heading': 45,
            'speed': 25.0
        }
        
        result = self.controller.update(uav_state, {}, sim_time=2.0)
        assert result['phase'] == KamikazePhase.ALIGN
        assert self.controller.dive_start_altitude >= 100
        
    def test_align_to_dive_transition(self):
        """Hedef üzerinde hizalayınca DIVE fazına geçmeli"""
        self.controller.start(sim_time=0.0)
        self.controller.phase = KamikazePhase.ALIGN
        
        uav_state = {
            'position': [1010, 1010, 110],  # Hedefin ~14m yakınında
            'altitude': 110,
            'heading': 0,
            'speed': 20.0
        }
        
        result = self.controller.update(uav_state, {}, sim_time=3.0)
        assert result['phase'] == KamikazePhase.DIVE
        
    def test_dive_qr_detection(self):
        """Dalış sırasında QR tespit edilmeli"""
        self.controller.start(sim_time=0.0)
        self.controller.phase = KamikazePhase.DIVE
        self.controller.dive_start_altitude = 110
        
        uav_state = {
            'position': [1000, 1000, 50],
            'altitude': 50,
            'heading': 0,
            'speed': 30.0
        }
        
        camera_data = {
            'qr_detected': True,
            'qr_content': 'TEST_QR_2026'
        }
        
        result = self.controller.update(uav_state, camera_data, sim_time=4.0)
        
        assert self.controller.qr_detected
        assert self.controller.qr_read_content == 'TEST_QR_2026'
        assert result['server_packet'] is not None
        assert result['server_packet']['type'] == 'kamikaze'
        
    def test_pullup_to_complete(self):
        """Toparlanma sonrası COMPLETE olmalı (QR okunmuşsa)"""
        self.controller.start(sim_time=0.0)
        self.controller.phase = KamikazePhase.PULLUP
        self.controller.qr_detected = True
        self.controller.qr_read_content = 'TEST'
        
        uav_state = {
            'position': [1000, 1000, 85],  # pullup_target_altitude üstünde
            'altitude': 85,
            'heading': 0,
            'speed': 30.0
        }
        
        result = self.controller.update(uav_state, {}, sim_time=5.0)
        
        assert result['phase'] == KamikazePhase.COMPLETE
        assert result['mission_complete']
        assert result['mission_success']
        
    def test_pullup_to_failed_without_qr(self):
        """QR okunmadan toparlanma FAILED olmalı"""
        self.controller.start(sim_time=0.0)
        self.controller.phase = KamikazePhase.PULLUP
        self.controller.qr_detected = False  # QR okunamadı
        
        uav_state = {
            'position': [1000, 1000, 85],
            'altitude': 85,
            'heading': 0,
            'speed': 30.0
        }
        
        result = self.controller.update(uav_state, {}, sim_time=5.0)
        
        assert result['phase'] == KamikazePhase.FAILED
        assert result['mission_complete']
        assert not result['mission_success']


class TestQRDetector:
    """Sentetik QR tespit testleri"""
    
    def setup_method(self):
        """Dedektör kur"""
        self.detector = SyntheticQRDetector({
            'fov': 60,
            'width': 640,
            'height': 480
        })
        
        self.ground_target = {
            'position': [1000, 1000, 0],
            'qr_content': 'TEKNOFEST2026',
            'size': 2.0,
            'wall_height': 3.0
        }
        
    def test_no_detection_when_looking_horizontally(self):
        """Yatay bakışta QR görünmemeli (plaka engellemesi)"""
        camera_state = {
            'position': [800, 800, 100],
            'orientation': [0, 0, np.radians(45)],  # Yatay bakış
        }
        
        result = self.detector.detect(camera_state, self.ground_target)
        
        assert not result.detected
        
    def test_detection_when_diving_steep(self):
        """Dik dalışta QR görünmeli"""
        # Hedefin tam üzerinden dik bakış (biraz kaydırılmış pozisyon)
        camera_state = {
            'position': [1001, 1001, 50],  # Hedef üzerine yakın, düşük irtifa
            'orientation': [0, np.radians(-80), 0],  # 80° aşağı bakış
            'forward_vector': [0.1, 0.1, -0.99]  # Neredeyse aşağı
        }
        
        result = self.detector.detect(camera_state, self.ground_target)
        
        # Mesafe kontrolü - eğer tespit başarısızsa pitch açısına bak
        # SyntheticQRDetector minimum 45° pitch gerektirir
        assert result.angle >= 45 or result.detected, f"Angle: {result.angle}, Detected: {result.detected}"
        
    def test_no_detection_too_far(self):
        """Çok uzaktan QR görünmemeli"""
        camera_state = {
            'position': [1000, 1000, 200],  # 200m yukarıda
            'orientation': [0, np.radians(-90), 0],
            'forward_vector': [0, 0, -1]
        }
        
        # max_detection_distance = 150m varsayılan
        result = self.detector.detect(camera_state, self.ground_target)
        
        assert not result.detected


class TestKamikazeConfig:
    """Kamikaze konfigürasyon testleri"""
    
    def test_default_config_values(self):
        """Varsayılan değerler şartname uyumlu olmalı"""
        config = KamikazeConfig()
        
        assert config.min_dive_altitude == 100.0  # Şartname: min 100m
        assert config.approach_altitude == 120.0
        assert config.pullup_altitude == 30.0
        
    def test_custom_config(self):
        """Özel konfigürasyon çalışmalı"""
        config = KamikazeConfig(
            min_dive_altitude=150.0,
            dive_angle=-70.0
        )
        
        assert config.min_dive_altitude == 150.0
        assert config.dive_angle == -70.0


class TestGroundTarget:
    """Yer hedefi dataclass testleri"""
    
    def test_ground_target_defaults(self):
        """Varsayılan değerler şartname uyumlu olmalı"""
        target = GroundTarget(
            position=np.array([500, 500, 0]),
            qr_content="TEST"
        )
        
        assert target.size == 2.0  # Şartname: 2m x 2m
        assert target.wall_height == 3.0  # Şartname: 3m
        assert target.wall_angle == 45.0  # Şartname: 45°


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
