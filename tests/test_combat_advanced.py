"""
Gelişmiş Takip ve Lock Sistemi Test Modülü

AdvancedPursuitLogic, TargetSelector ve CompetitionLockValidator testleri.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.uav.combat import (
    AdvancedPursuitLogic, TargetSelector, AdvancedVisualServo, CombatConfig
)
from src.vision.lock_on import CompetitionLockValidator, LockConfig


class TestAdvancedPursuitLogic:
    """Gelişmiş takip algoritmalarının testleri"""
    
    def test_lead_pursuit_stationary_target(self):
        """Sabit hedef için lead pursuit"""
        my_pos = np.array([0, 0, 100])
        my_vel = np.array([25, 0, 0])
        target_pos = np.array([100, 0, 100])
        target_vel = np.array([0, 0, 0])  # Sabit
        
        heading, pitch = AdvancedPursuitLogic.lead_pursuit(
            my_pos, my_vel, target_pos, target_vel, lead_time=2.0
        )
        
        # Sabit hedef için direkt hedefe yönelmeli
        assert abs(heading - 0) < 1  # ~0 derece
        assert abs(pitch) < 1  # ~0 derece (aynı irtifa)
        
    def test_lead_pursuit_moving_target(self):
        """Hareketli hedef için lead pursuit"""
        my_pos = np.array([0, 0, 100])
        my_vel = np.array([25, 0, 0])
        target_pos = np.array([100, 0, 100])
        target_vel = np.array([0, 20, 0])  # Y yönünde hareket
        
        heading, pitch = AdvancedPursuitLogic.lead_pursuit(
            my_pos, my_vel, target_pos, target_vel, lead_time=2.0
        )
        
        # Hedef ileri gidecek, heading pozitif olmalı
        assert heading > 0
        assert heading < 90
        
    def test_collision_course_calculation(self):
        """Çarpışma kursu hesaplaması"""
        my_pos = np.array([0, 0, 100])
        my_speed = 30.0
        target_pos = np.array([200, 100, 100])
        target_vel = np.array([-20, 0, 0])  # Bize doğru geliyor
        
        heading, pitch = AdvancedPursuitLogic.collision_course(
            my_pos, my_speed, target_pos, target_vel
        )
        
        # Hedef bize doğru geldiği için intercept noktası daha yakın
        assert 0 < heading < 60  # Sağa yönelmeli
        
    def test_collision_course_very_close_target(self):
        """Çok yakın hedef için çarpışma kursu"""
        my_pos = np.array([0, 0, 100])
        my_speed = 30.0
        target_pos = np.array([0.5, 0, 100])  # Çok yakın
        target_vel = np.array([0, 0, 0])
        
        heading, pitch = AdvancedPursuitLogic.collision_course(
            my_pos, my_speed, target_pos, target_vel
        )
        
        # Çok yakınken (< 1m) sıfır dönmeli
        assert heading == 0.0
        assert pitch == 0.0


class TestTargetSelector:
    """Hedef seçici testleri"""
    
    def setup_method(self):
        """Her test için yeni selector"""
        self.selector = TargetSelector()
        
    def create_mock_track(self, track_id, world_pos, is_confirmed=True):
        """Test için mock track oluştur"""
        track = MagicMock()
        track.id = track_id
        track.world_pos = np.array(world_pos)
        track.is_confirmed = is_confirmed
        return track
        
    def test_select_closest_target(self):
        """En yakın hedef seçilmeli"""
        tracks = [
            self.create_mock_track(1, [100, 100, 100]),  # 141m
            self.create_mock_track(2, [50, 50, 100]),    # 70m - en yakın
            self.create_mock_track(3, [200, 200, 100]),  # 283m
        ]
        
        my_pos = np.array([0, 0, 100])
        selected = self.selector.select_target(tracks, my_pos)
        
        assert selected.id == 2
        
    def test_consecutive_lock_prevention(self):
        """Art arda aynı hedefe kilitlenme engellenmeli"""
        track1 = self.create_mock_track(1, [50, 50, 100])
        track2 = self.create_mock_track(2, [100, 100, 100])
        
        # İlk seçim
        selected = self.selector.select_target([track1, track2], np.array([0, 0, 100]))
        assert selected.id == 1  # En yakın
        
        # Lock kaydet
        self.selector.register_lock(1, timestamp=10.0)
        
        # Tekrar seçim - aynı hedef seçilmemeli
        selected = self.selector.select_target([track1, track2], np.array([0, 0, 100]))
        assert selected.id == 2  # Artık 1 yerine 2 seçilmeli
        
    def test_can_lock_target(self):
        """Hedefe kilitlenebilir mi kontrolü"""
        assert self.selector.can_lock_target(1)  # İlk hedef OK
        
        self.selector.register_lock(1, timestamp=10.0)
        
        assert not self.selector.can_lock_target(1)  # Art arda aynı hedef NO
        assert self.selector.can_lock_target(2)  # Farklı hedef OK
        
    def test_lock_count_tracking(self):
        """Kilitlenme sayısı takibi"""
        self.selector.register_lock(1, timestamp=10.0)
        self.selector.register_lock(2, timestamp=20.0)
        self.selector.register_lock(1, timestamp=30.0)  # Tekrar 1
        
        assert self.selector.get_lock_count() == 3
        assert self.selector.get_unique_lock_count() == 2
        
    def test_confirmed_targets_preferred(self):
        """Onaylı hedefler tercih edilmeli"""
        tracks = [
            self.create_mock_track(1, [30, 30, 100], is_confirmed=False),  # En yakın ama onaysız
            self.create_mock_track(2, [50, 50, 100], is_confirmed=True),   # Biraz uzak ama onaylı
        ]
        
        selected = self.selector.select_target(tracks, np.array([0, 0, 100]))
        assert selected.id == 2  # Onaylı hedef seçilmeli


class TestAdvancedVisualServo:
    """Gelişmiş görsel servo testleri"""
    
    def setup_method(self):
        """Her test için yeni servo"""
        self.servo = AdvancedVisualServo()
        
    def test_center_target_zero_output(self):
        """Merkezdeki hedef için sıfır çıktı"""
        center = (320, 240)  # Ekran merkezi
        frame_size = (640, 480)
        
        roll, pitch = self.servo.calculate_commands(center, frame_size, dt=0.016)
        
        assert abs(roll) < 0.1
        assert abs(pitch) < 0.1
        
    def test_left_target_positive_roll(self):
        """Soldaki hedef için pozitif roll"""
        center = (100, 240)  # Solda
        frame_size = (640, 480)
        
        roll, pitch = self.servo.calculate_commands(center, frame_size, dt=0.016)
        
        assert roll > 0  # Sola dön = pozitif roll
        
    def test_right_target_negative_roll(self):
        """Sağdaki hedef için negatif roll"""
        center = (540, 240)  # Sağda
        frame_size = (640, 480)
        
        roll, pitch = self.servo.calculate_commands(center, frame_size, dt=0.016)
        
        assert roll < 0  # Sağa dön = negatif roll
        
    def test_output_clamped(self):
        """Çıktı -1, 1 arasında olmalı"""
        # Aşırı sapma
        center = (0, 0)  # Sol üst köşe
        frame_size = (640, 480)
        
        roll, pitch = self.servo.calculate_commands(center, frame_size, dt=0.016)
        
        assert -1 <= roll <= 1
        assert -1 <= pitch <= 1
        
    def test_pid_integral_accumulation(self):
        """PID integral birikimi çalışmalı"""
        center = (200, 240)  # Sol
        frame_size = (640, 480)
        
        # Birden fazla adım
        for _ in range(10):
            roll1, _ = self.servo.calculate_commands(center, frame_size, dt=0.016)
            
        # İntegral birikmiş olmalı (daha büyük çıktı)
        roll2, _ = self.servo.calculate_commands(center, frame_size, dt=0.016)
        
        # İlk birkaç iterasyondan sonra integral etkisi görülmeli
        assert roll2 != 0


class TestCompetitionLockValidator:
    """Şartname uyumlu lock validatör testleri"""
    
    def setup_method(self):
        """Her test için yeni validatör"""
        self.validator = CompetitionLockValidator()
        
    def test_valid_frame_in_center(self):
        """Merkezdeki büyük hedef geçerli olmalı"""
        detection = {
            'bbox': [250, 180, 140, 120],  # Merkeze yakın, büyük
            'center': (320, 240),
            'id': 1,
            'confidence': 0.9
        }
        
        result = self.validator.validate_frame(detection, (640, 480), sim_time=1.0)
        
        assert result['in_target_area']
        assert result['size_sufficient']
        assert result['valid_frame']
        
    def test_invalid_frame_outside_target_area(self):
        """Hedef alanı dışındaki tespit geçersiz olmalı"""
        detection = {
            'bbox': [10, 10, 50, 50],  # Sol üst köşe
            'center': (35, 35),
            'id': 1,
            'confidence': 0.9
        }
        
        result = self.validator.validate_frame(detection, (640, 480), sim_time=1.0)
        
        assert not result['in_target_area']
        assert not result['valid_frame']
        assert result['reason'] == 'outside_target_area'
        
    def test_invalid_frame_too_small(self):
        """Çok küçük hedef geçersiz olmalı"""
        detection = {
            'bbox': [310, 230, 20, 20],  # Merkez ama çok küçük
            'center': (320, 240),
            'id': 1,
            'confidence': 0.9
        }
        
        result = self.validator.validate_frame(detection, (640, 480), sim_time=1.0)
        
        assert result['in_target_area']
        assert not result['size_sufficient']
        assert not result['valid_frame']
        
    def test_continuous_lock_detection(self):
        """4 saniye kesintisiz lock tespiti"""
        # 5 saniye boyunca geçerli frame ekle
        for i in range(300):  # 60 fps x 5 saniye
            t = i / 60.0
            frame_result = {'valid_frame': True}
            self.validator.add_frame_to_history(frame_result, target_id=1, sim_time=t)
            
        result = self.validator.check_continuous_lock(target_id=1, end_time=5.0)
        
        assert result['is_locked']
        assert result['continuous_duration'] >= 3.8  # Toleransla
        assert result['valid_for_scoring']
        
    def test_consecutive_lock_invalid(self):
        """Art arda aynı hedefe lock geçersiz olmalı"""
        # İlk lock
        self.validator.register_successful_lock(target_id=1, timestamp=10.0)
        
        # Aynı hedefe tekrar lock kontrolü
        for i in range(300):
            t = 15.0 + i / 60.0
            frame_result = {'valid_frame': True}
            self.validator.add_frame_to_history(frame_result, target_id=1, sim_time=t)
            
        result = self.validator.check_continuous_lock(target_id=1, end_time=20.0)
        
        assert result['is_locked']
        assert not result['valid_for_scoring']  # Art arda = geçersiz
        
    def test_gap_breaks_continuous_lock(self):
        """Boşluk kesintisiz lock'u bozmalı"""
        # 2 saniye geçerli
        for i in range(120):
            t = i / 60.0
            frame_result = {'valid_frame': True}
            self.validator.add_frame_to_history(frame_result, target_id=1, sim_time=t)
            
        # 0.5 saniye boşluk
        
        # 2 saniye daha geçerli
        for i in range(120):
            t = 2.5 + i / 60.0
            frame_result = {'valid_frame': True}
            self.validator.add_frame_to_history(frame_result, target_id=1, sim_time=t)
            
        result = self.validator.check_continuous_lock(target_id=1, end_time=4.5)
        
        assert not result['is_locked']  # 4s kesintisiz yok
        assert result['continuous_duration'] < 3.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
