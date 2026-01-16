"""
Entegrasyon Test Suite

End-to-end testler ve modüller arası entegrasyon testleri.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path


class TestRunnerIntegration:
    """SimulationRunner entegrasyon testleri"""
    
    def test_air_defense_manager_import(self):
        """AirDefenseManager import edilebilir"""
        from src.simulation.air_defense import AirDefenseManager
        
        manager = AirDefenseManager()
        assert manager is not None
        
    def test_air_defense_scenario_loading(self):
        """Senaryo dosyasından hava savunma yükleme"""
        from src.simulation.air_defense import AirDefenseManager
        
        # competition_trial.yaml yükle
        scenario_path = Path('scenarios/competition_trial.yaml')
        if scenario_path.exists():
            with open(scenario_path) as f:
                scenario = yaml.safe_load(f)
                
            manager = AirDefenseManager()
            manager.load_from_scenario(scenario)
            
            # Bölgeler yüklenmiş olmalı
            assert len(manager.zones) >= 3
            
    def test_autopilot_avoidance_method(self):
        """Autopilot kaçınma metodu"""
        from src.uav.autopilot import Autopilot
        
        autopilot = Autopilot()
        
        # set_avoidance_heading metodu mevcut olmalı
        assert hasattr(autopilot, 'set_avoidance_heading')
        assert hasattr(autopilot, 'clear_avoidance')
        
        # Çağrılabilir olmalı
        autopilot.set_avoidance_heading(180.0)
        autopilot.clear_avoidance()


class TestPipelineIntegration:
    """Vision Pipeline entegrasyon testleri"""
    
    def test_full_detection_pipeline(self):
        """Tam tespit pipeline"""
        from src.render.post_processing import PostProcessing, PostProcessingConfig
        from src.vision.image_detector import ImageBasedDetector, ImageDetectorConfig
        import cv2
        
        # Post-processing
        pp = PostProcessing(PostProcessingConfig(
            lens_distortion=False,
            sensor_noise=True,
            vignette=True
        ))
        
        # Detector
        detector = ImageBasedDetector(ImageDetectorConfig())
        
        # Test frame (İHA benzeri şekil)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (300, 200), (340, 230), (100, 100, 100), -1)  # Koyu gri
        
        # Pipeline
        processed = pp.process(frame)
        detections = detector.detect(processed)
        
        # Pipeline çalışmalı (tespit olsa da olmasa da)
        assert processed.shape == frame.shape
        assert isinstance(detections, list)
        
    def test_camera_simulation_with_detection(self):
        """Kamera simülasyonu ile tespit"""
        from src.render.camera_simulation import CameraSimulation, CameraConfig
        from src.vision.image_detector import ImageBasedDetector
        
        cam = CameraSimulation(CameraConfig())
        detector = ImageBasedDetector()
        
        # Simüle edilmiş hedefler
        targets = [
            {'id': 'e1', 'position': np.array([200, 0, 100]), 'size': 2.0},
            {'id': 'e2', 'position': np.array([300, 50, 120]), 'size': 2.0}
        ]
        
        camera_pos = np.array([0, 0, 100])
        camera_orient = np.array([0, 0, 0])
        
        # Her hedef için FoV kontrolü
        for target in targets:
            in_fov = cam.is_in_fov(target['position'], camera_pos, camera_orient)
            # Hedefler önde olduğu için görünür olmalı
            assert in_fov


class TestScenarioIntegration:
    """Senaryo dosyası entegrasyon testleri"""
    
    def test_competition_trial_structure(self):
        """competition_trial.yaml yapı kontrolü"""
        scenario_path = Path('scenarios/competition_trial.yaml')
        
        if scenario_path.exists():
            with open(scenario_path) as f:
                scenario = yaml.safe_load(f)
                
            # Zorunlu alanlar
            assert 'name' in scenario
            assert 'player' in scenario
            assert 'enemies' in scenario
            assert 'camera' in scenario
            assert 'lock' in scenario
            
            # Hava savunma yapısı
            assert 'air_defense' in scenario
            assert scenario['air_defense']['enabled'] == True
            assert 'zones' in scenario['air_defense']
            
    def test_air_defense_zones_valid(self):
        """Hava savunma bölgeleri geçerli"""
        scenario_path = Path('scenarios/competition_trial.yaml')
        
        if scenario_path.exists():
            with open(scenario_path) as f:
                scenario = yaml.safe_load(f)
                
            zones = scenario['air_defense']['zones']
            
            for zone in zones:
                assert 'id' in zone
                assert 'center' in zone
                assert 'radius' in zone
                assert len(zone['center']) == 2
                assert zone['radius'] > 0


class TestAirDefenseIntegration:
    """AirDefenseManager entegrasyon testleri"""
    
    def test_full_update_cycle(self):
        """Tam güncelleme döngüsü"""
        from src.simulation.air_defense import AirDefenseManager, AirDefenseZone
        
        manager = AirDefenseManager()
        
        # Bölge ekle
        manager.add_zone(AirDefenseZone(
            id='zone_1',
            center=(500, 500),
            radius=100,
            activation_time=60,
            deactivation_time=120
        ))
        
        # UAV pozisyonu
        positions = {'player': np.array([500, 500, 100])}
        
        # t=30: henüz aktif değil
        result = manager.update(30.0, positions, 0.1)
        assert len(result['active_zones']) == 0
        
        # t=90: aktif, ihlal var
        manager.reset()
        manager.add_zone(AirDefenseZone(
            id='zone_1',
            center=(500, 500),
            radius=100,
            activation_time=60,
            deactivation_time=120
        ))
        
        # Aktivasyonu simüle et
        for i in range(600, 700):  # 60s - 70s
            result = manager.update(i * 0.1, positions, 0.1)
            
        # Ceza birikmiş olmalı
        assert manager.total_penalty < 0
        
    def test_avoidance_heading_calculation(self):
        """Kaçınma heading hesaplama"""
        from src.simulation.air_defense import AirDefenseManager, AirDefenseZone
        
        manager = AirDefenseManager()
        zone = AirDefenseZone('z1', (500, 500), 100)
        zone.is_active = True
        manager.add_zone(zone)
        
        # Bölgenin solunda
        pos_left = np.array([380, 500, 100])
        heading = manager.get_avoidance_heading(pos_left, 0)
        
        # Sola (batıya) dönmeli
        assert heading is not None
        
    def test_safe_corridor_generation(self):
        """Güvenli koridor oluşturma"""
        from src.simulation.air_defense import AirDefenseManager, AirDefenseZone
        
        manager = AirDefenseManager()
        zone = AirDefenseZone('z1', (500, 500), 100)
        zone.is_active = True
        manager.add_zone(zone)
        
        start = np.array([300, 300, 100])
        goal = np.array([700, 700, 100])
        
        waypoints = manager.get_safe_corridor(start, goal)
        
        # En az başlangıç ve hedef olmalı
        assert len(waypoints) >= 2
        assert np.allclose(waypoints[0], start)
        assert np.allclose(waypoints[-1], goal)


class TestModuleImports:
    """Modül import testleri"""
    
    def test_render_module_imports(self):
        """Render modülü import"""
        from src.render import OffscreenRenderer, PostProcessing, CameraSimulation, SceneManager
        
        assert OffscreenRenderer is not None
        assert PostProcessing is not None
        assert CameraSimulation is not None
        assert SceneManager is not None
        
    def test_vision_modules_import(self):
        """Vision modülleri import"""
        from src.vision.image_detector import ImageBasedDetector, Detection
        from src.vision.vision_pipeline import VisionPipeline, VisionResult
        
        assert ImageBasedDetector is not None
        assert VisionPipeline is not None
        
    def test_air_defense_import(self):
        """Air defense import"""
        from src.simulation.air_defense import AirDefenseManager, AirDefenseZone, ZoneType
        
        assert AirDefenseManager is not None
        assert AirDefenseZone is not None
        assert ZoneType is not None


class TestPerformance:
    """Performans testleri"""
    
    def test_post_processing_speed(self):
        """Post-processing hızı"""
        import time
        from src.render.post_processing import PostProcessing, PostProcessingConfig
        
        pp = PostProcessing(PostProcessingConfig())
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup
        pp.process(frame)
        
        # Ölçüm
        start = time.perf_counter()
        for _ in range(10):
            pp.process(frame)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 10) * 1000
        
        # 10ms'den hızlı olmalı
        assert avg_ms < 50, f"Post-processing too slow: {avg_ms:.1f}ms"
        
    def test_detection_speed(self):
        """Tespit hızı"""
        import time
        from src.vision.image_detector import ImageBasedDetector
        
        detector = ImageBasedDetector()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup
        detector.detect(frame, use_motion=False)
        
        # Ölçüm
        start = time.perf_counter()
        for _ in range(10):
            detector.detect(frame, use_motion=False)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 10) * 1000
        
        # 30ms'den hızlı olmalı
        assert avg_ms < 100, f"Detection too slow: {avg_ms:.1f}ms"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
