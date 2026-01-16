"""
Render Modülü Test Suite

OffscreenRenderer, PostProcessing, CameraSimulation ve SceneManager testleri.
"""

import pytest
import numpy as np


class TestPostProcessingConfig:
    """PostProcessingConfig testleri"""
    
    def test_default_config(self):
        """Varsayılan konfigürasyon"""
        from src.render.post_processing import PostProcessingConfig
        
        config = PostProcessingConfig()
        
        assert config.lens_distortion == True
        assert config.sensor_noise == True
        assert config.vignette == True
        assert config.motion_blur == False
        assert config.k1 == -0.15
        
    def test_custom_config(self):
        """Özel konfigürasyon"""
        from src.render.post_processing import PostProcessingConfig
        
        config = PostProcessingConfig(
            lens_distortion=False,
            noise_sigma=10.0,
            vignette_strength=0.6
        )
        
        assert config.lens_distortion == False
        assert config.noise_sigma == 10.0
        assert config.vignette_strength == 0.6


class TestPostProcessing:
    """PostProcessing sınıfı testleri"""
    
    def setup_method(self):
        """Her test için setup"""
        from src.render.post_processing import PostProcessing, PostProcessingConfig
        self.config = PostProcessingConfig()
        self.pp = PostProcessing(self.config)
        
    def test_process_returns_same_shape(self):
        """İşlem sonucu aynı boyutta"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = self.pp.process(frame)
        
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        
    def test_process_empty_frame(self):
        """Boş frame işleme"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.pp.process(frame)
        
        assert result.shape == frame.shape
        
    def test_vignette_darkens_corners(self):
        """Vignette köşeleri karartıyor mu"""
        from src.render.post_processing import PostProcessing, PostProcessingConfig
        
        config = PostProcessingConfig(
            lens_distortion=False,
            sensor_noise=False,
            vignette=True
        )
        pp = PostProcessing(config)
        
        # Beyaz frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        result = pp.process(frame)
        
        # Merkez daha parlak olmalı
        center_brightness = result[240, 320].mean()
        corner_brightness = result[0, 0].mean()
        
        assert center_brightness > corner_brightness
        
    def test_noise_adds_variation(self):
        """Noise varyasyon ekliyor"""
        from src.render.post_processing import PostProcessing, PostProcessingConfig
        
        config = PostProcessingConfig(
            lens_distortion=False,
            sensor_noise=True,
            vignette=False,
            noise_sigma=10.0
        )
        pp = PostProcessing(config)
        
        # Düz gri frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = pp.process(frame)
        
        # Std > 0 olmalı
        assert result.std() > 0
        
    def test_update_config(self):
        """Konfigürasyon güncelleme"""
        self.pp.update_config(noise_sigma=20.0)
        
        assert self.pp.config.noise_sigma == 20.0


class TestCameraSimulation:
    """CameraSimulation sınıfı testleri"""
    
    def setup_method(self):
        from src.render.camera_simulation import CameraSimulation, CameraConfig
        self.config = CameraConfig()
        self.cam = CameraSimulation(self.config)
        
    def test_intrinsic_matrix(self):
        """İçsel matris hesaplaması"""
        K = self.cam.K
        
        assert K.shape == (3, 3)
        assert K[0, 0] > 0  # fx > 0
        assert K[1, 1] > 0  # fy > 0
        assert K[0, 2] == self.config.width / 2  # cx
        assert K[1, 2] == self.config.height / 2  # cy
        
    def test_focal_length_px(self):
        """Piksel cinsinden focal length"""
        fx = self.cam.focal_length_px
        
        # FOV = 60° → fx ≈ 0.866 * width
        expected = self.config.width / (2 * np.tan(np.radians(30)))
        
        assert abs(fx - expected) < 1
        
    def test_project_point_in_view(self):
        """Görünür noktayı projekte et"""
        camera_pos = np.array([0, 0, 100])
        camera_orient = np.array([0, 0, 0])  # İleri bak
        world_point = np.array([100, 0, 100])  # Önde
        
        result = self.cam.project_point(world_point, camera_pos, camera_orient)
        
        # Projeksiyon başarılı olmalı
        assert result is not None
        x, y = result
        assert 0 <= x < self.config.width
        assert 0 <= y < self.config.height
        
    def test_project_point_behind_camera(self):
        """Kameranın arkasındaki nokta"""
        camera_pos = np.array([100, 0, 100])
        camera_orient = np.array([0, 0, 0])
        world_point = np.array([0, 0, 100])  # Arkada
        
        result = self.cam.project_point(world_point, camera_pos, camera_orient)
        
        # Projeksiyon None olmalı
        assert result is None
        
    def test_get_camera_pose(self):
        """Kamera pozisyon/oryantasyon hesabı"""
        uav_pos = np.array([500, 500, 100])
        uav_orient = np.array([0, 0, np.pi/4])  # 45° heading
        
        cam_pos, cam_orient = self.cam.get_camera_pose(uav_pos, uav_orient)
        
        # Kamera UAV'ye yakın olmalı
        assert np.linalg.norm(cam_pos - uav_pos) < 1.0
        
    def test_is_in_fov(self):
        """Görüş alanı kontrolü"""
        camera_pos = np.array([0, 0, 100])
        camera_orient = np.array([0, 0, 0])
        
        # Önde - görünür
        point_front = np.array([100, 0, 100])
        assert self.cam.is_in_fov(point_front, camera_pos, camera_orient)
        
        # Arkada - görünmez
        point_back = np.array([-100, 0, 100])
        assert not self.cam.is_in_fov(point_back, camera_pos, camera_orient)
        
    def test_calculate_apparent_size(self):
        """Görünür boyut hesabı"""
        # 100m mesafede 2m nesne
        size = self.cam.calculate_apparent_size(100, 2.0)
        
        # Piksel cinsinden boyut > 0
        assert size > 0
        
        # Yakındaki nesne daha büyük görünmeli
        size_near = self.cam.calculate_apparent_size(50, 2.0)
        assert size_near > size


class TestSceneManager:
    """SceneManager sınıfı testleri"""
    
    def setup_method(self):
        from src.render.scene_manager import SceneManager
        self.scene = SceneManager()
        
    def test_add_uav(self):
        """İHA ekleme"""
        self.scene.add_uav('enemy_1', [100, 100, 100], heading=45, team='red')
        
        assert 'enemy_1' in self.scene.uavs
        assert self.scene.uavs['enemy_1']['team'] == 'red'
        
    def test_remove_uav(self):
        """İHA kaldırma"""
        self.scene.add_uav('enemy_1', [100, 100, 100])
        self.scene.remove_uav('enemy_1')
        
        assert 'enemy_1' not in self.scene.uavs
        
    def test_update_uav(self):
        """İHA güncelleme"""
        self.scene.add_uav('enemy_1', [100, 100, 100], heading=0)
        self.scene.update_uav('enemy_1', heading=90)
        
        assert self.scene.uavs['enemy_1']['heading'] == 90
        
    def test_add_air_defense_zone(self):
        """Hava savunma bölgesi ekleme"""
        self.scene.add_air_defense_zone(
            'zone_1', (500, 500), 100,
            zone_type='air_defense',
            activation_time=60
        )
        
        assert 'zone_1' in self.scene.air_defense_zones
        assert self.scene.air_defense_zones['zone_1']['radius'] == 100
        
    def test_get_visible_uavs_distance_filter(self):
        """Mesafe filtreleme"""
        self.scene.add_uav('near', [100, 100, 100])
        self.scene.add_uav('far', [10000, 10000, 100])
        
        camera_pos = np.array([0, 0, 100])
        visible = self.scene.get_visible_uavs(camera_pos, max_distance=500)
        
        ids = [u['id'] for u in visible]
        assert 'near' in ids
        assert 'far' not in ids
        
    def test_get_active_zones_time_based(self):
        """Zaman tabanlı aktivasyon"""
        self.scene.add_air_defense_zone(
            'zone_1', (500, 500), 100,
            activation_time=60,
            deactivation_time=120
        )
        
        # t=30: henüz aktif değil
        active = self.scene.get_active_zones(30)
        assert len(active) == 0
        
        # t=90: aktif
        active = self.scene.get_active_zones(90)
        assert len(active) == 1
        
        # t=150: deaktif
        active = self.scene.get_active_zones(150)
        assert len(active) == 0
        
    def test_check_zone_violation(self):
        """Bölge ihlali kontrolü"""
        self.scene.add_air_defense_zone('zone_1', (500, 500), 100, is_active=True)
        
        # İçeride
        inside = np.array([500, 500, 100])
        violation = self.scene.check_zone_violation(inside)
        assert violation is not None
        
        # Dışarıda
        outside = np.array([700, 700, 100])
        violation = self.scene.check_zone_violation(outside)
        assert violation is None
        
    def test_load_from_scenario(self):
        """Senaryo yükleme"""
        scenario = {
            'world': {'size': 2000},
            'air_defense': {
                'enabled': True,
                'zones': [
                    {'id': 'z1', 'center': [500, 500], 'radius': 100, 'activation_time': 60, 'duration': 30}
                ]
            }
        }
        
        self.scene.load_from_scenario(scenario)
        
        assert 'z1' in self.scene.air_defense_zones


class TestFallbackRenderer:
    """FallbackRenderer testleri"""
    
    def test_render_frame_returns_valid_array(self):
        """Geçerli numpy array döndürür"""
        from src.render.offscreen_renderer import FallbackRenderer
        
        renderer = FallbackRenderer(640, 480)
        
        frame = renderer.render_frame(
            np.array([0, 0, 100]),
            np.array([0, 0, 0]),
            [{'id': 'e1', 'position': [100, 100, 100], 'heading': 0}]
        )
        
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        
    def test_resolution_property(self):
        """Resolution property"""
        from src.render.offscreen_renderer import FallbackRenderer
        
        renderer = FallbackRenderer(800, 600)
        
        assert renderer.resolution == (800, 600)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
