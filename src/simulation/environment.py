"""
Environment System

Simülasyon dünyası için gerçek arazi ve nesneler.
Heightmap tabanlı terrain ve statik world object'ler içerir.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class WorldObject:
    """Dünyadaki statik nesne (bina, ağaç vb.)"""
    
    obj_type: str  # "building", "tree", "rock"
    position: np.ndarray  # [x, y, z]
    rotation: float = 0.0  # Y ekseninde rotasyon (radyan)
    scale: float = 1.0
    size: Tuple[float, float, float] = (10.0, 20.0, 10.0)  # width, height, depth
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float32)
    
    def to_dict(self) -> Dict:
        return {
            'type': self.obj_type,
            'position': self.position.tolist(),
            'rotation': self.rotation,
            'scale': self.scale,
            'size': self.size,
            'color': self.color
        }


class Terrain:
    """
    Heightmap tabanlı arazi sistemi.
    
    Simülasyon dünyasının zemini. Düz veya tepelik olabilir.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Terrain boyutu (dünya koordinatlarında metre)
        self.size = config.get('size', [2000.0, 2000.0])
        
        # Heightmap çözünürlüğü
        self.resolution = config.get('resolution', 128)
        
        # Yükseklik aralığı
        self.min_height = config.get('min_height', 0.0)
        self.max_height = config.get('max_height', 50.0)
        
        # Texture ayarları
        self.texture_scale = config.get('texture_scale', 50.0)
        
        # Heightmap (başlangıçta düz)
        self.heightmap = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Terrain tipi
        terrain_type = config.get('type', 'flat')
        if terrain_type == 'flat':
            self.generate_flat()
        elif terrain_type == 'hills':
            self.generate_hills(config.get('seed', 42))
        elif terrain_type == 'valley':
            self.generate_valley(config.get('seed', 42))
    
    def generate_flat(self):
        """Düz zemin oluştur"""
        self.heightmap[:] = 0.0
    
    def generate_hills(self, seed: int = 42):
        """
        Tepelik arazi oluştur (Perlin noise benzeri).
        
        Args:
            seed: Random seed
        """
        np.random.seed(seed)
        
        # Multi-octave noise for natural looking terrain
        result = np.zeros((self.resolution, self.resolution))
        
        for octave in range(4):
            freq = 2 ** octave
            amplitude = 1.0 / freq
            
            # Simple noise interpolation
            noise_size = max(4, self.resolution // (2 ** (3 - octave)))
            noise = np.random.rand(noise_size, noise_size)
            
            # Bilinear upscale
            from scipy.ndimage import zoom
            scaled = zoom(noise, self.resolution / noise_size, order=1)
            
            result += scaled[:self.resolution, :self.resolution] * amplitude
        
        # Normalize to height range
        result = (result - result.min()) / (result.max() - result.min() + 1e-6)
        self.heightmap = result * (self.max_height - self.min_height) + self.min_height
    
    def generate_valley(self, seed: int = 42):
        """Vadi oluştur (ortası düşük, kenarlar yüksek)"""
        np.random.seed(seed)
        
        x = np.linspace(-1, 1, self.resolution)
        y = np.linspace(-1, 1, self.resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Parabolik vadi
        valley = (xx ** 2 + yy ** 2) * 0.5
        
        # Biraz noise ekle
        noise = np.random.rand(self.resolution, self.resolution) * 0.1
        
        self.heightmap = (valley + noise) * self.max_height
    
    def get_height_at(self, x: float, y: float) -> float:
        """WorldMap'ten kesin yüksekliği al"""
        from src.simulation.map_data import WorldMap
        return WorldMap.get_terrain_height(x, y)
    
    def get_normal_at(self, x: float, y: float) -> np.ndarray:
        """Terrain normal vektörünü hesapla"""
        eps = self.size[0] / self.resolution
        
        hL = self.get_height_at(x - eps, y)
        hR = self.get_height_at(x + eps, y)
        hD = self.get_height_at(x, y - eps)
        hU = self.get_height_at(x, y + eps)
        
        normal = np.array([hL - hR, 2 * eps, hD - hU])
        return normal / np.linalg.norm(normal)


class Environment:
    """
    Simülasyon dünyası environment'ı.
    
    Terrain ve tüm statik nesneleri yönetir.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Terrain oluştur
        terrain_config = config.get('terrain', {'type': 'flat'})
        self.terrain = Terrain(terrain_config)
        
        # World objects
        self.objects: List[WorldObject] = []
        
        # Auto-spawn settings
        if config.get('auto_spawn', True):
            self._spawn_default_objects(config)
    
    def _spawn_default_objects(self, config: dict):
        """WorldMap.STATIC_OBJECTS kullanarak nesneleri spawn et"""
        from src.simulation.map_data import WorldMap
        
        # MapData'daki statik nesneleri yükle
        for obj in WorldMap.STATIC_OBJECTS:
            if obj.obj_type == "building":
                self.spawn_building(
                    position=list(obj.position),
                    size=obj.size,
                    rotation=obj.rotation
                )
            # Ağaçlar vb. eklenebilir
        
        # Şimdilik ağaçlar MapData'da yoksa rastgele ekle (Görsel zenginlik için)
        # Ancak ideal olan MapData'ya taşımaktır.
        np.random.seed(config.get('seed', 42))
        num_trees = config.get('num_trees', 50)
        for i in range(num_trees):
            x = np.random.uniform(100, 1900)
            y = np.random.uniform(100, 1900)
            
            # Pist üzerine ağaç koyma (Basit kontrol)
            if 1300 < x < 1700 and 1300 < y < 1700:
                continue

            z = self.terrain.get_height_at(x, y)
            self.spawn_tree(
                position=[x, y, z],
                scale=np.random.uniform(0.8, 1.2)
            )
    
    def spawn_building(self, position: List[float], 
                       size: Tuple[float, float, float] = (30, 50, 30),
                       rotation: float = 0.0) -> WorldObject:
        """
        Bina spawn et.
        
        Args:
            position: [x, y, z] dünya koordinatları
            size: (width, height, depth)
            rotation: Y ekseninde rotasyon
        """
        building = WorldObject(
            obj_type='building',
            position=np.array(position),
            rotation=rotation,
            size=size,
            color=(0.6, 0.55, 0.5)  # Concrete grey
        )
        self.objects.append(building)
        return building
    
    def spawn_tree(self, position: List[float], scale: float = 1.0) -> WorldObject:
        """
        Ağaç spawn et.
        
        Args:
            position: [x, y, z] dünya koordinatları
            scale: Ağaç ölçeği
        """
        tree = WorldObject(
            obj_type='tree',
            position=np.array(position),
            scale=scale,
            size=(3 * scale, 12 * scale, 3 * scale),  # Trunk + canopy
            color=(0.2, 0.5, 0.2)  # Green
        )
        self.objects.append(tree)
        return tree
    
    def spawn_rock(self, position: List[float], scale: float = 1.0) -> WorldObject:
        """Kaya spawn et"""
        rock = WorldObject(
            obj_type='rock',
            position=np.array(position),
            scale=scale,
            size=(5 * scale, 3 * scale, 4 * scale),
            color=(0.4, 0.4, 0.4)
        )
        self.objects.append(rock)
        return rock
    
    def get_ground_height(self, x: float, y: float) -> float:
        """Terrain yüksekliğini al"""
        return self.terrain.get_height_at(x, y)
    
    def get_all_objects(self) -> List[WorldObject]:
        """Tüm world object'leri döndür"""
        return self.objects
    
    def get_objects_by_type(self, obj_type: str) -> List[WorldObject]:
        """Belirli tipteki nesneleri döndür"""
        return [obj for obj in self.objects if obj.obj_type == obj_type]
    
    def get_objects_in_radius(self, center: np.ndarray, radius: float) -> List[WorldObject]:
        """Belirli yarıçaptaki nesneleri döndür"""
        result = []
        for obj in self.objects:
            dist = np.linalg.norm(obj.position[:2] - center[:2])
            if dist <= radius:
                result.append(obj)
        return result
    
    def clear(self):
        """Tüm nesneleri temizle"""
        self.objects.clear()
    
    def to_dict(self) -> Dict:
        """Environment durumunu dict olarak döndür"""
        return {
            'terrain': {
                'size': self.terrain.size,
                'resolution': self.terrain.resolution,
                'min_height': self.terrain.min_height,
                'max_height': self.terrain.max_height
            },
            'objects': [obj.to_dict() for obj in self.objects]
        }
