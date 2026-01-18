"""
Map Data & World Generation - Single Source of Truth

Simülasyon dünyasının (hem fiziksel hem görsel) tekil veri kaynağıdır.
Arazi yüksekliği, statik nesneler ve dünya sınırları burada tanımlanır.
Koordinat Sistemi: 0..2000 (x, y) metre.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# Dünya Boyutları
WORLD_WIDTH = 2000.0
WORLD_DEPTH = 2000.0
WORLD_HEIGHT_LIMIT = 500.0

@dataclass
class StaticObject:
    """Dünyadaki statik nesne tanımı"""
    obj_type: str        # "building", "tree", "rock", "runway", "mountain_cone"
    position: Tuple[float, float, float] # x, y, z
    size: Tuple[float, float, float] = (1, 1, 1) # width, depth, height
    rotation: float = 0.0 # yaw (degrees)
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    meta: dict = None    # Ekstra parametreler (örn: mountain height/radius)

class WorldMap:
    """
    Dünya Haritası Verisi
    """

    # Pist Merkezi (Visual'da 500,500 idi, 0..2000 sisteminde 1500,1500'e denk gelir)
    RUNWAY_CENTER = (1500.0, 1500.0)

    # Statik Nesneler Listesi
    STATIC_OBJECTS: List[StaticObject] = []

    @staticmethod
    def _init_static_objects():
        """Statik nesneleri başlat"""
        WorldMap.STATIC_OBJECTS = []

        # --- Pist ---
        # Görsel terrain.py'de pist çizgileri ve zemin vardı.
        # Burada sadece mantıksal tanımını yapıyoruz, görsel yine çizebilir.
        # Pist çevresindeki binalar (Visual koordinatlar + 1000)

        # Kontrol Kulesi (600, 450) -> (1600, 1450)
        WorldMap.STATIC_OBJECTS.append(StaticObject(
            "building", (1600, 1450, 0), (8, 8, 25), 0, (0.5, 0.5, 0.55, 1)
        ))

        # Hangarlar
        # (650, 520) -> (1650, 1520)
        WorldMap.STATIC_OBJECTS.append(StaticObject(
            "building", (1650, 1520, 0), (40, 25, 12), 0, (0.6, 0.55, 0.5, 1)
        ))
        # (650, 560) -> (1650, 1560)
        WorldMap.STATIC_OBJECTS.append(StaticObject(
            "building", (1650, 1560, 0), (40, 25, 12), 0, (0.55, 0.5, 0.45, 1)
        ))

        # Depolar
        WorldMap.STATIC_OBJECTS.append(StaticObject(
            "building", (1700, 1480, 0), (20, 15, 8), 0, (0.5, 0.45, 0.4, 1)
        ))
        WorldMap.STATIC_OBJECTS.append(StaticObject(
            "building", (1720, 1520, 0), (15, 15, 6), 0, (0.45, 0.4, 0.38, 1)
        ))

        # --- Dağlar (Ekstra Koniler) ---
        # terrain.py: (-800, -600) -> (200, 400)
        mountains = [
            (200, 400, 0, 150),   # x, y, z, height
            (400, 1800, 0, 120),  # (-600, 800) -> (400, 1800)
            (1900, 300, 0, 180),  # (900, -700) -> (1900, 300)
            (1700, 1900, 0, 140), # (700, 900) -> (1700, 1900)
            (100, 1200, 0, 100),  # (-900, 200) -> (100, 1200)
        ]

        for mx, my, mz, h in mountains:
            # Radius approx height * 1.5 (from terrain.py)
            radius = h * 1.5
            WorldMap.STATIC_OBJECTS.append(StaticObject(
                "mountain_cone", (mx, my, mz), (radius*2, radius*2, h), 0,
                (0.5, 0.5, 0.5, 1), # Renk visualizer'da halledilir
                meta={'height': h, 'radius': radius}
            ))

    _NOISE_CACHE = {}

    @staticmethod
    def _get_noise_params(seed):
        if seed not in WorldMap._NOISE_CACHE:
             rng = np.random.RandomState(seed)
             phases = []
             for _ in range(4):
                 phases.append((rng.rand() * 100, rng.rand() * 100))
             WorldMap._NOISE_CACHE[seed] = phases
        return WorldMap._NOISE_CACHE[seed]

    @staticmethod
    def _perlin_noise(x, y, seed=42):
        """Terrain.py'den alınan noise fonksiyonu - Local RandomState + Cache"""
        phases = WorldMap._get_noise_params(seed)
        noise = 0
        amplitude = 1
        frequency = 0.002
        for px, py in phases:
            noise += amplitude * np.sin(x * frequency + px) * \
                     np.cos(y * frequency + py)
            amplitude *= 0.5
            frequency *= 2
        return noise

    @staticmethod
    def get_terrain_height(x: float, y: float) -> float:
        """
        Belirtilen koordinattaki zemin yüksekliğini döndürür.
        SONSUZ KOORDİNAT DESTEKLİ: Her koordinat için tutarlı yükseklik üretir.
        """
        # Pist merkezi (1500, 1500) etrafındaki alan düz olmalı
        RUNWAY_CENTER_X, RUNWAY_CENTER_Y = 1500.0, 1500.0
        dist_runway = np.sqrt((x - RUNWAY_CENTER_X)**2 + (y - RUNWAY_CENTER_Y)**2)
        
        if dist_runway < 300:
            base_h = 0.0
        else:
            # Perlin Noise - sonsuz koordinatlar için çalışır
            # Offset ekleyerek negatif değerleri pozitife çeviriyoruz
            noise = WorldMap._perlin_noise(x, y) * 30
            noise += WorldMap._perlin_noise(x * 2, y * 2, seed=123) * 15
            base_h = noise + 25  # Offset: Temel zemin seviyesi
            
            # Pist etrafında yükselme efekti (sadece orijin bölgesinde)
            if 0 <= x <= 2000 and 0 <= y <= 2000:
                x_orig = x - 1000.0
                y_orig = y - 1000.0
                dist_edge = min(abs(x_orig + 1000), abs(x_orig - 1000),
                                abs(y_orig + 1000), abs(y_orig - 1000))
                if dist_edge < 400:
                    base_h += (400 - dist_edge) * 0.15

        base_h = max(0.0, base_h)

        # Dağlar (sadece orijin bölgesinde)
        max_cone_h = 0.0
        if 0 <= x <= 2000 and 0 <= y <= 2000:
            if not WorldMap.STATIC_OBJECTS:
                WorldMap._init_static_objects()

            for obj in WorldMap.STATIC_OBJECTS:
                if obj.obj_type == "mountain_cone":
                    dx = x - obj.position[0]
                    dy = y - obj.position[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    r = obj.meta['radius']
                    h = obj.meta['height']

                    if dist < r:
                        cone_h = h * (1.0 - dist / r)
                        if cone_h > max_cone_h:
                            max_cone_h = cone_h

        return max(base_h, max_cone_h)

# Modül yüklenince listeyi doldur
WorldMap._init_static_objects()
