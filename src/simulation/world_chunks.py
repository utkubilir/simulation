"""
World Chunks - Chunk-Based Infinite Terrain System

İHA ilerledikçe genişleyen sonsuz dünya için chunk yönetimi.
Her chunk bağımsız olarak yüklenebilir/kaldırılabilir.
"""

import numpy as np
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TerrainChunk:
    """Tek bir terrain chunk'ı"""
    
    coords: Tuple[int, int]  # Chunk koordinatları (cx, cy)
    chunk_size: float = 500.0  # Metre cinsinden chunk boyutu
    resolution: int = 32  # Heightmap çözünürlüğü
    
    # GPU kaynakları (renderer tarafından doldurulur)
    vao: object = None
    vbo: object = None
    vertex_count: int = 0
    
    # Heightmap verisi
    heightmap: np.ndarray = field(default_factory=lambda: None)
    
    def __post_init__(self):
        if self.heightmap is None:
            self.heightmap = np.zeros((self.resolution, self.resolution), dtype=np.float32)
    
    @property 
    def world_origin(self) -> Tuple[float, float]:
        """Chunk'ın dünya koordinatlarındaki başlangıç noktası"""
        return (
            self.coords[0] * self.chunk_size,
            self.coords[1] * self.chunk_size
        )
    
    @property
    def world_center(self) -> Tuple[float, float]:
        """Chunk'ın dünya koordinatlarındaki merkezi"""
        ox, oy = self.world_origin
        half = self.chunk_size / 2
        return (ox + half, oy + half)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Nokta bu chunk içinde mi?"""
        ox, oy = self.world_origin
        return (ox <= x < ox + self.chunk_size and 
                oy <= y < oy + self.chunk_size)


class ChunkManager:
    """
    Chunk yönetim sistemi.
    
    İHA pozisyonuna göre chunk'ları dinamik olarak yükler/kaldırır.
    """
    
    def __init__(self, chunk_size: float = 500.0, view_distance: int = 3, 
                 chunk_resolution: int = 32, seed: int = 42):
        """
        Args:
            chunk_size: Her chunk'ın boyutu (metre)
            view_distance: Görüş mesafesi (chunk sayısı)
            chunk_resolution: Chunk heightmap çözünürlüğü
            seed: Procedural generation seed
        """
        self.chunk_size = chunk_size
        self.view_distance = view_distance
        self.chunk_resolution = chunk_resolution
        self.seed = seed
        
        # Yüklü chunk'lar: {(cx, cy): TerrainChunk}
        self.loaded_chunks: Dict[Tuple[int, int], TerrainChunk] = {}
        
        # Son güncellenen merkez chunk
        self._last_center: Optional[Tuple[int, int]] = None
        
        # Callback: Yeni chunk yüklendiğinde çağrılır (renderer için)
        self.on_chunk_loaded = None
        self.on_chunk_unloaded = None
    
    def get_chunk_coords(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Dünya koordinatlarından chunk koordinatlarına dönüşüm"""
        return (
            int(np.floor(world_x / self.chunk_size)),
            int(np.floor(world_y / self.chunk_size))
        )
    
    def update(self, player_x: float, player_y: float) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Oyuncu pozisyonuna göre chunk'ları güncelle.
        
        Args:
            player_x, player_y: Oyuncu dünya koordinatları
            
        Returns:
            (yeni_yüklenen, kaldırılan) chunk koordinatları
        """
        center = self.get_chunk_coords(player_x, player_y)
        
        # Merkez değişmediyse bir şey yapma
        if center == self._last_center:
            return set(), set()
        
        self._last_center = center
        
        # Hangi chunk'lar yüklü olmalı?
        required = self._get_required_chunks(center)
        current = set(self.loaded_chunks.keys())
        
        # Yeni chunk'ları yükle
        to_load = required - current
        for coords in to_load:
            chunk = self._generate_chunk(coords)
            self.loaded_chunks[coords] = chunk
            if self.on_chunk_loaded:
                self.on_chunk_loaded(chunk)
        
        # Uzaktaki chunk'ları kaldır
        to_unload = current - required
        for coords in to_unload:
            chunk = self.loaded_chunks.pop(coords)
            if self.on_chunk_unloaded:
                self.on_chunk_unloaded(chunk)
        
        return to_load, to_unload
    
    def _get_required_chunks(self, center: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Merkez etrafındaki gerekli chunk koordinatları"""
        required = set()
        cx, cy = center
        d = self.view_distance
        
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                required.add((cx + dx, cy + dy))
        
        return required
    
    def _generate_chunk(self, coords: Tuple[int, int]) -> TerrainChunk:
        """Yeni chunk oluştur ve heightmap'ini doldur"""
        chunk = TerrainChunk(
            coords=coords,
            chunk_size=self.chunk_size,
            resolution=self.chunk_resolution
        )
        
        # Heightmap'i doldur (seed-based procedural generation)
        self._fill_heightmap(chunk)
        
        return chunk
    
    def _fill_heightmap(self, chunk: TerrainChunk):
        """Chunk heightmap'ini procedural noise ile doldur"""
        ox, oy = chunk.world_origin
        res = chunk.resolution
        step = chunk.chunk_size / (res - 1)
        
        for i in range(res):
            for j in range(res):
                world_x = ox + i * step
                world_y = oy + j * step
                
                # Perlin noise ile yükseklik hesapla
                chunk.heightmap[i, j] = self._get_height_at(world_x, world_y)
    
    def _get_height_at(self, x: float, y: float) -> float:
        """
        Belirli koordinattaki zemin yüksekliği.
        Tutarlı olması için seed-based Perlin noise kullanır.
        """
        # Multi-octave noise
        noise = 0.0
        amplitude = 1.0
        frequency = 0.002
        
        # Basit sin/cos tabanlı noise (tutarlı ve deterministik)
        for octave in range(4):
            # Seed'i octave'a göre offset'le
            seed_offset = self.seed + octave * 1000
            
            nx = x * frequency + seed_offset * 0.1
            ny = y * frequency + seed_offset * 0.2
            
            noise += amplitude * (np.sin(nx) * np.cos(ny) + np.sin(nx * 1.5 + ny * 0.7))
            
            amplitude *= 0.5
            frequency *= 2
        
        # Normalize ve ölçekle (0-50m arası tepeler)
        height = (noise + 2) * 12.5  # -2..2 -> 0..50
        
        # Minimum yükseklik 0
        return max(0.0, height)
    
    def get_height_at(self, x: float, y: float) -> float:
        """Public API: Herhangi bir koordinattaki yükseklik"""
        return self._get_height_at(x, y)
    
    def get_visible_chunks(self) -> list:
        """Şu an yüklü olan tüm chunk'lar"""
        return list(self.loaded_chunks.values())
    
    def get_chunk_at(self, x: float, y: float) -> Optional[TerrainChunk]:
        """Belirli koordinattaki chunk (yüklüyse)"""
        coords = self.get_chunk_coords(x, y)
        return self.loaded_chunks.get(coords)
