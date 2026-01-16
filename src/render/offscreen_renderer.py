"""
Panda3D Offscreen Renderer

Görünmez pencerede 3D render yaparak numpy array döndürür.
Gerçekçi kamera görüntüsü oluşturur.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import threading
import queue

# Panda3D imports
try:
    from panda3d.core import (
        loadPrcFileData,
        GraphicsEngine, GraphicsPipe, GraphicsPipeSelection,
        FrameBufferProperties, WindowProperties,
        Texture, GraphicsOutput,
        NodePath, Camera, Lens, PerspectiveLens,
        DirectionalLight, AmbientLight, PointLight,
        Vec3, Vec4, Point3, LVector3f,
        TransparencyAttrib, AntialiasAttrib,
        ClockObject
    )
    from direct.showbase.ShowBase import ShowBase
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False
    print("⚠️ Panda3D not available - using fallback renderer")


class OffscreenRenderer:
    """
    Panda3D ile offscreen (görünmez pencerede) 3D render
    
    Features:
    - Gerçek 3D geometri render
    - Dinamik aydınlatma ve gölgeler
    - Atmosferik efektler
    - GPU hızlandırmalı
    
    Usage:
        renderer = OffscreenRenderer(640, 480)
        renderer.setup_scene()
        frame = renderer.render_frame(camera_pos, camera_orient, uav_states)
    """
    
    def __init__(self, width: int = 640, height: int = 480, 
                 headless: bool = True, fov: float = 60.0):
        """
        Offscreen renderer başlat
        
        Args:
            width: Frame genişliği
            height: Frame yüksekliği
            headless: Görünmez pencere modu
            fov: Görüş alanı (derece)
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.headless = headless
        
        self._initialized = False
        self._frame_count = 0
        
        # Scene objects
        self.terrain = None
        self.sky = None
        self.uav_models: Dict[str, NodePath] = {}
        self.air_defense_visuals: Dict[str, NodePath] = {}
        
        # Panda3D objects (lazy init)
        self.base = None
        self.engine = None
        self.buffer = None
        self.texture = None
        self.camera = None
        self.render = None
        
        # Thread safety
        self._lock = threading.Lock()
        
    def initialize(self):
        """Panda3D render sistemini başlat"""
        if self._initialized:
            return True
            
        if not PANDA3D_AVAILABLE:
            print("❌ Panda3D not available")
            return False
            
        try:
            # Headless mode konfigürasyonu
            loadPrcFileData('', 'window-type offscreen')
            loadPrcFileData('', 'audio-library-name null')
            loadPrcFileData('', 'load-display pandagl')
            loadPrcFileData('', f'win-size {self.width} {self.height}')
            
            # ShowBase başlat
            self.base = ShowBase(windowType='offscreen')
            self.render = self.base.render
            
            # Offscreen buffer oluştur
            self._setup_offscreen_buffer()
            
            # Kamera ayarla
            self._setup_camera()
            
            # Aydınlatma
            self._setup_lighting()
            
            self._initialized = True
            print(f"✅ OffscreenRenderer initialized ({self.width}x{self.height})")
            return True
            
        except Exception as e:
            print(f"❌ OffscreenRenderer init failed: {e}")
            return False
            
    def _setup_offscreen_buffer(self):
        """Offscreen render buffer oluştur"""
        # Frame buffer özellikleri
        fb_props = FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.setRgbaBits(8, 8, 8, 8)
        fb_props.setDepthBits(24)
        fb_props.setMultisamples(4)  # Anti-aliasing
        
        # Pencere özellikleri
        win_props = WindowProperties()
        win_props.setSize(self.width, self.height)
        
        # Graphics engine
        self.engine = self.base.graphicsEngine
        pipe = self.base.pipe
        
        # Buffer oluştur
        self.buffer = self.engine.makeOutput(
            pipe, "offscreen_buffer", -2,
            fb_props, win_props,
            GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFRequireCallbackWindow,
            self.base.win.getGsg(), self.base.win
        )
        
        if self.buffer is None:
            # Fallback: basit buffer
            self.buffer = self.base.win
            
        # Texture bağla (render to texture)
        self.texture = Texture()
        self.buffer.addRenderTexture(
            self.texture, 
            GraphicsOutput.RTMCopyRam,
            GraphicsOutput.RTPColor
        )
        
    def _setup_camera(self):
        """Render kamerası ayarla"""
        # Lens
        lens = PerspectiveLens()
        lens.setFov(self.fov)
        lens.setNear(1.0)
        lens.setFar(10000.0)
        lens.setAspectRatio(self.width / self.height)
        
        # Kamera node
        self.camera = self.render.attachNewNode(Camera('offscreen_cam'))
        self.camera.node().setLens(lens)
        
        # Display region
        if self.buffer:
            dr = self.buffer.makeDisplayRegion()
            dr.setCamera(self.camera)
            
    def _setup_lighting(self):
        """Sahne aydınlatması"""
        # Güneş (directional light)
        sun = DirectionalLight('sun')
        sun.setColor(Vec4(1.0, 0.95, 0.9, 1))
        sun.setShadowCaster(True, 1024, 1024)
        
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)  # Azimuth, Elevation
        self.render.setLight(sun_np)
        
        # Ortam ışığı
        ambient = AmbientLight('ambient')
        ambient.setColor(Vec4(0.35, 0.35, 0.4, 1))
        self.render.setLight(self.render.attachNewNode(ambient))
        
        # Fill light (gölgeleri yumuşat)
        fill = DirectionalLight('fill')
        fill.setColor(Vec4(0.2, 0.25, 0.3, 1))
        fill_np = self.render.attachNewNode(fill)
        fill_np.setHpr(-135, -30, 0)
        self.render.setLight(fill_np)
        
    def setup_scene(self, terrain_config: dict = None):
        """
        Sahne elemanlarını yükle
        
        Args:
            terrain_config: Arazi konfigürasyonu
        """
        if not self._initialized:
            if not self.initialize():
                return False
                
        try:
            # Arazi
            from src.simulation.terrain import TerrainGenerator
            self.terrain = TerrainGenerator.create_terrain(
                self.render,
                size=terrain_config.get('size', 2000) if terrain_config else 2000,
                resolution=terrain_config.get('resolution', 50) if terrain_config else 50
            )
            
            # Gökyüzü (basit gradient için kart)
            self._create_sky()
            
            print("✅ Scene setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Scene setup failed: {e}")
            return False
            
    def _create_sky(self):
        """Gökyüzü dome/kart oluştur"""
        from panda3d.core import CardMaker
        
        # Basit arka plan rengi
        self.base.setBackgroundColor(0.5, 0.7, 0.9, 1)  # Açık mavi
        
    def add_uav_model(self, uav_id: str, team: str = 'red', 
                      position: np.ndarray = None):
        """
        İHA modeli ekle
        
        Args:
            uav_id: Benzersiz ID
            team: Takım rengi ('red', 'blue', 'green')
            position: Başlangıç pozisyonu [x, y, z]
        """
        if not self._initialized:
            return None
            
        try:
            from src.simulation.uav_model_3d import DetailedUAVModel
            
            model = DetailedUAVModel.create(
                self.render,
                name=f"uav_{uav_id}",
                team=team,
                scale=1.0
            )
            
            if position is not None:
                model.setPos(position[0], position[1], position[2])
                
            self.uav_models[uav_id] = model
            return model
            
        except Exception as e:
            print(f"❌ UAV model creation failed: {e}")
            return None
            
    def update_uav(self, uav_id: str, position: np.ndarray, 
                   heading: float, pitch: float = 0, roll: float = 0):
        """
        İHA pozisyon ve oryantasyonunu güncelle
        
        Args:
            uav_id: İHA ID
            position: [x, y, z]
            heading: Yaw açısı (derece)
            pitch: Pitch açısı (derece)
            roll: Roll açısı (derece)
        """
        if uav_id not in self.uav_models:
            return
            
        model = self.uav_models[uav_id]
        model.setPos(position[0], position[1], position[2])
        model.setHpr(heading, pitch, roll)
        
    def add_air_defense_zone(self, zone_id: str, center: Tuple[float, float],
                              radius: float, is_active: bool = True):
        """
        Hava savunma bölgesi görselleştirmesi ekle
        
        Args:
            zone_id: Bölge ID
            center: (x, y) merkez
            radius: Yarıçap (metre)
            is_active: Aktif mi?
        """
        if not self._initialized:
            return
            
        from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter
        from panda3d.core import Geom, GeomLines, GeomNode
        
        # Daire çiz
        format = GeomVertexFormat.get_v3c4()
        vdata = GeomVertexData('circle', format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, 'vertex')
        color = GeomVertexWriter(vdata, 'color')
        
        segments = 32
        zone_color = (1, 0.2, 0.2, 0.5) if is_active else (0.5, 0.5, 0.5, 0.3)
        
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            vertex.addData3(x, y, 1)  # Hafif yerden yukarı
            color.addData4(*zone_color)
            
        prim = GeomLines(Geom.UHStatic)
        for i in range(segments):
            prim.addVertices(i, i + 1)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode(f'ad_zone_{zone_id}')
        node.addGeom(geom)
        
        np_zone = self.render.attachNewNode(node)
        np_zone.setTransparency(TransparencyAttrib.MAlpha)
        
        self.air_defense_visuals[zone_id] = np_zone
        
    def render_frame(self, 
                     camera_pos: np.ndarray,
                     camera_orient: np.ndarray,
                     uav_states: List[dict] = None) -> np.ndarray:
        """
        Tek frame render et
        
        Args:
            camera_pos: Kamera pozisyonu [x, y, z]
            camera_orient: Kamera oryantasyonu [roll, pitch, yaw] (radyan)
            uav_states: İHA durumları listesi
            
        Returns:
            RGB numpy array (height, width, 3)
        """
        if not self._initialized:
            # Fallback: boş frame
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        with self._lock:
            # 1. Kamerayı konumlandır
            self._update_camera(camera_pos, camera_orient)
            
            # 2. İHA modellerini güncelle
            if uav_states:
                self._update_uav_models(uav_states)
                
            # 3. Render
            self.engine.renderFrame()
            
            # 4. Texture'dan numpy array'e
            return self._texture_to_numpy()
            
    def _update_camera(self, pos: np.ndarray, orient: np.ndarray):
        """Kamera pozisyonunu güncelle"""
        if self.camera is None:
            return
            
        roll, pitch, yaw = orient
        
        # Panda3D koordinat sistemi: Y=forward, Z=up
        # Bizim sistem: X=forward, Y=left, Z=up
        # Dönüşüm gerekli
        
        self.camera.setPos(pos[0], pos[1], pos[2])
        
        # HPR: Heading (yaw), Pitch, Roll
        self.camera.setHpr(
            np.degrees(yaw),
            np.degrees(pitch),
            np.degrees(roll)
        )
        
    def _update_uav_models(self, uav_states: List[dict]):
        """Tüm İHA modellerini güncelle"""
        for uav in uav_states:
            uav_id = str(uav.get('id', 'unknown'))
            
            # Model yoksa oluştur
            if uav_id not in self.uav_models:
                team = uav.get('team', 'red')
                self.add_uav_model(uav_id, team)
                
            # Pozisyon güncelle
            if uav_id in self.uav_models:
                pos = np.array(uav.get('position', [0, 0, 100]))
                heading = uav.get('heading', 0)
                pitch = uav.get('pitch', 0)
                roll = uav.get('roll', 0)
                
                self.update_uav(uav_id, pos, heading, pitch, roll)
                
    def _texture_to_numpy(self) -> np.ndarray:
        """Panda3D texture'ı numpy array'e dönüştür"""
        if self.texture is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        try:
            # RAM'a kopyala
            if self.buffer and self.buffer.getGsg():
                self.engine.extractTextureData(self.texture, self.buffer.getGsg())
            
            # Numpy array olarak al
            data = self.texture.getRamImage()
            if data is None:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
            img = np.frombuffer(data, dtype=np.uint8)
            
            # BGRA format
            component_type = self.texture.getComponentType()
            num_components = self.texture.getNumComponents()
            
            if num_components == 4:
                img = img.reshape((self.height, self.width, 4))
                # BGRA -> RGB
                img = img[::-1, :, [2, 1, 0]]  # Flip Y, BGR to RGB
            elif num_components == 3:
                img = img.reshape((self.height, self.width, 3))
                img = img[::-1, :, ::-1]  # Flip Y, BGR to RGB
            else:
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
            return img.copy()
            
        except Exception as e:
            print(f"⚠️ Texture extraction failed: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
    def cleanup(self):
        """Kaynakları temizle"""
        with self._lock:
            self.uav_models.clear()
            self.air_defense_visuals.clear()
            
            if self.terrain:
                self.terrain.removeNode()
                self.terrain = None
                
            if self.base:
                self.base.destroy()
                self.base = None
                
            self._initialized = False
            
    @property
    def is_initialized(self) -> bool:
        return self._initialized
        
    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


# Fallback renderer for when Panda3D is not available
class FallbackRenderer:
    """
    Panda3D olmadan basit 2D render
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._initialized = True
        
    def initialize(self) -> bool:
        return True
        
    def setup_scene(self, terrain_config: dict = None) -> bool:
        return True
        
    def add_uav_model(self, *args, **kwargs):
        pass
        
    def render_frame(self, camera_pos, camera_orient, uav_states=None) -> np.ndarray:
        """Basit 2D render (fallback)"""
        import cv2
        
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Gökyüzü gradient
        for y in range(self.height // 2):
            intensity = int(150 + 50 * y / (self.height // 2))
            frame[y, :] = [intensity, int(intensity * 0.9), int(intensity * 0.8)]
            
        # Zemin
        frame[self.height // 2:, :] = [80, 120, 100]
        
        # Ufuk çizgisi
        cv2.line(frame, (0, self.height // 2), (self.width, self.height // 2),
                (180, 180, 180), 2)
                
        # İHA'ları basit şekilde çiz
        if uav_states:
            for uav in uav_states:
                # Basit projeksiyon
                pos = np.array(uav.get('position', [100, 100, 100]))
                rel_pos = pos - camera_pos
                
                if rel_pos[0] > 0:  # Önde mi?
                    dist = np.linalg.norm(rel_pos)
                    if dist > 10:
                        # Ekran pozisyonu
                        cx = int(self.width / 2 + rel_pos[1] * 200 / dist)
                        cy = int(self.height / 2 - rel_pos[2] * 200 / dist)
                        
                        # Boyut
                        size = max(5, int(100 / dist * 10))
                        
                        # Çiz
                        color = (0, 0, 200) if uav.get('team') == 'red' else (200, 100, 0)
                        cv2.ellipse(frame, (cx, cy), (size, size//3), 0, 0, 360, color, -1)
                        
        return frame
        
    def cleanup(self):
        pass
        
    @property
    def is_initialized(self) -> bool:
        return True
        
    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


def create_renderer(width: int = 640, height: int = 480, 
                    use_panda3d: bool = True) -> 'OffscreenRenderer':
    """
    Uygun renderer oluştur
    
    Args:
        width: Frame genişliği
        height: Frame yüksekliği
        use_panda3d: Panda3D kullanmayı dene
        
    Returns:
        OffscreenRenderer veya FallbackRenderer
    """
    if use_panda3d and PANDA3D_AVAILABLE:
        renderer = OffscreenRenderer(width, height)
        if renderer.initialize():
            return renderer
        print("⚠️ Falling back to simple renderer")
        
    return FallbackRenderer(width, height)
