"""
OpenGL Camera System for Simulation

Coordinate System Mapping:
--------------------------
Simulation (NED-like):     OpenGL Convention:
  +X = Forward               +X = Right  
  +Y = Right                 +Y = Up
  +Z = Down                  +Z = Toward Camera (out of screen)

Transformation Notes:
- Simulation uses NED-like coordinates where +Z points down
- OpenGL uses a right-handed coordinate system where -Z is forward
- The view matrix transforms world coordinates to camera space
- Shader code may need to account for coordinate flipping

Euler Angle Convention (Simulation):
  - Roll: Rotation around X-axis (forward)
  - Pitch: Rotation around Y-axis (right) 
  - Yaw: Rotation around Z-axis (down)
"""

import numpy as np
import pyrr

class GLCamera:
    """
    OpenGL Kamera Sistemi.
    View ve Projection matrislerini hesaplar ve yönetir.
    
    Handles coordinate transformation between simulation space and OpenGL camera space.
    See module docstring for coordinate system details.
    """
    def __init__(self, fov=60.0, aspect_ratio=1.33, near=0.1, far=1000.0):
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far
        
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Orientation: [roll, pitch, yaw] in radians
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.proj_matrix = np.eye(4, dtype=np.float32)
        
        self._update_projection()
        self.update()

    def set_projection(self, fov: float, aspect_ratio: float, near: float = 0.1, far: float = 1000.0):
        """Projeksiyon parametrelerini güncelle"""
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far
        self._update_projection()

    def _update_projection(self):
        """Projection matrisini yeniden hesapla"""
        self.proj_matrix = pyrr.matrix44.create_perspective_projection_matrix(
            self.fov, self.aspect_ratio, self.near, self.far, dtype=np.float32
        )

    def set_transform(self, position: np.ndarray, rotation: np.ndarray):
        """Kamera pozisyon ve rotasyonunu güncelle (Simulation koordinatlarından)"""
        # Simülasyon: X=İleri, Y=Sağ, Z=Aşağı (NED benzeri ama Z pozitif aşağı)
        # OpenGL: X=Sağ, Y=Yukarı, Z=-İleri (Kameraya doğru)
        
        # Simülasyon koordinatlarını OpenGL koordinatlarına çevirmemiz gerekebilir
        # Ancak genelde dünya matrisini sabit tutup kamerayı taşımak daha kolaydır.
        # Burada basitçe pozisyonu alıyoruz, shader'da koordinat dönüşümü yapacağız.
        
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32) # r, p, y
        self.update()

    def update(self):
        """View matrisini güncelle (Euler açıları -> LookAt veya Matrix)"""
        # Roll, Pitch, Yaw -> Rotasyon Matrisi
        # pyrr.matrix44.create_from_eulers(eulers) -> eulers order: [roll, pitch, yaw] ?
        # Not: pyrr euler sırası bazen karışıktır, basit rotasyonlarla yapalım.
        
        # Simülasyon: r/p/y -> Camera Body Frame
        
        # Kamera Pozisyonu (Translation)
        translation = pyrr.matrix44.create_from_translation(-self.position, dtype=np.float32)
        
        # Rotasyon (Ters rotasyon çünkü dünya dönüyor, kamera değil)
        # Simülasyon Yaw (Z axis rotation), Pitch (Y), Roll (X)
        roll, pitch, yaw = self.rotation
        
        rot_x = pyrr.matrix44.create_from_x_rotation(-roll, dtype=np.float32)
        rot_y = pyrr.matrix44.create_from_y_rotation(-pitch, dtype=np.float32)
        rot_z = pyrr.matrix44.create_from_z_rotation(-yaw, dtype=np.float32)
        
        # Dönüşüm sırası önemli: Translate * Rotate
        rotation = pyrr.matrix44.multiply(rot_z, pyrr.matrix44.multiply(rot_y, rot_x))
        self.view_matrix = pyrr.matrix44.multiply(translation, rotation)
        
        # Koordinat sistemi düzeltmesi (OpenGL Convention)
        # OpenGL: -Z ileri, +Y yukarı. Sim: +X ileri, +Z aşağı (muhtemelen NED)
        # Bir düzeltme matrisi gerekebilir, şimdilik basit bırakalım.

    def get_view_matrix(self):
        return self.view_matrix
        
    def get_projection_matrix(self):
        return self.proj_matrix
