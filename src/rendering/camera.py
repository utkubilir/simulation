"""
OpenGL Camera System for Simulation

Coordinate System Mapping:
--------------------------
Simulation (NED-like):     OpenGL Convention:
  +X = Forward               +X = Right  
  +Y = Right                 +Y = Up
  +Z = Down                  +Z = Toward Camera (out of screen)

Transformation Applied:
  sim_to_gl = [Y, -Z, -X] (swap and negate axes)

Euler Angle Convention (Simulation):
  - Roll: Rotation around X-axis (forward)
  - Pitch: Rotation around Y-axis (right) 
  - Yaw: Rotation around Z-axis (down)
"""

import numpy as np
import pyrr
from typing import Optional, Tuple


class GLCamera:
    """
    Enhanced OpenGL Camera System.
    
    Features:
    - Proper coordinate system transformation (NED → OpenGL)
    - Quaternion support
    - LookAt functionality
    - Frustum plane extraction
    - Dirty flag optimization
    """
    
    # Coordinate system transformation matrix (Simulation NED → OpenGL)
    # Sim: X=Forward, Y=Right, Z=Down
    # GL:  X=Right, Y=Up, Z=-Forward
    SIM_TO_GL = np.array([
        [0, 1, 0, 0],   # GL.X = Sim.Y
        [0, 0, -1, 0],  # GL.Y = -Sim.Z
        [-1, 0, 0, 0],  # GL.Z = -Sim.X
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    def __init__(self, fov: float = 60.0, aspect_ratio: float = 1.33, 
                 near: float = 0.1, far: float = 1000.0):
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far
        
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll, pitch, yaw
        self._quaternion: Optional[np.ndarray] = None
        
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.proj_matrix = np.eye(4, dtype=np.float32)
        self._vp_matrix: Optional[np.ndarray] = None  # Cached view-projection
        
        # Frustum planes for culling [left, right, bottom, top, near, far]
        self._frustum_planes: Optional[np.ndarray] = None
        
        # Dirty flags for optimization
        self._view_dirty = True
        self._proj_dirty = True
        
        self._update_projection()
        self.update()

    def set_projection(self, fov: float, aspect_ratio: float, 
                       near: float = 0.1, far: float = 1000.0):
        """Update projection parameters."""
        if (self.fov != fov or self.aspect_ratio != aspect_ratio or 
            self.near != near or self.far != far):
            self.fov = fov
            self.aspect_ratio = aspect_ratio
            self.near = near
            self.far = far
            self._proj_dirty = True
            self._update_projection()

    def _update_projection(self):
        """Recalculate projection matrix."""
        if not self._proj_dirty:
            return
        self.proj_matrix = pyrr.matrix44.create_perspective_projection_matrix(
            self.fov, self.aspect_ratio, self.near, self.far, dtype=np.float32
        )
        self._proj_dirty = False
        self._vp_matrix = None  # Invalidate VP cache

    def set_transform(self, position: np.ndarray, rotation: np.ndarray):
        """Update camera position and rotation from simulation coordinates."""
        new_pos = np.array(position, dtype=np.float32)
        new_rot = np.array(rotation, dtype=np.float32)
        
        if not (np.allclose(self.position, new_pos) and np.allclose(self.rotation, new_rot)):
            self.position = new_pos
            self.rotation = new_rot
            self._quaternion = None  # Clear quaternion
            self._view_dirty = True
            self.update()
    
    def set_quaternion(self, position: np.ndarray, quaternion: np.ndarray):
        """Set camera pose using quaternion (w, x, y, z)."""
        self.position = np.array(position, dtype=np.float32)
        self._quaternion = np.array(quaternion, dtype=np.float32)
        self._view_dirty = True
        self._update_view_from_quaternion()
    
    def look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray = None):
        """
        Set camera using look-at parameters.
        
        Args:
            eye: Camera position in world space
            target: Point to look at
            up: Up vector (default: [0, 0, -1] for NED)
        """
        if up is None:
            up = np.array([0, 0, -1], dtype=np.float32)
        
        self.position = np.array(eye, dtype=np.float32)
        
        # Calculate view matrix directly
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_corrected = np.cross(right, forward)
        
        # Build rotation matrix
        rot = np.eye(4, dtype=np.float32)
        rot[0, :3] = right
        rot[1, :3] = up_corrected
        rot[2, :3] = -forward
        
        # Translation
        trans = np.eye(4, dtype=np.float32)
        trans[:3, 3] = -eye
        
        # View = Rotation * Translation (correct order)
        self.view_matrix = rot @ trans
        
        # Apply coordinate system correction
        self.view_matrix = self.view_matrix @ self.SIM_TO_GL.T
        
        self._view_dirty = False
        self._vp_matrix = None

    def update(self):
        """Update view matrix from Euler angles."""
        if not self._view_dirty:
            return
            
        if self._quaternion is not None:
            self._update_view_from_quaternion()
            return
        
        roll, pitch, yaw = self.rotation
        
        # Build rotation matrices (negative for view matrix)
        rot_x = pyrr.matrix44.create_from_x_rotation(-roll, dtype=np.float32)
        rot_y = pyrr.matrix44.create_from_y_rotation(-pitch, dtype=np.float32)
        rot_z = pyrr.matrix44.create_from_z_rotation(-yaw, dtype=np.float32)
        
        # Rotation order: Z * Y * X (yaw-pitch-roll)
        rotation = rot_z @ rot_y @ rot_x
        
        # Translation matrix
        translation = pyrr.matrix44.create_from_translation(-self.position, dtype=np.float32)
        
        # View matrix: Rotation * Translation (correct order!)
        self.view_matrix = rotation @ translation
        
        # Apply coordinate system correction (NED → OpenGL)
        self.view_matrix = self.view_matrix @ self.SIM_TO_GL.T
        
        self._view_dirty = False
        self._vp_matrix = None
        self._frustum_planes = None
    
    def _update_view_from_quaternion(self):
        """Build view matrix from quaternion."""
        if self._quaternion is None:
            return
            
        # Quaternion to rotation matrix
        rot = pyrr.matrix44.create_from_quaternion(self._quaternion, dtype=np.float32)
        
        # Invert rotation for view matrix
        rot[:3, :3] = rot[:3, :3].T
        
        # Translation
        translation = pyrr.matrix44.create_from_translation(-self.position, dtype=np.float32)
        
        self.view_matrix = rot @ translation @ self.SIM_TO_GL.T
        
        self._view_dirty = False
        self._vp_matrix = None

    def get_view_matrix(self) -> np.ndarray:
        """Get view matrix."""
        if self._view_dirty:
            self.update()
        return self.view_matrix
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get projection matrix."""
        if self._proj_dirty:
            self._update_projection()
        return self.proj_matrix
    
    def get_view_projection_matrix(self) -> np.ndarray:
        """Get combined view-projection matrix (cached)."""
        if self._vp_matrix is None:
            self._vp_matrix = self.proj_matrix @ self.view_matrix
        return self._vp_matrix
    
    def get_frustum_planes(self) -> np.ndarray:
        """
        Extract frustum planes for culling.
        
        Returns:
            6x4 array of plane equations [A, B, C, D] where Ax + By + Cz + D = 0
        """
        if self._frustum_planes is not None:
            return self._frustum_planes
        
        vp = self.get_view_projection_matrix()
        planes = np.zeros((6, 4), dtype=np.float32)
        
        # Left plane
        planes[0] = vp[3] + vp[0]
        # Right plane
        planes[1] = vp[3] - vp[0]
        # Bottom plane
        planes[2] = vp[3] + vp[1]
        # Top plane
        planes[3] = vp[3] - vp[1]
        # Near plane
        planes[4] = vp[3] + vp[2]
        # Far plane
        planes[5] = vp[3] - vp[2]
        
        # Normalize planes
        for i in range(6):
            norm = np.linalg.norm(planes[i, :3])
            if norm > 0:
                planes[i] /= norm
        
        self._frustum_planes = planes
        return planes
    
    def is_point_in_frustum(self, point: np.ndarray) -> bool:
        """Check if a point is inside the view frustum."""
        planes = self.get_frustum_planes()
        for plane in planes:
            if np.dot(plane[:3], point) + plane[3] < 0:
                return False
        return True
    
    def is_sphere_in_frustum(self, center: np.ndarray, radius: float) -> bool:
        """Check if a sphere intersects the view frustum."""
        planes = self.get_frustum_planes()
        for plane in planes:
            dist = np.dot(plane[:3], center) + plane[3]
            if dist < -radius:
                return False
        return True
    
    def get_forward_vector(self) -> np.ndarray:
        """Get camera forward direction in world space."""
        # Inverse of view matrix rotation gives world-space orientation
        inv_rot = self.view_matrix[:3, :3].T
        return -inv_rot[2]  # -Z in camera space is forward
    
    def get_right_vector(self) -> np.ndarray:
        """Get camera right direction in world space."""
        inv_rot = self.view_matrix[:3, :3].T
        return inv_rot[0]
    
    def get_up_vector(self) -> np.ndarray:
        """Get camera up direction in world space."""
        inv_rot = self.view_matrix[:3, :3].T
        return inv_rot[1]
