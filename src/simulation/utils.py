"""
Simulation math utilities.
Centralizes common geometric calculations.
"""
import numpy as np
from typing import Tuple, Optional

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Euler angles (radians) to rotation matrix (ZYX convention).
    World to Body transformation.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # R_z * R_y * R_x
    return np.array([
        [cy*cp, sy*cp, -sp],
        [cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr],
        [cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr]
    ])

def project_point_simple(world_point: np.ndarray,
                         camera_pos: np.ndarray,
                         camera_orient: np.ndarray,
                         width: int,
                         height: int,
                         fov: float) -> Optional[Tuple[float, float]]:
    """
    Simple perspective projection (no distortion).
    Used for SimulationDetector and basic rendering.
    
    Args:
        world_point: Target point in World Frame (3,)
        camera_pos: Camera position in World Frame (3,)
        camera_orient: Camera orientation [roll, pitch, yaw]
        width: Frame width (pixels)
        height: Frame height (pixels)
        fov: Horizontal Field of View (degrees)
        
    Returns:
        (x, y) coordinates in screen space, or None if behind camera.
    """
    rel_pos = world_point - camera_pos
    
    # Camera rotation matrix
    R = euler_to_rotation_matrix(*camera_orient)
    
    # World -> Camera coordinates
    # R transforms World vector to Body vector
    cam_coords = R.T @ rel_pos
    
    # Behind camera check
    if cam_coords[0] <= 0.1:
        return None
        
    # Perspective projection
    # f = width / (2 * tan(fov/2)) assuming horizontal FOV coverage matches width
    f = width / (2 * np.tan(np.radians(fov / 2)))
    
    x = f * cam_coords[1] / cam_coords[0] + width / 2
    y = f * cam_coords[2] / cam_coords[0] + height / 2
    
    return (x, y)
