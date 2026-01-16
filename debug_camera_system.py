import unittest
import numpy as np
import pygame
import cv2
import time
from src.simulation.camera import FixedCamera
from src.simulation.utils import euler_to_rotation_matrix

class TestCameraSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Headless mode for Pygame
        pygame.init()
        # OpenGL Attribute Setup (macOS Core Profile)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        
        # Create hidden window
        cls.resolution = (640, 480)
        pygame.display.set_mode(cls.resolution, pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
        
    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        
    def setUp(self):
        self.camera_pos = [0, 0, -100]
        self.config = {
            'resolution': self.resolution,
            'fps': 30,
            'fov': 60,
            'distortion_enabled': True
        }
        self.camera = FixedCamera(position=self.camera_pos, config=self.config)
        
    def test_renderer_initialization(self):
        """Test if OpenGL renderer is initialized correctly"""
        print("\n--- Testing Renderer Initialization ---")
        self.assertIsNotNone(self.camera.renderer, "OpenGL Renderer should be initialized")
        print("✅ Renderer initialized")
        
    def test_intrinsic_matrix(self):
        """Test Camera Matrix (K) calculation"""
        print("\n--- Testing Intrinsic Matrix ---")
        K = self.camera.K
        self.assertEqual(K.shape, (3, 3))
        w, h = self.resolution
        # Principal point should be center
        self.assertAlmostEqual(K[0, 2], w/2, delta=1.0)
        self.assertAlmostEqual(K[1, 2], h/2, delta=1.0)
        print(f"✅ Intrinsic Matrix verified:\n{K}")
        
    def test_projection(self):
        """Test 3D projection to 2D"""
        print("\n--- Testing Projection ---")
        # Point directly in front of camera
        # Camera at [0, 0, -100], looking down (Pitch=90) -> Z axis aligns with World Z
        # Simulation coords: Z down. Camera looks 'down' (+Z).
        # Let's orient camera looking North (X+) first for simplicity
        
        # Camera at Origin, looking +X
        cam = FixedCamera(position=[0, 0, 0], config=self.config)
        cam_orient = np.array([0, 0, 0]) # Roll, Pitch, Yaw = 0 -> Looking +X
        
        # Point at X=100, Y=0, Z=0 (Center of FOV)
        pt_center = np.array([100, 0, 0])
        proj = cam.project_point(pt_center, np.array([0,0,0]), cam_orient)
        
        self.assertIsNotNone(proj)
        cx, cy = self.resolution[0]/2, self.resolution[1]/2
        
        # Should be at center (allow small distortion error)
        print(f"Projected Center: {proj}, Expected: ({cx}, {cy})")
        self.assertAlmostEqual(proj[0], cx, delta=5.0)
        self.assertAlmostEqual(proj[1], cy, delta=5.0)
        print("✅ Center Projection verified")
        
    def test_gpu_render_pipeline(self):
        """Test full GPU render pipeline"""
        print("\n--- Testing GPU Render Pipeline ---")
        targets = [
            {'id': 't1', 'position': [100, 0, 0], 'heading': 0, 'size': 5.0}
        ]
        
        start_t = time.time()
        # Camera Orient: Looking +X
        frame = self.camera.generate_synthetic_frame(
            uav_states=targets,
            camera_pos=np.array([0, 0, 0]),
            camera_orient=np.array([0, 0, 0]),
            own_velocity=np.array([0, 0, 0])
        )
        duration = time.time() - start_t
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (self.resolution[1], self.resolution[0], 3))
        self.assertEqual(frame.dtype, np.uint8)
        
        mean_val = np.mean(frame)
        print(f"Render Time: {duration*1000:.2f} ms")
        print(f"Mean Brightness: {mean_val:.2f}")
        
        # Expect some brightness (sky or object), not pure black
        # Sky color in fragment shader is (0.5, 0.7, 0.9) approx 180 brightness
        self.assertGreater(mean_val, 10, "Render output should not be black")
        print("✅ Frame rendered and validated")
        
        # Save for manual inspection
        cv2.imwrite("debug_test_output.jpg", frame)

if __name__ == '__main__':
    unittest.main()
