
import pytest
import time
import numpy as np
import cv2
from src.simulation.camera import FixedCamera

class TestCameraBlurPerformance:
    """Performance tests for camera motion blur."""

    @pytest.mark.performance
    def test_motion_blur_kernel_caching(self):
        """
        Test that motion blur kernel generation is efficient (cached).
        """
        # Setup camera with high blur strength to allow large kernels
        camera = FixedCamera([0, 0, 100], config={
            'motion_blur': True,
            'blur_strength': 50,
            'resolution': (640, 480)
        })

        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)

        # Parameters for the test
        # Speed 400 -> blur_size ~ 41
        velocity = np.array([300.0, 300.0, 0.0])
        camera_orient = np.array([0.0, 0.0, 0.0])

        # Warmup
        for _ in range(10):
            camera._apply_motion_blur(frame, velocity, camera_orient)

        # Benchmark
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            camera._apply_motion_blur(frame, velocity, camera_orient)

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        print(f"\n📊 Motion blur (same params): {avg_time_ms:.4f}ms avg")

        # Case 2: Alternating velocities (should hit cache if size is enough)
        velocity1 = np.array([300.0, 300.0, 0.0])
        velocity2 = np.array([300.0, -300.0, 0.0])

        start_time = time.perf_counter()
        for i in range(iterations):
            vel = velocity1 if i % 2 == 0 else velocity2
            camera._apply_motion_blur(frame, vel, camera_orient)

        end_time = time.perf_counter()
        avg_time_mixed_ms = ((end_time - start_time) / iterations) * 1000
        print(f"📊 Motion blur (mixed params): {avg_time_mixed_ms:.4f}ms avg")
