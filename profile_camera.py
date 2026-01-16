"""
Camera Module Performance Profiler

Measures render times, identifies bottlenecks, and compares CPU vs GPU paths.
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

def profile_fixed_camera():
    """Profile FixedCamera generate_synthetic_frame method"""
    print("=" * 60)
    print("Camera Performance Profiler")
    print("=" * 60)
    
    # Import camera
    from src.simulation.camera import FixedCamera
    
    # Test configurations
    resolutions = [
        (320, 240),
        (640, 480),
        (1280, 720),
    ]
    
    results = []
    
    for res in resolutions:
        print(f"\n--- Resolution: {res[0]}x{res[1]} ---")
        
        config = {
            'resolution': res,
            'fov': 60,
            'distortion_enabled': True,
            'shake_enabled': True,
            'motion_blur_enabled': True,
            'sensor_noise_enabled': True,
            'vignette_enabled': True,
            'haze_enabled': True,
        }
        
        camera = FixedCamera(position=[0, 0, -100], config=config)
        
        # Test data
        uav_states = [
            {'id': 'uav1', 'position': [0, 0, -110], 'heading': 0, 'is_player': True, 'size': 5.0},
            {'id': 'uav2', 'position': [50, 0, -110], 'heading': 45, 'is_player': False, 'size': 5.0},
            {'id': 'uav3', 'position': [-30, 20, -120], 'heading': 90, 'is_player': False, 'size': 5.0},
        ]
        
        camera_pos = np.array([0, 0, -100])
        camera_orient = np.array([0.0, np.radians(90), 0.0])
        
        # Warmup
        for _ in range(5):
            camera.generate_synthetic_frame(uav_states, camera_pos, camera_orient)
        
        # Profile
        num_frames = 50
        times = []
        
        for i in range(num_frames):
            start = time.perf_counter()
            frame = camera.generate_synthetic_frame(uav_states, camera_pos, camera_orient)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        times = np.array(times)
        avg_ms = times.mean() * 1000
        std_ms = times.std() * 1000
        min_ms = times.min() * 1000
        max_ms = times.max() * 1000
        fps = 1.0 / times.mean()
        
        render_mode = "GPU (OpenGL)" if camera.renderer else "CPU"
        
        print(f"  Render Mode: {render_mode}")
        print(f"  Avg Time: {avg_ms:.2f} ms (±{std_ms:.2f})")
        print(f"  Min/Max: {min_ms:.2f} / {max_ms:.2f} ms")
        print(f"  FPS: {fps:.1f}")
        
        results.append({
            'resolution': res,
            'mode': render_mode,
            'avg_ms': avg_ms,
            'std_ms': std_ms,
            'fps': fps,
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Resolution':<15} {'Mode':<12} {'Avg (ms)':<12} {'FPS':<10}")
    print("-" * 50)
    for r in results:
        res_str = f"{r['resolution'][0]}x{r['resolution'][1]}"
        print(f"{res_str:<15} {r['mode']:<12} {r['avg_ms']:<12.2f} {r['fps']:<10.1f}")
    
    return results

def profile_individual_effects():
    """Profile individual camera effects to find bottlenecks"""
    print("\n" + "=" * 60)
    print("Individual Effect Profiling")
    print("=" * 60)
    
    from src.simulation.camera import FixedCamera
    import cv2
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Base config (all effects off)
    base_config = {
        'resolution': (640, 480),
        'fov': 60,
        'distortion_enabled': False,
        'shake_enabled': False,
        'motion_blur_enabled': False,
        'chromatic_aberration_enabled': False,
        'sensor_noise_enabled': False,
        'vignette_enabled': False,
        'haze_enabled': False,
        'dof_enabled': False,
        'lens_flare_enabled': False,
        'tonemapping_enabled': False,
        'color_grading_enabled': False,
        'softness_enabled': False,
    }
    
    camera = FixedCamera(position=[0, 0, -100], config=base_config)
    
    effects = [
        ('_apply_sensor_noise', lambda: camera._apply_sensor_noise(test_frame.copy())),
        ('_apply_vignette', lambda: camera._apply_vignette(test_frame.copy())),
        ('_apply_motion_blur', lambda: camera._apply_motion_blur(test_frame.copy(), np.array([10, 0, 0]), np.array([0, 0, 0]))),
        ('_apply_chromatic_aberration', lambda: camera._apply_chromatic_aberration(test_frame.copy())),
        ('_apply_tonemapping', lambda: camera._apply_tonemapping(test_frame.copy())),
        ('_apply_color_grading', lambda: camera._apply_color_grading(test_frame.copy())),
        ('_apply_lens_softness', lambda: camera._apply_lens_softness(test_frame.copy())),
    ]
    
    print(f"\n{'Effect':<30} {'Time (ms)':<12} {'% of 16ms':<12}")
    print("-" * 55)
    
    for name, func in effects:
        # Enable the effect temporarily
        times = []
        for _ in range(100):
            start = time.perf_counter()
            try:
                func()
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                break
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        if times:
            avg_ms = np.mean(times) * 1000
            pct_budget = (avg_ms / 16.67) * 100  # 60 FPS target
            print(f"{name:<30} {avg_ms:<12.3f} {pct_budget:<12.1f}%")

if __name__ == "__main__":
    try:
        profile_fixed_camera()
        profile_individual_effects()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
