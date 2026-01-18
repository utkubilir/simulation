"""
Camera Renderer State Debug

Bu script explore.py'daki kamera durumunu debug eder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
import numpy as np

def debug_camera_state():
    print("=" * 60)
    print("🔍 KAMERA RENDERER STATE DEBUG")
    print("=" * 60)
    
    # Önce UI Renderer gibi Pygame window oluştur (OpenGL olmadan)
    print("\n[1/5] Pygame window oluşturuluyor (OpenGL OLMADAN - UI tarzı)...")
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))  # OpenGL flag YOK!
    pygame.display.set_caption("Debug - Non-OpenGL Window")
    print(f"    ✅ Pygame screen: {screen.get_size()}")
    
    # Şimdi FixedCamera oluştur
    print("\n[2/5] FixedCamera oluşturuluyor...")
    from src.simulation.camera import FixedCamera
    camera = FixedCamera([0, 0, -100], {
        'resolution': (640, 480),
        'fov': 60
    })
    print(f"    ✅ Camera created")
    print(f"    - camera.renderer: {camera.renderer}")
    print(f"    - camera._environment: {camera._environment}")
    print(f"    - camera._environment_initialized: {camera._environment_initialized}")
    
    if camera.renderer:
        print(f"    - camera.renderer.ctx: {camera.renderer.ctx}")
        print(f"    - camera.renderer hasattr environment: {hasattr(camera.renderer, 'environment')}")
        if hasattr(camera.renderer, 'environment'):
            print(f"    - camera.renderer.environment: {camera.renderer.environment}")
    
    # Environment oluştur ve set et
    print("\n[3/5] Environment oluşturup set_environment çağrılıyor...")
    from src.simulation.environment import Environment
    env = Environment()
    camera.set_environment(env)
    
    print(f"    ✅ set_environment called")
    print(f"    - camera._environment_initialized: {camera._environment_initialized}")
    if camera.renderer:
        print(f"    - camera.renderer hasattr environment: {hasattr(camera.renderer, 'environment')}")
        if hasattr(camera.renderer, 'environment'):
            print(f"    - camera.renderer.environment is None: {camera.renderer.environment is None}")
    
    # Arena set et
    print("\n[4/5] Arena set ediliyor...")
    from src.simulation.arena import TeknofestArena
    arena = TeknofestArena()
    camera.set_arena(arena)
    
    print(f"    ✅ set_arena called")
    if camera.renderer:
        print(f"    - camera.renderer hasattr arena: {hasattr(camera.renderer, 'arena')}")
    
    # Test frame oluştur
    print("\n[5/5] Test frame oluşturuluyor...")
    uav_states = [
        {'id': 'test', 'position': [100, 100, -50], 'heading': 45, 'roll': 0, 'pitch': 0, 'is_player': True}
    ]
    camera_pos = np.array([250.0, 100.0, 250.0])
    camera_orient = np.array([0.0, 0.0, 0.0])
    
    try:
        frame = camera.generate_synthetic_frame(uav_states, camera_pos, camera_orient)
        print(f"    ✅ Frame generated")
        print(f"    - Shape: {frame.shape}")
        print(f"    - Mean: {frame.mean():.2f}")
        print(f"    - Min/Max: {frame.min()}/{frame.max()}")
        
        # Frame kaydet
        import cv2
        cv2.imwrite("/Users/utkubilir/Documents/GitHub/simulation/debug_camera_frame.png", frame)
        print(f"    📸 Saved to debug_camera_frame.png")
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    pygame.quit()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    debug_camera_state()
