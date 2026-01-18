"""
Render Pipeline Debug - Terrain ve Arena render ediliyor mu?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
import numpy as np

def debug_render_pipeline():
    print("=" * 60)
    print("🔍 RENDER PIPELINE DEBUG")
    print("=" * 60)
    
    # OpenGL Context oluştur
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    
    from src.rendering.renderer import GLRenderer
    from src.simulation.environment import Environment
    from src.simulation.arena import TeknofestArena
    
    renderer = GLRenderer(640, 480)
    env = Environment()
    arena = TeknofestArena()
    
    renderer.init_environment(env)
    renderer.init_arena(arena)
    
    print(f"✅ Renderer initialized")
    print(f"   - hasattr(renderer, 'environment'): {hasattr(renderer, 'environment')}")
    print(f"   - hasattr(renderer, 'arena'): {hasattr(renderer, 'arena')}")
    print(f"   - hasattr(renderer, 'vao_terrain'): {hasattr(renderer, 'vao_terrain')}")
    
    # Kamera pozisyonu - aşağı bakış
    camera_pos = np.array([0.0, 100.0, 0.0])  # Y=100 yukarıda, merkez
    camera_rot = np.array([0.0, 0.5, 0.0])   # pitch=0.5 rad aşağı
    
    print(f"\n📍 Camera Position: {camera_pos}")
    print(f"   Camera Rotation: {camera_rot}")
    
    renderer.update_camera(camera_pos, camera_rot)
    
    # Render
    print("\n🎬 Rendering...")
    renderer.begin_frame()
    renderer.end_frame(time=0.0)
    
    frame = renderer.read_pixels()
    
    print(f"\n📊 Frame Analysis:")
    print(f"   Shape: {frame.shape}")
    print(f"   Mean: {frame.mean():.2f}")
    print(f"   Center pixel: {frame[240, 320]}")
    print(f"   Top pixel: {frame[10, 320]}")
    print(f"   Bottom pixel: {frame[470, 320]}")
    
    # Eğer sadece sky render ediliyorsa tüm pikseller benzer olacak
    unique = len(np.unique(frame))
    print(f"   Unique values: {unique}")
    
    if frame.mean() > 200:
        print("\n⚠️  Frame çok açık - muhtemelen sadece sky render ediliyor")
    elif frame.mean() < 50:
        print("\n⚠️  Frame çok koyu - rendering başarısız")
    else:
        print("\n✅ Frame düzgün görünüyor")
    
    # Frame kaydet
    import cv2
    cv2.imwrite("/Users/utkubilir/Documents/GitHub/simulation/debug_pipeline_frame.png", frame)
    print(f"\n📸 Saved: debug_pipeline_frame.png")
    
    pygame.quit()

if __name__ == "__main__":
    debug_render_pipeline()
