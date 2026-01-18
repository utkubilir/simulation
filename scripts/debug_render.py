"""
Kamera Rendering Debug Script

Bu script rendering pipeline'ını adım adım debug eder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
import numpy as np

def debug_rendering():
    print("=" * 60)
    print("🔍 KAMERA RENDERING DEBUG")
    print("=" * 60)
    
    # 1. Pygame + OpenGL Context
    print("\n[1/7] Pygame ve OpenGL Context oluşturuluyor...")
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Debug Renderer")
    print("    ✅ Pygame OK")
    
    # 2. GLRenderer
    print("\n[2/7] GLRenderer oluşturuluyor...")
    from src.rendering.renderer import GLRenderer
    renderer = GLRenderer(640, 480)
    print(f"    ✅ Renderer OK: {renderer.width}x{renderer.height}")
    print(f"    - FBO: {renderer.fbo}")
    print(f"    - Context: {renderer.ctx}")
    
    # 3. Environment
    print("\n[3/7] Environment oluşturuluyor...")
    from src.simulation.environment import Environment
    env = Environment()
    print(f"    ✅ Environment OK")
    print(f"    - Terrain size: {env.terrain.size}")
    print(f"    - Terrain heightmap shape: {env.terrain.heightmap.shape}")
    print(f"    - Objects count: {len(env.get_all_objects())}")
    
    # 4. Init Environment in Renderer
    print("\n[4/7] renderer.init_environment() çağrılıyor...")
    renderer.init_environment(env)
    print(f"    ✅ Environment initialized")
    print(f"    - renderer.environment: {hasattr(renderer, 'environment') and renderer.environment is not None}")
    print(f"    - renderer.vao_terrain: {hasattr(renderer, 'vao_terrain')}")
    print(f"    - renderer.terrain_vertex_count: {getattr(renderer, 'terrain_vertex_count', 'N/A')}")
    
    # 5. Arena
    print("\n[5/7] Arena oluşturuluyor...")
    from src.simulation.arena import TeknofestArena
    arena = TeknofestArena()
    renderer.init_arena(arena)
    print(f"    ✅ Arena initialized")
    print(f"    - renderer.arena: {hasattr(renderer, 'arena') and renderer.arena is not None}")
    print(f"    - Markers: {len(arena.markers)}")
    
    # 6. Kamera konumu ayarla ve render et
    print("\n[6/7] Test render başlatılıyor...")
    camera_pos = np.array([250.0, 100.0, 250.0])  # Arena merkezinde, yukarıda
    camera_rot = np.array([0.0, -0.3, 0.0])  # Hafif aşağı bakış
    
    renderer.update_camera(camera_pos, camera_rot)
    print(f"    - Camera position: {camera_pos}")
    print(f"    - Camera rotation: {camera_rot}")
    
    # Begin Frame
    print("\n    [6a] renderer.begin_frame()...")
    renderer.begin_frame()
    print("        ✅ begin_frame OK")
    
    # End Frame
    print("    [6b] renderer.end_frame()...")
    renderer.end_frame(time=0.0)
    print("        ✅ end_frame OK")
    
    # Read Pixels
    print("    [6c] renderer.read_pixels()...")
    frame = renderer.read_pixels()
    print(f"        ✅ read_pixels OK")
    
    # 7. Frame analizi
    print("\n[7/7] Frame analizi...")
    print(f"    - Shape: {frame.shape}")
    print(f"    - Dtype: {frame.dtype}")
    print(f"    - Min value: {frame.min()}")
    print(f"    - Max value: {frame.max()}")
    print(f"    - Mean value: {frame.mean():.2f}")
    print(f"    - Std dev: {frame.std():.2f}")
    
    # Renk kanalları
    if len(frame.shape) == 3 and frame.shape[2] >= 3:
        print(f"    - B channel mean: {frame[:,:,0].mean():.2f}")
        print(f"    - G channel mean: {frame[:,:,1].mean():.2f}")
        print(f"    - R channel mean: {frame[:,:,2].mean():.2f}")
    
    # Histogram
    unique_values = len(np.unique(frame))
    print(f"    - Unique pixel values: {unique_values}")
    
    # Merkez piksel
    h, w = frame.shape[:2]
    center_pixel = frame[h//2, w//2]
    print(f"    - Center pixel (BGR): {center_pixel}")
    
    # Sonuç değerlendirmesi
    print("\n" + "=" * 60)
    if frame.mean() > 100:
        print("✅ SONUÇ: Frame BAŞARIYLA render edildi!")
        print(f"   Mean {frame.mean():.1f} > 100 → Görüntü var")
    elif frame.mean() > 30:
        print("⚠️ SONUÇ: Frame kısmen render edildi")
        print(f"   Mean {frame.mean():.1f} → Karanlık ama boş değil")
    else:
        print("❌ SONUÇ: Frame BOŞ veya SİYAH")
        print(f"   Mean {frame.mean():.1f} → Rendering başarısız")
    print("=" * 60)
    
    # Frame'i kaydet
    import cv2
    debug_path = "/Users/utkubilir/Documents/GitHub/simulation/debug_frame.png"
    cv2.imwrite(debug_path, frame)
    print(f"\n📸 Debug frame kaydedildi: {debug_path}")
    
    pygame.quit()
    return frame

if __name__ == "__main__":
    debug_rendering()
