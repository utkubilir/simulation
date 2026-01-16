import pygame
import numpy as np
import time
from src.simulation.camera import FixedCamera
import cv2

def main():
    # 1. Pygame ve OpenGL Context Başlat
    pygame.init()
    
    # macOS için OpenGL 3.3 Core Profile zorunlu
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    
    resolution = (800, 600)
    
    # OpenGL Render Context Penceresi (Görünmez olabilir veya debug için görünür)
    # Simülasyon normalde headless çalışabilir ama OpenGL için context şart.
    screen = pygame.display.set_mode(resolution, pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Simulation Integration Test (OpenGL)")
    
    clock = pygame.time.Clock()
    
    # 2. Kamera Kurulumu (Context hazır olduktan sonra)
    camera_pos = [0, 0, -100] # 100m yukarıda
    config = {
        'resolution': resolution,
        'fps': 60,
        'fov': 60,
        # Efektleri aç
        'distortion_enabled': True,
        'dof_enabled': True,
        'fog_enabled': True
    }
    
    print("Kamera başlatılıyor...")
    camera = FixedCamera(position=camera_pos, config=config)
    
    if camera.renderer:
        print("✅ OpenGL Renderer aktif!")
    else:
        print("❌ OpenGL Renderer başlatılamadı, CPU moduna geçildi.")
        
    # Test Verisi (UAVs)
    targets = [
        {'id': 'uav1', 'position': [0, 0, -110], 'heading': 0, 'is_player': True, 'size': 5.0}, # Player
        {'id': 'uav2', 'position': [20, 0, -110], 'heading': 45, 'is_player': False, 'size': 5.0}, # Target
    ]
    
    running = True
    start_time = time.time()
    
    # Debug için ayrı bir pencere (OpenCV)
    # Pygame penceresi OpenGL buffer olduğu için oraya 2D resim çizmek zor (Texture olarak çizmek lazım).
    # Biz read_pixels sonucunu OpenCV penceresinde görelim.
    
    # Debug için Frame Counter
    frame_count = 0
    max_frames = 100 # 100 frame sonra otomatik dur
    fps_list = []
    
    print("Test döngüsü başlıyor...")
    
    while running and frame_count < max_frames:
        loop_start = time.time()
        dt = clock.tick(60) / 1000.0
        t = time.time() - start_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Simüle edilmiş hareket
        targets[1]['position'] = [30 * np.cos(t), 30 * np.sin(t), -110 + 10*np.sin(t*0.5)]
        targets[1]['heading'] = np.degrees(t)
        targets[1]['roll'] = 15 * np.sin(t)
        
        # Kamera bakış açısı (Aşağı bakıyor)
        camera_orient = np.array([0.0, np.radians(90), 0.0])
        
        # 3. Frame Oluştur
        try:
            render_start = time.time()
            frame_bgr = camera.generate_synthetic_frame(
                uav_states=targets,
                camera_pos=np.array(camera_pos),
                camera_orient=camera_orient,
                own_velocity=np.array([0,0,0])
            )
            render_time = time.time() - render_start
            current_fps = 1.0 / render_time if render_time > 0 else 60
            fps_list.append(current_fps)
            
        except Exception as e:
            print(f"Render hatası: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # 4. Sonucu Kaydet (İlk ve son frame)
        if frame_count == 0 or frame_count == max_frames - 1:
            cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame_bgr)
            print(f"Frame {frame_count} kaydedildi. Render Süresi: {render_time*1000:.2f}ms ({current_fps:.1f} FPS)")
        
        # Ekrana basmıyoruz (Headless debug için) veya sadece logluyoruz
        # cv2.imshow("Simulator", frame_bgr)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
        pygame.display.flip()
        frame_count += 1
        
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Test Tamamlandı. Ortalama FPS: {avg_fps:.2f}")
            
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
