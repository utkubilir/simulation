import pygame
import sys
import numpy as np
from src.rendering.renderer import GLRenderer

def main():
    # Pygame başlat
    pygame.init()
    
    # OpenGL Context Ayarları (ModernGL için gerekli)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    
    # Pencere oluştur (OPENGL Bayrağı ile)
    width, height = 800, 600
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("OpenGL Hello World")
    
    # Renderer oluştur
    try:
        renderer = GLRenderer(width, height)
    except Exception as e:
        print(f"Renderer başlatılamadı: {e}")
        pygame.quit()
        sys.exit(1)
        
    clock = pygame.time.Clock()
    running = True
    
    start_time = pygame.time.get_ticks()

    while running:
        # Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Kamera Hareketi (Daire çiz)
        t = (pygame.time.get_ticks() - start_time) / 1000.0
        cam_x = 100.0 * np.cos(t * 0.5)
        cam_y = 100.0 * np.sin(t * 0.5)
        cam_z = -50.0 # Yukarıdan bakış (Simülasyon Z aşağı pozitif, ama OpenGL kamera -Z bakıyor, biraz karışık. Deneyelim.)
        
        # Simülasyon koordinatları: [x, y, z]
        # Kamera pozisyonu: [x, y, -100] (100m yukarıda)
        renderer.update_camera(
            position=[cam_x, cam_y, 100.0], 
            rotation=[0, np.radians(45), t * 0.5] # Hafif aşağı bak, dön
        )
                
        # Render Başlat (FBO Bind + Clear + Grid Draw)
        renderer.begin_frame()
        
        # Hareketli Uçaklar (İHA Temsili)
        # Merkezde kırmızı uçak (Player)
        renderer.render_aircraft([0, 0, 0], heading=0, roll=np.radians(10)*np.sin(t), pitch=0, color=(1.0, 0.0, 0.0))
        
        # Etrafta dönen mavi uçak (Target)
        target_x = 20.0 * np.cos(t)
        target_y = 20.0 * np.sin(t)
        # Heading: Harekete teğet (-sin, cos) -> atan2
        target_heading = t + np.pi/2
        renderer.render_aircraft(
            [target_x, target_y, -5.0], 
            heading=target_heading, 
            roll=np.radians(30), # Dönüşe yatış
            color=(0.0, 0.0, 1.0)
        )
        
        # Yeşil duran uçak
        renderer.render_aircraft([10, 10, 0], heading=np.radians(45), color=(0.0, 1.0, 0.0))
        
        # Render Bitir (Post-Process + Screen Draw)
        renderer.end_frame()
        
        # Buffer Swap (Ekranı güncelle)
        pygame.display.flip()
        
        clock.tick(60)
        
    pygame.quit()

if __name__ == "__main__":
    main()
