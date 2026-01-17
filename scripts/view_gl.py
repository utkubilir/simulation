"""
Pure OpenGL Viewer - Teknofest Savaşan İHA Arena
Doğrudan GLRenderer kullanarak arena'yı render eder.
"""

import sys
import os
from pathlib import Path
import pygame
import numpy as np
import math

# Add project root path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rendering.renderer import GLRenderer
from src.simulation.world import SimulationWorld
from src.simulation.arena import TeknofestArena

def main():
    pygame.init()
    width, height = 800, 600
    
    # Create standard Pygame window (Software mode)
    screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
    pygame.display.set_caption("Teknofest Savaşan İHA Arena Viewer")
    
    # Initialize Renderer with increased far plane for large arena (500m)
    renderer = GLRenderer(width, height)
    renderer.camera.set_projection(fov=60.0, aspect_ratio=width/height, near=0.1, far=2000.0)
    
    # Create Teknofest Arena (500m x 500m)
    arena = TeknofestArena({
        'width': 500.0,
        'depth': 500.0,
        'min_altitude': 10.0,
        'max_altitude': 150.0,
        'safe_zone_size': 50.0
    })
    
    print(f"[INFO] Arena created: {arena.width}m x {arena.depth}m")
    print(f"[INFO] Safe zones: {len(arena.safe_zones)}")
    print(f"[INFO] Boundary markers: {len(arena.markers)}")
    
    # Setup Environment (for terrain)
    world = SimulationWorld()
    environment = world.environment
    renderer.init_environment(environment)
    
    # Initialize arena for rendering
    renderer.init_arena(arena)
    
    # Camera State - Start above arena center
    pos = np.array([0.0, 50.0, -200.0], dtype=np.float32)  # Above arena, looking forward
    rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    clock = pygame.time.Clock()
    running = True
    
    move_speed = 5.0
    rot_speed = 0.03
    
    print("\n" + "="*40)
    print("🎥 GL VIEWER STARTED")
    print("="*40)
    print("Controls:")
    print("   WASD       : Move (Forward/Left/Back/Right)")
    print("   Q / E      : Up / Down (Vertical)")
    print("   ARROWS     : Look Around")
    print("   ESC        : Exit")
    print("="*40 + "\n")
    
    frame_counter = 0
    
    while running:
        dt = clock.tick(60) / 1000.0
        frame_counter += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                
        keys = pygame.key.get_pressed()
        
        # AUTO-FLY: Camera moves forward automatically over terrain
        fly_speed = 100.0  # Units per second
        fly_direction = np.array([0, 0, 1], dtype=np.float32)  # Forward (+Z)
        pos += fly_direction * fly_speed * dt
        
        # Keep camera at constant height above ground
        pos[1] = 30.0  # Fixed height
        
        # Wrap camera position to stay within arena bounds (-250 to +250)
        if pos[2] > 200:
            pos[2] = -200
        
        # Update Renderer Camera - Manual LookAt (bypassing SIM_TO_GL)
        # Standard OpenGL lookAt: Y is up, -Z is forward
        eye = pos.copy()
        # Look forward with slight downward tilt to see terrain
        target = pos + np.array([0, -0.3, 1], dtype=np.float32)  # Look forward, slight down
        
        # Manually build view matrix without SIM_TO_GL
        forward = target - eye
        forward = forward / np.linalg.norm(forward)  # [0, -1, 0]
        
        # Up vector must be perpendicular to forward
        # If looking straight down (-Y), up should be along X or Z axis
        up_vec = np.array([0, 0, 1], dtype=np.float32)  # Z+ as reference up
        
        right = np.cross(forward, up_vec)  # cross([0,-1,0], [0,0,1]) = [-1, 0, 0]
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            right = np.array([1, 0, 0], dtype=np.float32)
        else:
            right = right / norm
        
        up_corrected = np.cross(right, forward)  # cross([-1,0,0], [0,-1,0]) = [0, 0, 1]
        
        # Build view matrix directly
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up_corrected
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(up_corrected, eye)
        view[2, 3] = -np.dot(-forward, eye)
        
        # Directly assign to camera (bypass SIM_TO_GL)
        # NOTE: Transpose for column-major OpenGL format
        renderer.camera.view_matrix = view.T
        renderer.camera.position = eye
        renderer.camera._view_dirty = False
        renderer.camera._vp_matrix = None
        
        # --- RENDER PASS ---
        
        # 1. Begin Frame (Clears Screen & Shadow Map bind)
        # NOTE: GLRenderer.begin_frame clears to Sky Blue now!
        renderer.begin_frame()

        # 2. Render Environment (Terrain + Objects)
        renderer.render_environment()
        
        # 3. Render Arena (Poles + Safe Zones)
        renderer.render_arena()
        
        # 3. Read Pixels from FBO and Blit to Screen
        # GLRenderer renders to 'fbo' (rendering step). 'read_pixels' reads from 'fbo_post' (post-process step)
        # Since we are skipping post-processing here, we must read directly from 'fbo'.
        buffer = renderer.fbo.read(components=3, alignment=1)
        frame_data = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))
        
        # Flip Y (OpenGL origin is bottom-left)
        frame_data = np.flipud(frame_data)
        
        if frame_counter % 60 == 0:
            avg_color = np.mean(frame_data)
            print(f"Frame {frame_counter}: Mean Color Value = {avg_color:.2f}")
        
        # Convert to Pygame Surface (Swap X/Y for pygame)
        # Note: read_pixels returns RGB. Pygame expects RGB usually.
        frame_surf = pygame.surfarray.make_surface(frame_data.swapaxes(0, 1))
        
        # Scale if necessary (GLRenderer size vs Window size)
        if (width, height) != (renderer.width, renderer.height):
             frame_surf = pygame.transform.scale(frame_surf, (width, height))
             
        screen.blit(frame_surf, (0, 0))
        
        # 4. Swap Buffers (Show to screen)
        pygame.display.flip()
        
    print("Exiting...")

if __name__ == "__main__":
    main()
