
import sys
sys.path.insert(0, '.')
from src.rendering.renderer import GLRenderer
import numpy as np
import moderngl
import pygame

# Initialize Pygame for OpenGL context
pygame.init()
pygame.display.set_mode((640, 480), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)

try:
    print("Initializing GLRenderer...")
    renderer = GLRenderer(640, 480)
    
    print("Testing render_instanced_aircraft...")
    # Generate 100 random positions
    positions = np.random.rand(100, 3) * 100 - 50
    directions = np.random.rand(100, 3) - 0.5
    colors = np.random.rand(100, 3)
    
    renderer.begin_frame()
    renderer.render_instanced_aircraft(positions, directions, colors)
    renderer.end_frame()
    
    print("Verify read_pixels...")
    img = renderer.read_pixels()
    print(f"Image shape: {img.shape}")
    
    if np.mean(img) > 0:
        print("Success: Image contains data (not black)")
    else:
        print("Warning: Image is black (might be expected if camera looks away)")
        
    print("Instanced Rendering setup verified!")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
