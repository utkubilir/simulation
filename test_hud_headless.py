
import pygame
from src.simulation.ui.hud import HUDOverlay

def test_hud_rendering():
    pygame.init()
    surface = pygame.Surface((1280, 720))
    hud = HUDOverlay(1280, 720)
    
    # Dummy data
    uav_state = {
        'altitude': 1500,
        'speed': 85,
        'heading': 45,
        'throttle': 0.75,
        'battery': 0.88,
        'position': [500, 500, 100],
        'id': 'player'
    }
    world_state = {
        'time': 12.5,
        'score': 150,
        'uavs': {
            'player': {'position': [500, 500, 100]},
            'target_01': {'position': [700, 600, 100]}
        }
    }
    lock_state = {
        'is_locked': True,
        'target_id': 'target_01',
        'is_valid': False,
        'progress': 0.5
    }
    
    hud.update(uav_state, world_state, lock_state)
    hud.render(surface, uav_state, world_state, lock_state)
    print("HUD rendered successfully")

if __name__ == "__main__":
    test_hud_rendering()
