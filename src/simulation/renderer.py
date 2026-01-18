"""
Main Renderer - Orchestrates 2D Map and Camera Inset

Composes Renderer2D and CameraInsetRenderer into a single display.
"""

from .renderer_2d import Renderer2D
from .camera_inset import CameraInsetRenderer
from .hud import HUD
from .ui.ui_mode import UIMode
import pygame
import numpy as np

class Renderer:
    def __init__(self, width=1280, height=720, radar_heading_mode: str = "heading_up"):
        self.ui_mode = UIMode.COMPETITION
        self.width = width
        self.height = height
        self.map_renderer = Renderer2D(width, height, radar_heading_mode=radar_heading_mode)
        
        # Inset size: 320x240 (standard QVGA)
        self.inset_width = 320
        self.inset_height = 240
        self.inset_renderer = CameraInsetRenderer(self.inset_width, self.inset_height)
        
        # Cockpit HUD
        self.hud = HUD(width, height)
        self.show_cockpit = False
        self.show_gl_world = False
        
        self.clock = None
        
    def init(self):
        """Initialize renderers"""
        self.map_renderer.init()
        self.clock = pygame.time.Clock()
        self.hud.init()
        
        # Initialize inset with position bottom-right
        margin = 20
        inset_x = self.width - self.inset_width - margin
        inset_y = self.height - self.inset_height - margin
        
        # Map renderer creates the display, so we pass that surface
        self.inset_renderer.init(self.map_renderer.screen, inset_x, inset_y)
        
    def close(self):
        """Cleanup"""
        self.map_renderer.close()
        
    def handle_events(self):
        """Pass events to sub-renderers"""
        events = self.map_renderer.handle_events()
        
        # Check for toggle and observer commands
        for e in events:
            if e['type'] == 'keydown':
                if e['key'] == 'tab':
                    self.show_cockpit = not self.show_cockpit
                elif e['key'] == 'g':
                    self.show_gl_world = not self.show_gl_world
                elif e['key'] == '[':
                    e['cmd'] = 'prev_uav'
                elif e['key'] == ']':
                    e['cmd'] = 'next_uav'
                elif e['key'] == 'f3' or e['key'] == 'm':
                    # Cycle modes: COMPETITION -> DEBUG -> TRAINING
                    if self.ui_mode == UIMode.COMPETITION:
                        self.ui_mode = UIMode.DEBUG
                    elif self.ui_mode == UIMode.DEBUG:
                        self.ui_mode = UIMode.TRAINING
                    else:
                        self.ui_mode = UIMode.COMPETITION
                    print(f"UI Mode: {self.ui_mode.name}")
                
        return events
        
    def set_camera(self, x, y):
        """Update map camera"""
        self.map_renderer.set_camera(x, y)
        
    def render(self, world_state, lock_state, sim_time=0.0, scenario="", seed=0,
               camera_frame=None, detections=None, tracks=None, observer_target_id=None,
               is_paused=False, gl_frame=None, inset_frame=None):

        """
        Render the complete UI.
        """
        screen = self.map_renderer.screen
        
        # Info Text
        target_text = f"Watching: {observer_target_id}" if observer_target_id else "Watching: Player"
        
        if self.show_cockpit:
            # === COCKPIT MODE ===
            
            # 1. Fullscreen Camera Background
            if camera_frame is not None:
                frame_surf = pygame.surfarray.make_surface(camera_frame.swapaxes(0, 1))
                frame_surf = pygame.transform.scale(frame_surf, (self.width, self.height))
                screen.blit(frame_surf, (0, 0))
            else:
                screen.fill((0, 0, 0))
                
            # 2. HUD Overlay
            # Determine HUD source uav
            hud_uav = None
            if observer_target_id:
                hud_uav = world_state.get('uavs', {}).get(observer_target_id)
            else:
                hud_uav = next((u for u in world_state.get('uavs', {}).values() if u.get('is_player')), None)
                
            if hud_uav:
                self.hud.render(screen, hud_uav, detections, lock_state, self.ui_mode, world_state=world_state)
            
            # 3. Simple Status Text
            font = self.map_renderer.font_small
            fps = self.clock.get_fps()
            txt = font.render(f"FPS: {fps:.1f} | COCKPIT | {target_text} | {self.ui_mode.name}", True, (200, 255, 200))
            screen.blit(txt, (10, 10))
            
        elif self.show_gl_world and gl_frame is not None:
            # === 3D WORLD MODE ===
            frame_surf = pygame.surfarray.make_surface(gl_frame.swapaxes(0, 1))
            frame_surf = pygame.transform.scale(frame_surf, (self.width, self.height))
            screen.blit(frame_surf, (0, 0))

            font = self.map_renderer.font_small
            fps = self.clock.get_fps()
            txt = font.render(
                f"FPS: {fps:.1f} | 3D WORLD | {target_text} | {self.ui_mode.name}",
                True,
                (200, 255, 200),
            )
            screen.blit(txt, (10, 10))
            help_txt = font.render(
                "TAB: Cockpit View | G: 2D Map | SPACE: Pause | R: Restart",
                True,
                (220, 220, 220),
            )
            screen.blit(help_txt, (10, 30))
        else:
            # === MAP MODE ===
            
            # 1. Render Map (Background)
            self.map_renderer.render(world_state, lock_state, sim_time, scenario, seed, self.ui_mode)
            
            # 2. Render Inset (Overlay)
            if inset_frame is None:
                # Use standard camera resolution for placeholder
                inset_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
            self.inset_renderer.render(inset_frame, detections or [], tracks or [], lock_state, self.ui_mode)
            
            # Help text
            font = self.map_renderer.font_small
            txt = font.render(
                f"TAB: Cockpit View | G: 3D World | SPACE: Pause | R: Restart | {target_text}",
                True,
                (200, 200, 200),
            )
            screen.blit(txt, (self.width - 450, 10))
            
        # Draw Pause Overlay
        if is_paused:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128)) # Semi-transparent black
            screen.blit(overlay, (0, 0))
            
            font_large = pygame.font.Font(None, 72)
            pause_txt = font_large.render("PAUSED", True, (255, 255, 255))
            txt_rect = pause_txt.get_rect(center=(self.width/2, self.height/2))
            screen.blit(pause_txt, txt_rect)
        
        # 3. Update Display
        pygame.display.flip()
        self.clock.tick(60)
