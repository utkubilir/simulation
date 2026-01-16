import pygame
import glob
import os
import yaml
from pathlib import Path
from .theme import Colors, Fonts

class Menu:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.running = True
        self.config = {}
        
        # State
        self.scenarios = self._load_scenarios()
        self.selected_scenario_idx = 0
        self.enemy_count = 3
        self.difficulty_options = ['Easy', 'Medium', 'Hard']
        self.difficulty_idx = 1 # Medium default
        self.seed = 42
        self.seed_input_active = False
        self.seed_text = "42"
        
        # Initialize
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Savaşan İHA - Simülasyon Ayarları")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_header = pygame.font.SysFont("Verdana", 40, bold=True)
        self.font_label = pygame.font.SysFont("Verdana", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
    def _load_scenarios(self):
        """List all yaml files in scenarios/"""
        # Assume generic path relative to this file
        # src/simulation/ui/menu.py -> ../../../scenarios
        base_path = Path(__file__).parent.parent.parent.parent
        scenarios_dir = base_path / 'scenarios'
        
        files = glob.glob(str(scenarios_dir / "*.yaml"))
        names = [Path(f).stem for f in files]
        names.sort()
        if not names:
            names = ['default']
        return names
        
    def run(self):
        """Main loop"""
        while self.running:
            self._handle_events()
            self._render()
            self.clock.tick(60)
            
        return self.config
        
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.config = None # Signal abort
                
            elif event.type == pygame.KEYDOWN:
                if self.seed_input_active:
                    if event.key == pygame.K_RETURN:
                        self.seed_input_active = False
                        try:
                            self.seed = int(self.seed_text)
                        except ValueError:
                            self.seed = 42
                            self.seed_text = "42"
                    elif event.key == pygame.K_BACKSPACE:
                        self.seed_text = self.seed_text[:-1]
                    else:
                        if event.unicode.isnumeric():
                            self.seed_text += event.unicode
                else:
                    # Navigation
                    if event.key == pygame.K_UP:
                        pass # Could implement proper focus navigation
                    elif event.key == pygame.K_DOWN:
                        pass
                        
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._handle_click(event.pos)
                    
    def _handle_click(self, pos):
        x, y = pos
        
        # Define areas (rough coords based on render)
        # Scenario Prev/Next
        if 400 < y < 440:
            if x < 350: # Prev
                self.selected_scenario_idx = (self.selected_scenario_idx - 1) % len(self.scenarios)
            elif x > 600: # Next
                self.selected_scenario_idx = (self.selected_scenario_idx + 1) % len(self.scenarios)
                
        # Enemy Count -/
        if 460 < y < 500:
            if x < 350:
                self.enemy_count = max(1, self.enemy_count - 1)
            elif x > 600:
                self.enemy_count = min(10, self.enemy_count + 1)
                
        # Difficulty
        if 520 < y < 560:
            if x < 350:
                self.difficulty_idx = (self.difficulty_idx - 1) % 3
            elif x > 600:
                self.difficulty_idx = (self.difficulty_idx + 1) % 3
                
        # Seed Input
        seed_rect = pygame.Rect(350, 580, 200, 30)
        # Check against render logic for Y pos
        # Let's use constant layout in render:
        # Start Button
        start_rect = pygame.Rect(self.width//2 - 100, 500, 200, 60)
        
        # Re-calc layout just like render to be precise or use simple zones
        # Let's re-align interaction with render function
        
        # 1. Scenario Row (y=150)
        row_h = 60
        y_scen = 150
        if y_scen < y < y_scen+40:
            if 560 < x < 600: # Right arrow (Center 400 + 160 = 560)
                self.selected_scenario_idx = (self.selected_scenario_idx + 1) % len(self.scenarios)
            elif 220 < x < 260: # Left arrow (Center 400 - 180 = 220)
                self.selected_scenario_idx = (self.selected_scenario_idx - 1) % len(self.scenarios)
                
        # 2. Enemy Count (y=230)
        y_n = 230
        if y_n < y < y_n+40:
            if 560 < x < 600:
                self.enemy_count = min(10, self.enemy_count + 1)
            elif 220 < x < 260:
                self.enemy_count = max(1, self.enemy_count - 1)

        # 3. Difficulty (y=310)
        y_diff = 310
        if y_diff < y < y_diff+40:
            if 560 < x < 600: 
                self.difficulty_idx = (self.difficulty_idx + 1) % 3
            elif 220 < x < 260:
                self.difficulty_idx = (self.difficulty_idx - 1) % 3
                
        # 4. Seed (y=390)
        y_seed = 390
        # Click on text box
        if 350 < x < 450 and y_seed < y < y_seed+40:
            self.seed_input_active = True
        else:
            if self.seed_input_active:
                # Commit basic
                try:
                    self.seed = int(self.seed_text)
                except:
                    self.seed = 42
            self.seed_input_active = False

        # START BUTTON
        y_start = 500
        if self.width//2 - 100 < x < self.width//2 + 100 and y_start < y < y_start + 60:
            self._start_sim()

    def _start_sim(self):
        """Prepare config and exit"""
        scenario_name = self.scenarios[self.selected_scenario_idx]
        
        # Load yaml to get basics
        # We will let SimulationRunner load it, we just pass the name
        # But we need to override the CLI-like args
        
        self.config = {
            'scenario': scenario_name,
            'enemy_count': self.enemy_count,
            'seed': self.seed,
            'difficulty': self.difficulty_options[self.difficulty_idx].lower(),
            'mode': 'ui' # Default
        }
        self.running = False
        
    def _render(self):
        self.screen.fill(Colors.BACKGROUND)
        
        cx = self.width // 2
        
        # Title
        title = self.font_header.render("SIMULATION CONFIG", True, Colors.PRIMARY)
        self.screen.blit(title, (cx - title.get_width()//2, 50))
        
        # Helper for rows
        def draw_row(y, label, value_text):
            # Label
            lbl = self.font_label.render(label, True, Colors.TEXT_MAIN)
            self.screen.blit(lbl, (100, y))
            
            # Value
            val = self.font_label.render(value_text, True, Colors.SUCCESS)
            val_rect = val.get_rect(center=(cx, y + 15))
            self.screen.blit(val, val_rect)
            
            # Arrows
            arr_l = self.font_label.render("<", True, Colors.ACCENT)
            arr_r = self.font_label.render(">", True, Colors.ACCENT)
            # Widen the gap to prevent overlap with long scenario names
            self.screen.blit(arr_l, (cx - 180, y)) 
            self.screen.blit(arr_r, (cx + 160, y))
            
        # 1. Scenario
        cur_scen = self.scenarios[self.selected_scenario_idx]
        draw_row(150, "SCENARIO", cur_scen)
        
        # 2. Enemy Count
        draw_row(230, "ENEMIES", str(self.enemy_count))
        
        # 3. Difficulty
        draw_row(310, "DIFFICULTY", self.difficulty_options[self.difficulty_idx].upper())
        
        # 4. Seed
        y_seed = 390
        lbl = self.font_label.render("SEED", True, Colors.TEXT_MAIN)
        self.screen.blit(lbl, (100, y_seed))
        
        # Minimal Input Line
        col = Colors.ACCENT if self.seed_input_active else Colors.PANEL_BORDER
        # Draw just a line at bottom
        pygame.draw.line(self.screen, col, (cx - 50, y_seed + 30), (cx + 50, y_seed + 30), 1)
        
        txt = self.font_label.render(self.seed_text, True, Colors.TEXT_MAIN)
        self.screen.blit(txt, (cx - txt.get_width()//2, y_seed))
        
        # START BUTTON (Minimal)
        y_start = 500
        btn_rect = pygame.Rect(cx - 100, y_start, 200, 50)
        
        mouse = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mouse)
        
        if hover:
            # Filled white/accent
            pygame.draw.rect(self.screen, Colors.ACCENT, btn_rect)
            start_txt = self.font_header.render("START", True, Colors.BACKGROUND)
        else:
            # Thin outline
            pygame.draw.rect(self.screen, Colors.TEXT_MAIN, btn_rect, 1)
            start_txt = self.font_header.render("START", True, Colors.TEXT_MAIN)
            
        self.screen.blit(start_txt, (cx - start_txt.get_width()//2, y_start + 5))
        
        pygame.display.flip()

if __name__ == "__main__":
    m = Menu()
    print(m.run())
