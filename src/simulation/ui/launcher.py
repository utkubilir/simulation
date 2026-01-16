import pygame
import yaml
from pathlib import Path
from typing import List, Dict

# Design Constants
WINDOW_SIZE = (440, 400) # Slightly taller for breathing room
COLOR_BG = (15, 15, 20)        # Very dark blue-grey
COLOR_BORDER = (40, 45, 50)    
COLOR_HEADER = (255, 255, 255)
COLOR_LABEL = (120, 130, 140)  # Muted
COLOR_PANEL = (20, 25, 30)
COLOR_INPUT_BG = (25, 30, 35)
COLOR_INPUT_BORDER = (45, 50, 55)
COLOR_ACCENT = (0, 200, 100)   # Vivid Green
COLOR_TEXT = (220, 220, 220)
COLOR_HINT = (80, 90, 100)

FONT_SIZE_HEADER = 20
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12
FONT_SIZE_BODY = 14
FONT_SIZE_DESC = 11

class Widget:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.focused = False
        self.active = False

    def handle_event(self, event):
        pass

    def draw(self, screen, fonts):
        pass

class TextInput(Widget):
    def __init__(self, x, y, w, h, label, initial_text=""):
        super().__init__(x, y, w, h)
        self.label = label
        self.text = str(initial_text)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            self.focused = self.active
        
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_TAB or event.key == pygame.K_RETURN:
                pass 
            else:
                self.text += event.unicode
                
    def draw(self, screen, fonts):
        # Label (Small, uppercase)
        lbl = fonts['label'].render(self.label.upper(), True, COLOR_LABEL)
        screen.blit(lbl, (self.rect.x, self.rect.y - 18))
        
        # Input Box
        color_border = COLOR_ACCENT if self.focused else COLOR_INPUT_BORDER
        pygame.draw.rect(screen, COLOR_INPUT_BG, self.rect, border_radius=4)
        pygame.draw.rect(screen, color_border, self.rect, 1, border_radius=4)
        
        # Text
        txt = fonts['body'].render(self.text, True, COLOR_TEXT)
        
        # Cursor
        if self.active and (pygame.time.get_ticks() // 500) % 2 == 0:
            cx = self.rect.x + 8 + txt.get_width()
            pygame.draw.line(screen, COLOR_ACCENT, (cx, self.rect.y + 6), (cx, self.rect.bottom - 6), 2)
            
        screen.set_clip(self.rect.inflate(-4, -4))
        screen.blit(txt, (self.rect.x + 8, self.rect.centery - txt.get_height()//2))
        screen.set_clip(None)

class Dropdown(Widget):
    def __init__(self, x, y, w, h, label, options, labels=None):
        super().__init__(x, y, w, h)
        self.label = label
        self.options = options # Real values
        self.labels = labels if labels else options # Display values
        self.index = 0
        
    def get_value(self):
        if 0 <= self.index < len(self.options):
            return self.options[self.index]
        return ""
        
    def get_rect(self):
        return self.rect

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.index = (self.index + 1) % len(self.options)
                self.focused = True
                self.active = True
            else:
                self.focused = False
                self.active = False
                
    def draw(self, screen, fonts):
        # Label
        lbl = fonts['label'].render(self.label.upper(), True, COLOR_LABEL)
        screen.blit(lbl, (self.rect.x, self.rect.y - 18))
        
        # Box
        color_border = COLOR_ACCENT if self.focused else COLOR_INPUT_BORDER
        pygame.draw.rect(screen, COLOR_INPUT_BG, self.rect, border_radius=4)
        pygame.draw.rect(screen, color_border, self.rect, 1, border_radius=4)
        
        # Text
        display_text = self.labels[self.index]
        txt = fonts['body'].render(display_text, True, COLOR_TEXT)
        screen.blit(txt, (self.rect.x + 8, self.rect.centery - txt.get_height()//2))
        
        # Arrow
        ax, ay = self.rect.right - 15, self.rect.centery
        pygame.draw.polygon(screen, COLOR_LABEL, [
            (ax - 5, ay - 3), (ax + 5, ay - 3), (ax, ay + 4)
        ])

class ScenarioCarousel(Widget):
    def __init__(self, x, y, w, h, label, scenarios: List[Dict[str, str]]):
        super().__init__(x, y, w, h)
        self.label = label
        self.scenarios = scenarios
        self.index = 0

    def get_value(self):
        if 0 <= self.index < len(self.scenarios):
            return self.scenarios[self.index]['name']
        return "default"

    def get_description(self):
        if 0 <= self.index < len(self.scenarios):
            return self.scenarios[self.index]['description']
        return "Default scenario configuration."

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                left_rect, right_rect = self._arrow_rects()
                if left_rect.collidepoint(event.pos):
                    self.index = (self.index - 1) % len(self.scenarios)
                elif right_rect.collidepoint(event.pos):
                    self.index = (self.index + 1) % len(self.scenarios)
                else:
                    self.index = (self.index + 1) % len(self.scenarios)
                self.focused = True
                self.active = True
            else:
                self.focused = False
                self.active = False
        elif event.type == pygame.KEYDOWN and self.focused:
            if event.key == pygame.K_LEFT:
                self.index = (self.index - 1) % len(self.scenarios)
            elif event.key == pygame.K_RIGHT:
                self.index = (self.index + 1) % len(self.scenarios)

    def _arrow_rects(self):
        size = 20
        left_rect = pygame.Rect(self.rect.x + 10, self.rect.centery - size // 2, size, size)
        right_rect = pygame.Rect(self.rect.right - 10 - size, self.rect.centery - size // 2, size, size)
        return left_rect, right_rect

    def _wrap_text(self, text, font, max_width):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def draw(self, screen, fonts):
        lbl = fonts['label'].render(self.label.upper(), True, COLOR_LABEL)
        screen.blit(lbl, (self.rect.x, self.rect.y - 18))

        color_border = COLOR_ACCENT if self.focused else COLOR_INPUT_BORDER
        pygame.draw.rect(screen, COLOR_PANEL, self.rect, border_radius=6)
        pygame.draw.rect(screen, color_border, self.rect, 1, border_radius=6)

        left_rect, right_rect = self._arrow_rects()
        pygame.draw.polygon(screen, COLOR_LABEL, [
            (left_rect.right, left_rect.top),
            (left_rect.right, left_rect.bottom),
            (left_rect.left, left_rect.centery)
        ])
        pygame.draw.polygon(screen, COLOR_LABEL, [
            (right_rect.left, right_rect.top),
            (right_rect.left, right_rect.bottom),
            (right_rect.right, right_rect.centery)
        ])

        scenario_name = self.get_value()
        description = self.get_description()

        title_font = fonts['title']
        desc_font = fonts['desc']
        name_surface = title_font.render(scenario_name.upper(), True, COLOR_TEXT)
        name_x = self.rect.centerx - name_surface.get_width() // 2
        name_y = self.rect.y + 10
        screen.blit(name_surface, (name_x, name_y))

        max_width = self.rect.width - 60
        lines = self._wrap_text(description, desc_font, max_width)
        start_y = name_y + name_surface.get_height() + 6
        for i, line in enumerate(lines[:2]):
            line_surface = desc_font.render(line, True, COLOR_HINT)
            line_x = self.rect.centerx - line_surface.get_width() // 2
            screen.blit(line_surface, (line_x, start_y + i * (line_surface.get_height() + 2)))

class Button(Widget):
    def __init__(self, x, y, w, h, text, callback):
        super().__init__(x, y, w, h)
        self.text = text
        self.callback = callback
        self.hovered = False
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                self.callback()
                
    def draw(self, screen, fonts):
        # Subtle gradient or solid
        if self.hovered:
            c_bg = COLOR_ACCENT
            c_txt = (0, 0, 0)
        else:
            c_bg = COLOR_INPUT_BG
            c_txt = COLOR_ACCENT
            
        pygame.draw.rect(screen, c_bg, self.rect, border_radius=4)
        if not self.hovered:
            pygame.draw.rect(screen, COLOR_ACCENT, self.rect, 1, border_radius=4)
            
        txt = fonts['body'].render(self.text, True, c_txt)
        screen.blit(txt, (self.rect.centerx - txt.get_width()//2, self.rect.centery - txt.get_height()//2))

class Launcher:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE, pygame.NOFRAME)
        pygame.display.set_caption("TEKNOFEST SIM")
        
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.fonts = {
            'header': pygame.font.SysFont("Verdana", FONT_SIZE_HEADER, bold=True),
            'title': pygame.font.SysFont("Verdana", FONT_SIZE_TITLE, bold=True),
            'body': pygame.font.SysFont("Verdana", FONT_SIZE_BODY),
            'label': pygame.font.SysFont("Verdana", FONT_SIZE_LABEL),
            'hint': pygame.font.SysFont("Verdana", 10, italic=True),
            'desc': pygame.font.SysFont("Verdana", FONT_SIZE_DESC)
        }
        
        # Data
        self.scenarios = self._load_scenarios()
        
        # Layout Config
        margin = 24
        input_h = 32
        gap = 42
        scenario_h = 96
        
        self.widgets: List[Widget] = []
        
        cur_y = 70
        
        # 1. Scenario (Primary)
        self.dd_scenario = ScenarioCarousel(margin, cur_y, WINDOW_SIZE[0] - margin*2, scenario_h,
                                            "Combat Scenario", self.scenarios)
        self.widgets.append(self.dd_scenario)

        cur_y += scenario_h + 26
        
        # 2. Mode (Primary)
        modes = ["ui", "headless", "ui"] # 3D maps to UI for now
        mode_lbls = ["UI (2D + Camera)", "HEADLESS (Benchmark)", "3D VIEW (Training)"]
        self.dd_mode = Dropdown(margin, cur_y, WINDOW_SIZE[0] - margin*2, input_h,
                              "Simulation Mode", modes, mode_lbls)
        self.widgets.append(self.dd_mode)
        
        cur_y += gap
        
        # 3. Seed & Duration (Secondary - Side by Side)
        w_half = (WINDOW_SIZE[0] - margin*2 - 20) // 2
        self.ti_seed = TextInput(margin, cur_y, w_half, input_h, "Random Seed", "42")
        self.ti_duration = TextInput(margin + w_half + 20, cur_y, w_half, input_h, "Duration (s)", "60")
        
        self.widgets.append(self.ti_seed)
        self.widgets.append(self.ti_duration)
        
        # 4. Start Button
        btn_y = WINDOW_SIZE[1] - 58
        self.btn_start = Button(margin, btn_y, WINDOW_SIZE[0] - margin*2, 40, "LAUNCH SIMULATION", self.start_simulation)
        self.widgets.append(self.btn_start)
        
        self.running = True
        self.result = None
        self.focus_idx = 0
        self.widgets[0].focused = True
        
    def _load_scenarios(self) -> List[Dict[str, str]]:
        scenario_dir = Path(__file__).parent.parent.parent.parent / 'scenarios'
        if not scenario_dir.exists():
            return [{"name": "default", "description": "Default scenario configuration."}]
        scenarios = []
        for scenario_file in sorted(scenario_dir.glob("*.yaml")):
            description = "Default scenario configuration."
            try:
                with open(scenario_file, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                if isinstance(data, dict):
                    description = data.get("description", description)
            except (OSError, yaml.YAMLError):
                description = "Scenario description could not be loaded."
            scenarios.append({"name": scenario_file.stem, "description": description})
        return scenarios if scenarios else [{"name": "default", "description": "Default scenario configuration."}]

    def start_simulation(self):
        try:
            seed = int(self.ti_seed.text)
        except ValueError:
            seed = 42
        try:
            duration = float(self.ti_duration.text)
        except ValueError:
            duration = 60.0
            
        self.result = {
            'scenario': self.dd_scenario.get_value(),
            'seed': seed,
            'duration': duration,
            'mode': self.dd_mode.get_value()
        }
        self.running = False

    def loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.result = None
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.result = None
                        self.running = False
                    elif event.key == pygame.K_RETURN:
                        self.start_simulation()
                    elif event.key == pygame.K_TAB:
                        self.widgets[self.focus_idx].focused = False
                        self.widgets[self.focus_idx].active = False
                        self.focus_idx = (self.focus_idx + 1) % len(self.widgets)
                        self.widgets[self.focus_idx].focused = True
                        if isinstance(self.widgets[self.focus_idx], TextInput):
                            self.widgets[self.focus_idx].active = True
                
                for w in self.widgets:
                    w.handle_event(event)

            # Draw
            self.screen.fill(COLOR_BG)
            
            # Header
            header = self.fonts['header'].render("SIMULATION LAUNCHER", True, COLOR_HEADER)
            self.screen.blit(header, (30, 20))
            
            # Separator (Subtle)
            pygame.draw.line(self.screen, COLOR_BORDER, (30, 50), (WINDOW_SIZE[0]-30, 50), 1)
            
            # Widgets
            for w in self.widgets:
                w.draw(self.screen, self.fonts)
            
            # Footer text
            ver = self.fonts['hint'].render("v2.0.4 | TEKNOFEST 2026", True, COLOR_BORDER)
            self.screen.blit(ver, (WINDOW_SIZE[0] - ver.get_width() - 30, 25))

            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()
        return self.result

if __name__ == "__main__":
    launcher = Launcher()
    res = launcher.loop()
    print(res)
