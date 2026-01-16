
"""
Tactical Glass Theme Configuration
Defines colors, fonts, and layout constants for the modern simulation UI.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Colors:
    # MINIMALIST PALETTE
    BACKGROUND = (18, 18, 18)       # Matte Black
    background_alpha = 240          # High opacity
    
    PRIMARY = (220, 220, 220)       # Off-White
    PRIMARY_DIM = (100, 100, 100)   # Dark Grey
    ACCENT = (255, 255, 255)        # Pure White
    
    WARNING = (255, 204, 0)         # Clean Amber
    DANGER = (235, 87, 87)          # Flat Red
    SUCCESS = (39, 174, 96)         # Flat Green
    
    TEXT_MAIN = (240, 240, 240)
    TEXT_DIM = (140, 140, 140)
    
    # UI ELEMENTS
    PANEL_BG = (30, 30, 30)
    PANEL_BORDER = (60, 60, 60)     # Subtle border
    
    HUD_LINE = (255, 255, 255, 120) # White transparent

@dataclass
class Layout:
    PADDING = 20
    MARGIN = 10
    
    # Top Bar
    TOP_BAR_HEIGHT = 60
    
    # Side Panels
    SIDE_PANEL_WIDTH = 250
    
    # Radar
    RADAR_SIZE = 200
    RADAR_CENTER_X = 1280 - PADDING - (RADAR_SIZE // 2)
    RADAR_CENTER_Y = TOP_BAR_HEIGHT + PADDING + (RADAR_SIZE // 2)

@dataclass
class Fonts:
    # Font sizes (will require loading actual font files, defaulting to sys font for now)
    HEADER_SIZE = 24
    NORMAL_SIZE = 16
    SMALL_SIZE = 12
    HUGE_SIZE = 48
