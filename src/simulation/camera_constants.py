"""
Camera Module Constants

Centralized configuration values for camera simulation parameters.
These replace hardcoded magic numbers throughout the codebase.
"""

# === Atmospheric Effects ===
DEFAULT_HAZE_DISTANCE = 500.0  # meters - visibility distance for haze effect
DEFAULT_VISIBILITY = 10000.0   # meters - maximum visibility range

# === Grid Rendering ===
GRID_SPACING = 50.0           # meters - distance between grid lines
GRID_RENDER_RANGE = 2000.0    # meters - max distance to render grid

# === Depth of Field ===
DEFAULT_FOCUS_DISTANCE = 500.0  # meters - default focus distance
DEFAULT_FOCUS_RANGE = 1000.0    # meters - depth of field range
DEFAULT_DOF_STRENGTH = 1.0      # blur intensity multiplier

# === Camera Lens ===
DEFAULT_FOV = 60.0              # degrees - field of view
DEFAULT_DISTORTION_K1 = -0.1    # barrel distortion coefficient
DEFAULT_DISTORTION_K2 = 0.02    # pincushion distortion coefficient

# === Rendering Quality ===
MIN_RENDER_DISTANCE = 0.1       # meters - objects closer than this are clipped
MAX_RENDER_DISTANCE = 1000.0    # meters - far clipping plane

# === Weather Effects ===
RAIN_DROPS_PER_FRAME = 500      # base number of rain drops at density 1.0
SNOW_FLAKES_PER_FRAME = 200     # base number of snow flakes at density 1.0

# === Sensor Simulation ===
BASE_ISO = 100                  # ISO value for clean daylight shots
NOISE_MULTIPLIER = 8.0          # noise scaling factor
