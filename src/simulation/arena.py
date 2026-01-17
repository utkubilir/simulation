"""
Teknofest Savaşan İHA Arena System

Defines the competition arena with:
- Arena boundaries
- Safe zones (takeoff/landing areas)
- Combat zone
- Zone detection for gameplay logic
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Zone:
    """Represents a rectangular zone in the arena."""
    name: str
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    min_y: float = 0.0
    max_y: float = 150.0
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    def contains(self, position: np.ndarray) -> bool:
        """Check if a position is within this zone."""
        x, y, z = position[0], position[1], position[2]
        return (self.min_x <= x <= self.max_x and
                self.min_z <= z <= self.max_z and
                self.min_y <= y <= self.max_y)
    
    @property
    def center(self) -> np.ndarray:
        """Get center of the zone."""
        return np.array([
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2
        ], dtype=np.float32)
    
    @property
    def size(self) -> Tuple[float, float, float]:
        """Get size of the zone (width, height, depth)."""
        return (
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z
        )


class TeknofestArena:
    """
    Teknofest Savaşan İHA Competition Arena
    
    Arena layout:
    - 500m x 500m total area
    - 4 safe zones at corners (50m x 50m each)
    - Central combat zone
    - Altitude limits: 10m - 150m
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Arena dimensions
        self.width = config.get('width', 500.0)  # X axis
        self.depth = config.get('depth', 500.0)  # Z axis
        self.min_altitude = config.get('min_altitude', 10.0)
        self.max_altitude = config.get('max_altitude', 150.0)
        
        # Safe zone size
        self.safe_zone_size = config.get('safe_zone_size', 50.0)
        
        # Arena center (for centering around origin)
        self.center_x = 0.0
        self.center_z = 0.0
        
        # Calculate boundaries (centered at origin)
        half_w = self.width / 2
        half_d = self.depth / 2
        
        self.bounds = Zone(
            name="Arena",
            min_x=-half_w,
            max_x=half_w,
            min_z=-half_d,
            max_z=half_d,
            min_y=0.0,
            max_y=self.max_altitude
        )
        
        # Create safe zones (corners)
        sz = self.safe_zone_size
        self.safe_zones: List[Zone] = [
            Zone(
                name="SafeZone_NW",
                min_x=-half_w, max_x=-half_w + sz,
                min_z=half_d - sz, max_z=half_d,
                min_y=0.0, max_y=self.max_altitude,
                color=(0.2, 0.6, 0.2)  # Green
            ),
            Zone(
                name="SafeZone_NE",
                min_x=half_w - sz, max_x=half_w,
                min_z=half_d - sz, max_z=half_d,
                min_y=0.0, max_y=self.max_altitude,
                color=(0.2, 0.2, 0.6)  # Blue
            ),
            Zone(
                name="SafeZone_SW",
                min_x=-half_w, max_x=-half_w + sz,
                min_z=-half_d, max_z=-half_d + sz,
                min_y=0.0, max_y=self.max_altitude,
                color=(0.6, 0.6, 0.2)  # Yellow
            ),
            Zone(
                name="SafeZone_SE",
                min_x=half_w - sz, max_x=half_w,
                min_z=-half_d, max_z=-half_d + sz,
                min_y=0.0, max_y=self.max_altitude,
                color=(0.6, 0.2, 0.2)  # Red
            )
        ]
        
        # Combat zone (center area, excluding safe zones)
        combat_margin = sz + 20  # 20m buffer from safe zones
        self.combat_zone = Zone(
            name="CombatZone",
            min_x=-half_w + combat_margin,
            max_x=half_w - combat_margin,
            min_z=-half_d + combat_margin,
            max_z=half_d - combat_margin,
            min_y=self.min_altitude,
            max_y=self.max_altitude,
            color=(0.8, 0.3, 0.3)  # Light red
        )
        
        # Spawn points for each team (in safe zones)
        self.spawn_points = {
            'team_a': np.array([-half_w + sz/2, 20.0, half_d - sz/2], dtype=np.float32),
            'team_b': np.array([half_w - sz/2, 20.0, -half_d + sz/2], dtype=np.float32),
        }
        
        # Boundary markers (poles at corners and edges)
        self.markers = self._generate_boundary_markers()
        
        # Detail objects (tents, boxes, etc.)
        self.detail_objects = self._generate_detail_objects()
    
    def _generate_detail_objects(self) -> List[dict]:
        """Generate decorative detail objects like tents and containers."""
        details = []
        half_w = self.width / 2
        half_d = self.depth / 2
        sz = self.safe_zone_size
        
        # Place team tents in safe zones
        # NW Safe Zone (Team A-ish)
        details.append({
            'type': 'tent',
            'position': np.array([-half_w + 10, 0, half_d - 15], dtype=np.float32),
            'rotation': 0.0,
            'color': (0.8, 0.8, 0.8), # White tent
            'scale': (4.0, 3.0, 6.0)
        })
        details.append({
            'type': 'box',
            'position': np.array([-half_w + 15, 0, half_d - 10], dtype=np.float32),
            'rotation': 0.5,
            'color': (0.4, 0.4, 0.5), # Blue container
            'scale': (2.4, 2.6, 6.0)
        })

        # NE Safe Zone
        details.append({
            'type': 'tent',
            'position': np.array([half_w - 15, 0, half_d - 10], dtype=np.float32),
            'rotation': 1.57, # 90 degrees
            'color': (0.2, 0.2, 0.7), # Blue tent
            'scale': (4.0, 3.0, 6.0)
        })
        
        # SW Safe Zone (Team B-ish)
        details.append({
            'type': 'tent',
            'position': np.array([-half_w + 15, 0, -half_d + 10], dtype=np.float32),
            'rotation': 0.0,
            'color': (0.7, 0.2, 0.2), # Red tent
            'scale': (4.0, 3.0, 6.0)
        })
        
        # SE Safe Zone
        details.append({
            'type': 'tent',
            'position': np.array([half_w - 10, 0, -half_d + 15], dtype=np.float32),
            'rotation': 1.57,
            'color': (0.7, 0.7, 0.2), # Yellow tent
            'scale': (4.0, 3.0, 6.0)
        })
        details.append({
            'type': 'box',
            'position': np.array([half_w - 20, 0, -half_d + 10], dtype=np.float32),
            'rotation': -0.3,
            'color': (0.5, 0.5, 0.5), # Grey box
            'scale': (2.0, 2.0, 2.0)
        })

        return details

    def _generate_boundary_markers(self) -> List[dict]:
        """Generate boundary marker positions."""
        markers = []
        half_w = self.width / 2
        half_d = self.depth / 2
        
        # Corner markers (tall poles)
        corners = [
            (-half_w, -half_d),
            (-half_w, half_d),
            (half_w, -half_d),
            (half_w, half_d),
        ]
        
        for x, z in corners:
            markers.append({
                'type': 'corner_pole',
                'position': np.array([x, 0, z], dtype=np.float32),
                'height': 30.0,
                'color': (1.0, 0.0, 0.0),  # Red
                'radius': 0.5
            })
        
        # Edge markers (Cones every 50m)
        spacing = 50.0
        
        def add_cones(start, end, constant_coord, axis='x'):
            coords = np.arange(start + spacing, end, spacing)
            for val in coords:
                if axis == 'x':
                    pos = np.array([val, 0, constant_coord], dtype=np.float32)
                else:
                    pos = np.array([constant_coord, 0, val], dtype=np.float32)
                
                markers.append({
                    'type': 'cone',
                    'position': pos,
                    'height': 2.0,
                    'color': (1.0, 0.5, 0.0),  # Orange
                    'radius': 0.5
                })

        # North and South edges
        add_cones(-half_w, half_w, -half_d, axis='x')
        add_cones(-half_w, half_w, half_d, axis='x')
        
        # East and West edges
        add_cones(-half_d, half_d, -half_w, axis='z')
        add_cones(-half_d, half_d, half_w, axis='z')
        
        # Helipads in Safe Zones (Rings)
        # NW Safe Zone (Team A)
        markers.append({
            'type': 'helipad',
            'position': self.spawn_points['team_a'].copy(),
            'radius': 5.0,
            'color': (1.0, 1.0, 0.0)  # Yellow ring
        })
        
        # SE Safe Zone (Team B)
        markers.append({
            'type': 'helipad',
            'position': self.spawn_points['team_b'].copy(),
            'radius': 5.0,
            'color': (1.0, 1.0, 0.0)
        })
        
        # SW Safe Zone
        sz = self.safe_zone_size
        pos_sw = np.array([-half_w + sz/2, 0.2, -half_d + sz/2], dtype=np.float32)
        markers.append({
            'type': 'helipad',
            'position': pos_sw,
            'radius': 5.0,
            'color': (1.0, 1.0, 1.0) # White
        })
        
        # NE Safe Zone
        pos_ne = np.array([half_w - sz/2, 0.2, half_d - sz/2], dtype=np.float32)
        markers.append({
            'type': 'helipad',
            'position': pos_ne,
            'radius': 5.0,
            'color': (1.0, 1.0, 1.0)
        })
        
        # Ensure helipad marking Y is near ground
        for m in markers:
            if m['type'] == 'helipad':
                m['position'][1] = 0.2
        
        return markers
    
    def is_in_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within arena bounds."""
        return self.bounds.contains(position)
    
    def is_in_safe_zone(self, position: np.ndarray) -> Optional[Zone]:
        """Check if position is in any safe zone. Returns the zone or None."""
        for zone in self.safe_zones:
            if zone.contains(position):
                return zone
        return None
    
    def is_in_combat_zone(self, position: np.ndarray) -> bool:
        """Check if position is in the combat zone."""
        return self.combat_zone.contains(position) and self.is_in_safe_zone(position) is None
    
    def get_zone_at(self, position: np.ndarray) -> str:
        """Get the name of the zone at the given position."""
        if not self.is_in_bounds(position):
            return "OutOfBounds"
        
        safe = self.is_in_safe_zone(position)
        if safe:
            return safe.name
        
        if self.is_in_combat_zone(position):
            return "CombatZone"
        
        return "TransitionZone"
    
    def check_altitude_violation(self, position: np.ndarray) -> Optional[str]:
        """Check if altitude is within limits. Returns violation type or None."""
        y = position[1]
        if y < self.min_altitude:
            return "TOO_LOW"
        if y > self.max_altitude:
            return "TOO_HIGH"
        return None
    
    def get_spawn_point(self, team: str) -> np.ndarray:
        """Get spawn point for a team."""
        return self.spawn_points.get(team, np.array([0, 20, 0], dtype=np.float32))
    
    def to_dict(self) -> dict:
        """Export arena configuration as dictionary."""
        return {
            'width': self.width,
            'depth': self.depth,
            'min_altitude': self.min_altitude,
            'max_altitude': self.max_altitude,
            'safe_zone_size': self.safe_zone_size,
            'safe_zones': [z.name for z in self.safe_zones],
            'spawn_points': {k: v.tolist() for k, v in self.spawn_points.items()},
            'marker_count': len(self.markers)
        }
