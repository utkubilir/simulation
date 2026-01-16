"""
Target Behavior Controller - Maneuver patterns for target UAVs.

Provides deterministic, parameterized maneuver behaviors:
- straight: constant heading
- constant_turn: circular flight
- zigzag: oscillating flight path
- evasive: burst + break turns
- random: seeded random maneuvers
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any
import numpy as np


class ManeuverPattern(Enum):
    """Available maneuver patterns."""
    STRAIGHT = "straight"
    CONSTANT_TURN = "constant_turn"
    ZIGZAG = "zigzag"
    EVASIVE = "evasive"
    RANDOM = "random"


@dataclass
class ManeuverCommand:
    """Output command from maneuver controller."""
    yaw_rate: float = 0.0       # rad/s, desired yaw rate
    pitch_rate: float = 0.0     # rad/s, desired pitch rate
    roll_rate: float = 0.0      # rad/s, desired roll rate
    throttle: float = 0.7       # 0-1, throttle setting
    
    # Alternative: direct control surfaces
    aileron: float = 0.0        # -1 to 1
    elevator: float = 0.0       # -1 to 1
    rudder: float = 0.0         # -1 to 1


@dataclass
class ManeuverParams:
    """Default parameters for each maneuver type."""
    # Common
    base_speed: float = 25.0         # m/s
    base_altitude: float = 100.0     # m
    
    # constant_turn
    turn_rate: float = 15.0          # deg/s
    turn_direction: int = 1          # 1=right, -1=left
    
    # zigzag
    zigzag_period: float = 4.0       # seconds per full cycle
    zigzag_amplitude: float = 30.0   # degrees max yaw deviation
    
    # evasive
    evasive_burst_duration: float = 1.5   # seconds of straight flight
    evasive_break_duration: float = 1.0   # seconds of break turn
    evasive_break_rate: float = 40.0      # deg/s during break
    
    # random
    random_change_interval: float = 2.0   # seconds between random changes
    random_max_turn_rate: float = 20.0    # deg/s max


class TargetManeuverController:
    """
    Controls target UAV maneuver behavior.
    
    Uses passed-in RNG for determinism - does NOT use global random.
    """
    
    def __init__(self, pattern: str, params: Dict = None, rng: np.random.Generator = None):
        """
        Initialize maneuver controller.
        
        Args:
            pattern: Maneuver pattern name (straight, constant_turn, zigzag, evasive, random)
            params: Override parameters dict
            rng: NumPy random generator for determinism (required for random pattern)
        """
        self.pattern = ManeuverPattern(pattern) if isinstance(pattern, str) else pattern
        self.rng = rng if rng is not None else np.random.default_rng(42)
        
        # Build parameters
        self.params = ManeuverParams()
        if params:
            for key, value in params.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
                    
        # Internal state
        self._time = 0.0
        self._phase = 0             # For multi-phase maneuvers
        self._phase_time = 0.0      # Time in current phase
        self._current_turn_rate = 0.0
        self._next_change_time = 0.0
        
        # Pre-compute random sequence for random pattern
        if self.pattern == ManeuverPattern.RANDOM:
            self._random_sequence = self._generate_random_sequence(100)
            self._random_index = 0
            
    def _generate_random_sequence(self, count: int):
        """Generate deterministic random sequence."""
        sequence = []
        for _ in range(count):
            sequence.append({
                'turn_rate': self.rng.uniform(-self.params.random_max_turn_rate, 
                                               self.params.random_max_turn_rate),
                'duration': self.rng.uniform(1.0, self.params.random_change_interval * 2)
            })
        return sequence
        
    def reset(self):
        """Reset controller state."""
        self._time = 0.0
        self._phase = 0
        self._phase_time = 0.0
        self._current_turn_rate = 0.0
        self._next_change_time = 0.0
        self._random_index = 0
        
        # Regenerate random sequence for fresh variety on reset
        if self.pattern == ManeuverPattern.RANDOM:
            self._random_sequence = self._generate_random_sequence(100)
        
    def update(self, dt: float, target_state: Dict, own_state: Dict = None) -> ManeuverCommand:
        """
        Update maneuver and get control command.
        
        Args:
            dt: Time step
            target_state: Current target UAV state dict
            own_state: Current player UAV state dict (for reactive behaviors)
            
        Returns:
            ManeuverCommand with control inputs
        """
        self._time += dt
        self._phase_time += dt
        
        if self.pattern == ManeuverPattern.STRAIGHT:
            return self._straight(dt, target_state)
        elif self.pattern == ManeuverPattern.CONSTANT_TURN:
            return self._constant_turn(dt, target_state)
        elif self.pattern == ManeuverPattern.ZIGZAG:
            return self._zigzag(dt, target_state)
        elif self.pattern == ManeuverPattern.EVASIVE:
            return self._evasive(dt, target_state, own_state)
        elif self.pattern == ManeuverPattern.RANDOM:
            return self._random(dt, target_state)
        else:
            return ManeuverCommand()
            
    def _straight(self, dt: float, target_state: Dict) -> ManeuverCommand:
        """Straight flight - maintain heading."""
        return ManeuverCommand(
            yaw_rate=0.0,
            throttle=0.7,
            aileron=0.0,
            elevator=0.0,
            rudder=0.0
        )
        
    def _constant_turn(self, dt: float, target_state: Dict) -> ManeuverCommand:
        """Constant rate turn."""
        turn_rate_rad = math.radians(self.params.turn_rate) * self.params.turn_direction
        
        # Convert yaw rate to bank angle for coordinated turn
        # Approximate: bank = atan(v * yaw_rate / g)
        speed = target_state.get('speed', self.params.base_speed)
        bank_angle = math.atan2(speed * turn_rate_rad, 9.81)
        
        # Aileron for bank, rudder for coordination
        aileron = np.clip(bank_angle / math.radians(30), -1, 1)
        
        return ManeuverCommand(
            yaw_rate=turn_rate_rad,
            throttle=0.75,  # Slightly more for turn
            aileron=aileron * self.params.turn_direction,
            elevator=0.05,  # Slight back pressure in turn
            rudder=0.2 * self.params.turn_direction
        )
        
    def _zigzag(self, dt: float, target_state: Dict) -> ManeuverCommand:
        """Zigzag oscillation pattern."""
        # Sinusoidal yaw rate
        omega = 2 * math.pi / self.params.zigzag_period
        phase = self._time * omega
        
        # Yaw rate varies sinusoidally
        max_rate_rad = math.radians(self.params.zigzag_amplitude) * omega
        yaw_rate = max_rate_rad * math.sin(phase)
        
        # Convert to control surfaces
        direction = 1 if yaw_rate >= 0 else -1
        intensity = abs(yaw_rate) / max_rate_rad
        
        return ManeuverCommand(
            yaw_rate=yaw_rate,
            throttle=0.7,
            aileron=0.4 * intensity * direction,
            elevator=0.0,
            rudder=0.3 * intensity * direction
        )
        
    def _evasive(self, dt: float, target_state: Dict, own_state: Dict = None) -> ManeuverCommand:
        """Evasive maneuver - burst straight, then break turn."""
        total_cycle = self.params.evasive_burst_duration + self.params.evasive_break_duration
        
        # Determine phase
        if self._phase_time >= total_cycle:
            self._phase_time = 0.0
            self._phase = (self._phase + 1) % 2  # Alternate break direction
            
        in_burst = self._phase_time < self.params.evasive_burst_duration
        
        if in_burst:
            # Straight flight (burst)
            return ManeuverCommand(
                yaw_rate=0.0,
                throttle=0.9,  # High speed
                aileron=0.0,
                elevator=-0.1,  # Slight dive for speed
                rudder=0.0
            )
        else:
            # Break turn
            direction = 1 if self._phase == 0 else -1
            turn_rate_rad = math.radians(self.params.evasive_break_rate) * direction
            
            return ManeuverCommand(
                yaw_rate=turn_rate_rad,
                throttle=0.6,
                aileron=0.7 * direction,
                elevator=0.2,  # Pull in turn
                rudder=0.4 * direction
            )
            
    def _random(self, dt: float, target_state: Dict) -> ManeuverCommand:
        """Random maneuvers - deterministic from seed."""
        # Check if we need to change
        if self._time >= self._next_change_time:
            if self._random_index < len(self._random_sequence):
                current = self._random_sequence[self._random_index]
                self._current_turn_rate = current['turn_rate']
                self._next_change_time = self._time + current['duration']
                self._random_index += 1
            else:
                # Regenerate sequence
                self._random_sequence = self._generate_random_sequence(100)
                self._random_index = 0
                
        turn_rate_rad = math.radians(self._current_turn_rate)
        direction = 1 if self._current_turn_rate >= 0 else -1
        intensity = abs(self._current_turn_rate) / self.params.random_max_turn_rate
        
        return ManeuverCommand(
            yaw_rate=turn_rate_rad,
            throttle=0.7,
            aileron=0.3 * intensity * direction,
            elevator=0.0,
            rudder=0.2 * intensity * direction
        )
        
    def get_info(self) -> Dict:
        """Get current maneuver info for logging."""
        return {
            'pattern': self.pattern.value,
            'time': self._time,
            'phase': self._phase,
            'phase_time': self._phase_time
        }
