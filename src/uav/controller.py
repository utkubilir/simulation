"""
Uçuş Kontrol Sistemi

Manuel kontrol ve joystick/klavye girdilerini işler.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ControlState:
    """Kontrol durumu"""
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.5
    
    
class FlightController:
    """
    Uçuş Kontrol Sistemi
    
    Klavye/joystick girdilerini kontrol sinyallerine dönüştürür.
    Rate mode ve stabilize mode destekler.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Kontrol hassasiyeti
        self.aileron_sensitivity = config.get('aileron_sensitivity', 1.0)
        self.elevator_sensitivity = config.get('elevator_sensitivity', 1.0)
        self.rudder_sensitivity = config.get('rudder_sensitivity', 0.5)
        self.throttle_step = config.get('throttle_step', 0.05)
        
        # Rate limiting (birim/saniye)
        self.aileron_rate = config.get('aileron_rate', 3.0)
        self.elevator_rate = config.get('elevator_rate', 2.0)
        self.rudder_rate = config.get('rudder_rate', 1.5)
        
        # Trim değerleri
        self.aileron_trim = 0.0
        self.elevator_trim = 0.0
        self.rudder_trim = 0.0
        
        # Mevcut kontrol durumu
        self.state = ControlState()
        
        # Hedef kontrol değerleri (klavye input'u)
        self.target = ControlState()
        
        # Mod
        self.mode = "rate"  # "rate" veya "stabilize"
        
    def update(self, dt: float, inputs: Dict[str, float] = None):
        """
        Kontrol güncellemesi
        
        Args:
            dt: Zaman adımı
            inputs: Kontrol girdileri {
                'roll': -1 to 1,
                'pitch': -1 to 1,
                'yaw': -1 to 1,
                'throttle_up': bool,
                'throttle_down': bool
            }
        """
        inputs = inputs or {}
        
        # Hedef değerleri güncelle
        self.target.aileron = inputs.get('roll', 0.0) * self.aileron_sensitivity
        self.target.elevator = inputs.get('pitch', 0.0) * self.elevator_sensitivity
        self.target.rudder = inputs.get('yaw', 0.0) * self.rudder_sensitivity
        
        # Throttle
        if inputs.get('throttle_up', False):
            self.target.throttle = min(1.0, self.target.throttle + self.throttle_step)
        if inputs.get('throttle_down', False):
            self.target.throttle = max(0.0, self.target.throttle - self.throttle_step)
            
        # Yumuşak geçiş (rate limiting)
        self.state.aileron = self._smooth_control(
            self.state.aileron, self.target.aileron, self.aileron_rate, dt
        )
        self.state.elevator = self._smooth_control(
            self.state.elevator, self.target.elevator, self.elevator_rate, dt
        )
        self.state.rudder = self._smooth_control(
            self.state.rudder, self.target.rudder, self.rudder_rate, dt
        )
        self.state.throttle = self.target.throttle
        
        # Trim uygula
        final_aileron = np.clip(self.state.aileron + self.aileron_trim, -1, 1)
        final_elevator = np.clip(self.state.elevator + self.elevator_trim, -1, 1)
        final_rudder = np.clip(self.state.rudder + self.rudder_trim, -1, 1)
        
        return {
            'aileron': final_aileron,
            'elevator': final_elevator,
            'rudder': final_rudder,
            'throttle': self.state.throttle
        }
        
    def _smooth_control(self, current: float, target: float, rate: float, dt: float) -> float:
        """Kontrol değerini yumuşak geçişle güncelle"""
        diff = target - current
        max_change = rate * dt
        
        if abs(diff) <= max_change:
            return target
        else:
            return current + np.sign(diff) * max_change
            
    def set_trim(self, aileron: float = None, elevator: float = None, rudder: float = None):
        """Trim değerlerini ayarla"""
        if aileron is not None:
            self.aileron_trim = np.clip(aileron, -0.3, 0.3)
        if elevator is not None:
            self.elevator_trim = np.clip(elevator, -0.3, 0.3)
        if rudder is not None:
            self.rudder_trim = np.clip(rudder, -0.3, 0.3)
            
    def reset(self):
        """Kontrolleri sıfırla"""
        self.state = ControlState()
        self.target = ControlState()
        
    def get_controls(self) -> Dict[str, float]:
        """Mevcut kontrol değerlerini döndür"""
        return {
            'aileron': self.state.aileron,
            'elevator': self.state.elevator,
            'rudder': self.state.rudder,
            'throttle': self.state.throttle
        }


class KeyboardMapper:
    """
    Klavye tuşlarını kontrol girdilerine eşler
    
    Varsayılan düzen:
        W/S: Pitch (yukarı/aşağı)
        A/D: Roll (sola/sağa)
        Q/E: Yaw
        Shift/Ctrl: Throttle artır/azalt
    """
    
    # Pygame tuş kodları
    KEY_MAP = {
        'w': 'pitch_down',
        's': 'pitch_up',
        'a': 'roll_left',
        'd': 'roll_right',
        'q': 'yaw_left',
        'e': 'yaw_right',
        'shift': 'throttle_up',
        'ctrl': 'throttle_down',
        'lshift': 'throttle_up',
        'rshift': 'throttle_up',
        'lctrl': 'throttle_down',
        'rctrl': 'throttle_down',
    }
    
    def __init__(self):
        self.pressed_keys = set()
        
    def key_down(self, key: str):
        """Tuş basıldı"""
        self.pressed_keys.add(key.lower())
        
    def key_up(self, key: str):
        """Tuş bırakıldı"""
        self.pressed_keys.discard(key.lower())
        
    def get_inputs(self) -> Dict[str, float]:
        """Mevcut tuş durumundan kontrol girdilerini hesapla"""
        inputs = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'throttle_up': False,
            'throttle_down': False
        }
        
        for key in self.pressed_keys:
            action = self.KEY_MAP.get(key)
            if action == 'roll_left':
                inputs['roll'] = -1.0
            elif action == 'roll_right':
                inputs['roll'] = 1.0
            elif action == 'pitch_up':
                inputs['pitch'] = 1.0
            elif action == 'pitch_down':
                inputs['pitch'] = -1.0
            elif action == 'yaw_left':
                inputs['yaw'] = -1.0
            elif action == 'yaw_right':
                inputs['yaw'] = 1.0
            elif action == 'throttle_up':
                inputs['throttle_up'] = True
            elif action == 'throttle_down':
                inputs['throttle_down'] = True
                
        return inputs
        
    def reset(self):
        """Tuş durumunu sıfırla"""
        self.pressed_keys.clear()
