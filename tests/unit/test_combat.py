import pytest
import numpy as np
from src.uav.combat import CombatStateManager, CombatState, PursuitLogic, SearchPattern
from src.vision.tracker import Track

def test_search_pattern_generation():
    pattern = SearchPattern.lawnmower(np.array([0,0,100]), 1000, 1000, 200)
    assert len(pattern) > 0
    assert len(pattern) % 2 == 0 # Pairs of waypoints
    assert np.allclose(pattern[0][2], 100) # Altitude check

def test_pursuit_logic_pn():
    my_pos = np.array([0., 0., 100.])
    my_vel = np.array([10., 0., 0.])
    
    # Target directly ahead moving away
    target_pos = np.array([100., 0., 100.])
    target_vel = np.array([10., 0., 0.])
    
    heading_cmd, pitch_cmd = PursuitLogic.proportional_navigation(my_pos, my_vel, target_pos, target_vel)
    
    # Converting heading/pitch from diff often results in simple atan2
    # In pure pursuit scenario (aligned), heading should be 0 (east)
    assert np.abs(heading_cmd) < 0.1
    assert np.abs(pitch_cmd) < 0.1

def test_state_machine_transitions():
    manager = CombatStateManager()
    uav_state = {
        'position': [1000,1000,100],
        'heading': 0,
        'velocity': [10,0,0],
        'altitude': 100,
        'speed': 10
    }
    
    # 1. Start in SEARCH
    decision = manager.update(uav_state, [])
    assert manager.state == CombatState.SEARCH
    assert decision['mode'] == 'waypoint'
    
    # 2. Transition to TRACK upon detection
    track = Track(
        id=1, 
        bbox=(0,0,10,10), 
        center=(5,5), 
        confidence=0.9,
        is_confirmed=True,
        world_pos=np.array([1100, 1000, 100]),
        velocity=np.array([0,0,0])
    )
    
    decision = manager.update(uav_state, [track])
    assert manager.state == CombatState.TRACK
    assert decision['mode'] == 'track'
    assert decision['params']['heading'] is not None

def test_target_selection_priority():
    manager = CombatStateManager()
    uav_state = {
        'position': [1000,1000,100],
        'velocity': [0,0,0],
        'altitude': 100,
        'speed': 10,
        'heading': 0
    }
    
    t1 = Track(id=1, bbox=(0,0,0,0), center=(0,0), confidence=0.8, is_confirmed=False)
    t2 = Track(id=2, bbox=(0,0,0,0), center=(0,0), confidence=0.9, is_confirmed=True, world_pos=np.array([1100,1000,100]), velocity=np.zeros(3))
    
    manager.update(uav_state, [t1, t2])
    assert manager.target_id == 2
    assert manager.state == CombatState.TRACK

def test_visual_servoing():
    from src.uav.combat import VisualServo
    servo = VisualServo()
    
    # Target at center -> No command
    cmd = servo.calculate_commands((320, 240), (640, 480))
    assert cmd == (0, 0)
    
    # Target right (x=640) -> Roll positive
    cmd = servo.calculate_commands((640, 240), (640, 480))
    assert cmd[0] > 0
    
    # Target below (y=480) -> Pitch negative (down)
    cmd = servo.calculate_commands((320, 480), (640, 480))
    assert cmd[1] < 0

def test_lock_transition_and_command():
    manager = CombatStateManager()
    # Mock UAV close to target
    uav_state = {
        'position': [1000,1000,100],
        'velocity': [0,0,0],
        'altitude': 100,
        'speed': 10,
        'heading': 0
    }
    
    # Track is very close (distance < engagement_distance)
    t = Track(
        id=1, bbox=(0,0,0,0), center=(320, 240), 
        confidence=1.0, is_confirmed=True,
        world_pos=np.array([1010, 1000, 100]), # 10m away
        velocity=np.zeros(3)
    )
    
    # Force state to TRACK first to allow transition
    manager.state = CombatState.TRACK
    manager.target_id = 1
    
    decision = manager.update(uav_state, [t])
    
    # Should transition to LOCK
    assert manager.state == CombatState.LOCK
    
    # Mode should be 'direct'
    assert decision['mode'] == 'direct'
    assert 'roll' in decision['params']
    assert 'pitch' in decision['params']
