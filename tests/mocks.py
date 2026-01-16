"""
Mock Objects for Testing

Provides mock implementations of simulation components for isolated testing.
"""

from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


# =============================================================================
# UAV MOCKS
# =============================================================================

def create_mock_uav_state(
    position: Optional[np.ndarray] = None,
    velocity: Optional[np.ndarray] = None,
    heading: float = 0.0,
    altitude: float = 100.0,
    speed: float = 20.0
) -> Dict[str, Any]:
    """
    Create a mock UAV state dict.
    
    Args:
        position: UAV position [x, y, z], defaults to [1000, 1000, 100]
        velocity: UAV velocity [vx, vy, vz], defaults to [20, 0, 0]
        heading: Heading in degrees
        altitude: Altitude in meters
        speed: Speed in m/s
        
    Returns:
        UAV state dictionary
    """
    if position is None:
        position = np.array([1000.0, 1000.0, altitude])
    if velocity is None:
        velocity = np.array([speed, 0.0, 0.0])
        
    return {
        'position': position,
        'velocity': velocity,
        'orientation': np.array([0.0, 0.0, np.radians(heading)]),
        'heading': heading,
        'altitude': altitude,
        'speed': speed,
        'throttle': 0.7,
        'battery': 1.0,
        'is_player': True
    }


def create_mock_enemy_state(
    id: str = 'target_1',
    position: Optional[np.ndarray] = None,
    heading: float = 180.0
) -> Dict[str, Any]:
    """Create a mock enemy UAV state."""
    if position is None:
        position = np.array([1200.0, 1000.0, 100.0])
        
    return {
        'id': id,
        'position': position,
        'velocity': np.array([-15.0, 0.0, 0.0]),
        'heading': heading,
        'team': 'red',
        'is_player': False
    }


# =============================================================================
# DETECTION & TRACKING MOCKS
# =============================================================================

def create_mock_detection(
    bbox: Tuple[int, int, int, int] = (300, 220, 340, 260),
    confidence: float = 0.85,
    world_id: str = 'target_1',
    distance: float = 150.0
) -> Dict[str, Any]:
    """
    Create a mock detection dict.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        confidence: Detection confidence [0, 1]
        world_id: ID of the detected object
        distance: Distance to target in meters
        
    Returns:
        Detection dictionary
    """
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return {
        'bbox': bbox,
        'center': center,
        'confidence': confidence,
        'world_id': world_id,
        'distance': distance,
        'width': bbox[2] - bbox[0],
        'height': bbox[3] - bbox[1]
    }


def create_mock_track(
    id: int = 1,
    center: Tuple[float, float] = (320, 240),
    confidence: float = 0.9,
    is_confirmed: bool = True,
    world_pos: Optional[np.ndarray] = None
) -> Mock:
    """
    Create a mock Track object.
    
    Args:
        id: Track ID
        center: Center position in frame (x, y)
        confidence: Track confidence
        is_confirmed: Whether track is confirmed
        world_pos: 3D world position (optional)
        
    Returns:
        Mock Track object
    """
    mock = Mock()
    mock.id = id
    mock.center = center
    mock.confidence = confidence
    mock.bbox = (center[0] - 20, center[1] - 20, center[0] + 20, center[1] + 20)
    mock.world_pos = world_pos
    mock.is_confirmed = is_confirmed
    mock.age = 10
    mock.hits = 5
    mock.velocity = (0, 0)
    return mock


def create_mock_tracks(count: int = 3, spread: int = 100) -> List[Mock]:
    """Create multiple mock tracks spread across the frame."""
    tracks = []
    for i in range(count):
        center = (320 + (i - count // 2) * spread, 240)
        tracks.append(create_mock_track(id=i + 1, center=center))
    return tracks


# =============================================================================
# SIMULATION COMPONENT MOCKS
# =============================================================================

def create_mock_camera(resolution: Tuple[int, int] = (640, 480)) -> MagicMock:
    """
    Create a mock camera that returns black frames.
    
    Args:
        resolution: Camera resolution (width, height)
        
    Returns:
        Mock camera object
    """
    mock = MagicMock()
    mock.resolution = resolution
    mock.generate_synthetic_frame.return_value = np.zeros(
        (resolution[1], resolution[0], 3), dtype=np.uint8
    )
    mock.fov_horizontal = 60.0
    mock.fov_vertical = 45.0
    return mock


def create_mock_detector(detections: Optional[List[Dict]] = None) -> MagicMock:
    """
    Create a mock detector.
    
    Args:
        detections: List of detections to return, defaults to empty
        
    Returns:
        Mock detector object
    """
    mock = MagicMock()
    mock.detect.return_value = detections or []
    mock.model_loaded = True
    return mock


def create_mock_tracker(tracks: Optional[List] = None) -> MagicMock:
    """
    Create a mock tracker.
    
    Args:
        tracks: List of tracks to return, defaults to empty
        
    Returns:
        Mock tracker object
    """
    mock = MagicMock()
    mock.update.return_value = tracks or []
    mock.get_confirmed_tracks.return_value = tracks or []
    mock.tracks = {}
    mock.next_id = 1
    return mock


def create_mock_world(
    player_position: Optional[np.ndarray] = None,
    enemy_positions: Optional[List[np.ndarray]] = None
) -> MagicMock:
    """
    Create a mock World object.
    
    Args:
        player_position: Player UAV position
        enemy_positions: List of enemy positions
        
    Returns:
        Mock World object
    """
    mock = MagicMock()
    mock.world_size = (2000, 2000, 500)
    mock.time = 0.0
    mock.is_paused = False
    
    # Create mock UAVs
    uavs = {}
    
    # Player
    player_pos = player_position if player_position is not None else np.array([1000, 1000, 100])
    uavs['player'] = create_mock_uav_state(position=player_pos)
    
    # Enemies
    if enemy_positions:
        for i, pos in enumerate(enemy_positions):
            uavs[f'target_{i+1}'] = create_mock_enemy_state(
                id=f'target_{i+1}',
                position=pos
            )
    else:
        uavs['target_1'] = create_mock_enemy_state()
    
    mock.uavs = uavs
    mock.get_world_state.return_value = {'uavs': uavs, 'time': 0.0}
    
    return mock


# =============================================================================
# LOCK STATE MOCKS
# =============================================================================

def create_mock_lock_state(
    state: str = 'idle',
    target_id: Optional[int] = None,
    progress: float = 0.0,
    is_valid: bool = False
) -> Dict[str, Any]:
    """
    Create a mock lock state dict.
    
    Args:
        state: Lock state ('idle', 'locking', 'success')
        target_id: ID of locked target
        progress: Lock progress [0, 1]
        is_valid: Whether lock is geometrically valid
        
    Returns:
        Lock state dictionary
    """
    return {
        'state': state,
        'target_id': target_id,
        'progress': progress,
        'lock_time': progress * 4.0,  # 4 second lock requirement
        'is_valid': is_valid,
        'is_locked': target_id is not None,
        'dx': 0.0,
        'dy': 0.0,
        'coverage_w': 0.06,
        'coverage_h': 0.06,
        'size_ok': True,
        'center_ok': True,
        'reason_invalid': None,
        'score': {'total_score': 0, 'correct_locks': 0, 'incorrect_locks': 0}
    }


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

class MockPygame:
    """Context manager for mocking pygame in tests."""
    
    def __init__(self):
        self.patches = []
        
    def __enter__(self):
        self.patches.append(patch('pygame.init'))
        self.patches.append(patch('pygame.quit'))
        self.patches.append(patch('pygame.display.set_mode'))
        self.patches.append(patch('pygame.font.Font'))
        
        for p in self.patches:
            p.start()
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.patches:
            p.stop()
        return False
