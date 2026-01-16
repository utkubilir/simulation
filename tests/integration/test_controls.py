import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import unittest
from unittest.mock import MagicMock, patch
from scripts.run import SimulationRunner

class TestSimulationControls(unittest.TestCase):
    def setUp(self):
        self.config = {
            'simulation': {'window_width': 100, 'window_height': 100},
            'camera': {'width': 100, 'height': 100},
            'lock': {},
            'detection': {},
            'tracking': {}
        }
        self.runner = SimulationRunner(self.config, mode='headless') # Use headless to avoid pygame window
        self.runner.renderer = MagicMock() # Mock renderer for event handling
        
    def test_pause_resume(self):
        # Initial state
        self.assertFalse(self.runner.world.is_paused)
        
        # Simulate SPACE press (Pause)
        self.runner.renderer.handle_events.return_value = [{'type': 'keydown', 'key': 'space'}]
        self.runner._handle_events()
        self.assertTrue(self.runner.world.is_paused, "Simulation should be paused after SPACE")
        
        # Simulate SPACE press (Resume)
        self.runner.renderer.handle_events.return_value = [{'type': 'keydown', 'key': 'space'}]
        self.runner._handle_events()
        self.assertFalse(self.runner.world.is_paused, "Simulation should be resumed after SPACE")

    def test_restart(self):
        # Advance simulation
        self.runner.sim_time = 10.0
        self.runner.frame_id = 100
        
        # Simulate R press (Restart)
        self.runner.renderer.handle_events.return_value = [{'type': 'keydown', 'key': 'r'}]
        
        # Capture reset calls
        self.runner.world.reset = MagicMock()
        self.runner.lock_sm.reset = MagicMock()
        
        self.runner._handle_events()
        
        # Verify resets
        self.runner.world.reset.assert_called_once()
        self.runner.lock_sm.reset.assert_called_once()
        self.assertEqual(self.runner.sim_time, 0.0, "Sim time should reset to 0")
        self.assertEqual(self.runner.frame_id, 0, "Frame ID should reset to 0")
        self.assertFalse(self.runner.world.is_paused, "Simulation should be unpaused after restart")

if __name__ == '__main__':
    unittest.main()
