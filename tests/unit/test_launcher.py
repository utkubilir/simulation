
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock pygame before importing launcher
sys.modules['pygame'] = MagicMock()

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.ui.launcher import Launcher

class TestLauncher(unittest.TestCase):
    def setUp(self):
        self.launcher = Launcher()
        
    def test_initialization(self):
        # Verify basic attributes exist
        self.assertTrue(hasattr(self.launcher, 'widgets'))
        self.assertEqual(len(self.launcher.widgets), 5) # 4 fields + 1 button
        
    def test_scenario_loading(self):
        # Verify scenarios are loaded (dummy check since mocked fs might vary, but list shouldn't be empty)
        self.assertIsInstance(self.launcher.scenarios, list)
        self.assertGreaterEqual(len(self.launcher.scenarios), 1)

    def test_start_simulation(self):
        # Simulate logic
        self.launcher.dd_scenario.get_value = MagicMock(return_value="test_scenario")
        self.launcher.ti_seed.text = "123"
        self.launcher.ti_duration.text = "30.0"
        self.launcher.dd_mode.get_value = MagicMock(return_value="headless")
        
        self.launcher.start_simulation()
        
        expected = {
            'scenario': "test_scenario",
            'seed': 123,
            'duration': 30.0,
            'mode': "headless"
        }
        self.assertEqual(self.launcher.result, expected)
        self.assertFalse(self.launcher.running)

if __name__ == '__main__':
    unittest.main()
