# Test Suite Documentation

## 📁 Test Structure

```
tests/
├── conftest.py         # Shared fixtures and pytest configuration
├── helpers.py          # Utility functions for tests
├── mocks.py            # Mock objects for isolated testing
│
├── unit/               # Fast unit tests (< 1s each)
│   ├── test_autopilot.py
│   ├── test_combat.py
│   ├── test_fixed_wing.py
│   ├── test_kalman_tracker.py
│   ├── test_launcher.py
│   ├── test_parametrized.py
│   └── test_tracker.py
│
├── integration/        # Integration tests (1-10s each)
│   ├── test_controls.py
│   ├── test_smoke.py
│   ├── test_ui_audit.py
│   └── test_vnext_integration.py
│
├── compliance/         # Rules compliance tests
│   ├── test_contracts.py
│   ├── test_rubric.py
│   └── test_structure.py
│
├── regression/         # Regression tests (10-60s each)
│   ├── test_default_regression.py
│   ├── test_determinism.py
│   └── legacy/
│
└── performance/        # Performance benchmarks
    └── test_benchmarks.py
```

## 🚀 Running Tests

### All Tests
```bash
# Standard run
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Generate coverage report only
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### By Category
```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Performance benchmarks
pytest tests/performance/ -v -m performance

# Regression tests
pytest tests/regression/ -v -m regression

# Compliance tests
pytest tests/compliance/ -v
```

### Filtering
```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Only parametrized tests
pytest tests/unit/test_parametrized.py -v

# Run specific test class
pytest tests/unit/test_autopilot.py::TestPIDController -v

# Run specific test function
pytest tests/unit/test_autopilot.py::TestPIDController::test_pid_proportional -v

# Run tests matching pattern
pytest tests/ -v -k "lock"
```

### Debugging
```bash
# Stop on first failure
pytest tests/ -v -x

# Show print statements
pytest tests/ -v -s

# Run with pdb on failure
pytest tests/ -v --pdb

# Verbose output with local variables
pytest tests/ -v --tb=long
```

## 📊 Available Markers

| Marker | Description | Usage |
|--------|-------------|-------|
| `unit` | Fast unit tests | Auto-applied to `tests/unit/` |
| `integration` | Integration tests | Auto-applied to `tests/integration/` |
| `compliance` | Compliance tests | Auto-applied to `tests/compliance/` |
| `regression` | Regression tests | Auto-applied to `tests/regression/` |
| `performance` | Performance benchmarks | `@pytest.mark.performance` |
| `slow` | Slow-running tests | `@pytest.mark.slow` |

## 🔧 Available Fixtures

### Environment
| Fixture | Description |
|---------|-------------|
| `temp_output_dir` | Temporary directory that cleans up after test |
| `pygame_init` | Initialize pygame for UI tests |

### UAV
| Fixture | Description |
|---------|-------------|
| `sample_uav` | Default FixedWingUAV instance |
| `sample_uav_state` | UAV state dictionary |
| `sample_controls` | Control inputs dictionary |

### Detection & Tracking
| Fixture | Description |
|---------|-------------|
| `sample_detection` | Detection dictionary |
| `sample_track` | Track object |
| `tracker` | TargetTracker instance |
| `kalman_tracker` | KalmanTracker instance |

### Lock-On
| Fixture | Description |
|---------|-------------|
| `lock_config` | LockConfig instance |
| `lock_sm` | LockOnStateMachine instance |
| `geometry_validator` | GeometryValidator instance |

### Autopilot
| Fixture | Description |
|---------|-------------|
| `autopilot` | Autopilot instance |
| `pid_controller` | PIDController instance |

### Simulation
| Fixture | Description |
|---------|-------------|
| `simulation_config` | SimulationConfig instance |
| `headless_runner` | SimulationRunner in headless mode |
| `scenarios_dir` | Path to scenarios directory |
| `easy_lock_config` | Easy lock scenario config |

## 📝 Writing New Tests

### Unit Test Template
```python
import pytest
from src.module import Class

class TestClassName:
    """Test description."""
    
    def test_basic_functionality(self, fixture_name):
        """Test that basic functionality works."""
        result = Class().method()
        assert result == expected
        
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_parametrized(self, input, expected):
        """Test with multiple inputs."""
        assert Class().double(input) == expected
```

### Using Mocks
```python
from tests.mocks import create_mock_uav_state, create_mock_track

def test_with_mocks():
    uav_state = create_mock_uav_state(heading=45.0)
    track = create_mock_track(confidence=0.9)
    
    # Test with mock objects
    result = function_under_test(uav_state, [track])
    assert result.is_valid
```

### Performance Test Template
```python
import pytest
import time

class TestPerformance:
    
    @pytest.mark.performance
    def test_operation_speed(self):
        """Operation should complete in < X ms."""
        # Warmup
        for _ in range(100):
            operation()
        
        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / iterations) * 1000
        assert avg_ms < 1.0, f"Too slow: {avg_ms:.3f}ms"
```

## 🎯 Coverage Goals

| Metric | Current | Target |
|--------|---------|--------|
| Line Coverage | TBD | > 80% |
| Branch Coverage | TBD | > 70% |
| Test Count | 133 | > 200 |

## ⚠️ Known Issues

1. **XFAIL Tests**: `test_easy_lock_achieves_success` and `test_easy_lock_reliably_locks` are marked as expected failures due to scenario tuning issues.

2. **Skipped Tests**: Rubric tests are skipped until rubric calculation is integrated into CLI output.

3. **UI Tests**: Require `SDL_VIDEODRIVER=dummy` environment variable for headless execution.
