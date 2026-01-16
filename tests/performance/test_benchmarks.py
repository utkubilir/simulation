"""
Performance Benchmark Tests

Tests that measure and validate performance characteristics of simulation components.
Run with: pytest tests/performance/ -v -m performance
"""

import pytest
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPhysicsPerformance:
    """Performance tests for physics engine."""
    
    @pytest.mark.performance
    def test_physics_step_performance(self, sample_uav, sample_controls):
        """Physics update should complete in < 1ms average."""
        uav = sample_uav
        
        # Set controls on UAV
        uav.controls.throttle = sample_controls['throttle']
        uav.controls.aileron = sample_controls['aileron']
        uav.controls.elevator = sample_controls['elevator']
        uav.controls.rudder = sample_controls['rudder']
        
        # Warmup
        for _ in range(100):
            uav.update(dt=0.016)
        
        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            uav.update(dt=0.016)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        
        print(f"\n📊 Physics step: {avg_time_ms:.4f}ms avg")
        assert avg_time_ms < 1.0, f"Physics step too slow: {avg_time_ms:.3f}ms > 1.0ms"
        
    @pytest.mark.performance
    def test_physics_1000_steps_under_1_second(self, sample_uav, sample_controls):
        """1000 physics steps should complete in under 1 second."""
        uav = sample_uav
        
        # Set controls on UAV
        uav.controls.throttle = sample_controls['throttle']
        uav.controls.aileron = sample_controls['aileron']
        uav.controls.elevator = sample_controls['elevator']
        uav.controls.rudder = sample_controls['rudder']
        
        start = time.perf_counter()
        for _ in range(1000):
            uav.update(dt=0.016)
        elapsed = time.perf_counter() - start
        
        print(f"\n📊 1000 physics steps: {elapsed:.3f}s")
        assert elapsed < 1.0, f"1000 steps took {elapsed:.3f}s, expected < 1s"


class TestTrackerPerformance:
    """Performance tests for tracking system."""
    
    @pytest.mark.performance
    def test_tracker_update_performance(self, tracker):
        """Tracker update with 10 detections should be < 5ms."""
        detections = [
            {
                'bbox': (100 + i * 50, 100, 150 + i * 50, 150),
                'confidence': 0.8,
                'center': (125 + i * 50, 125)
            }
            for i in range(10)
        ]
        
        # Warmup
        for _ in range(100):
            tracker.update(detections)
        
        # Benchmark
        iterations = 500
        start = time.perf_counter()
        for _ in range(iterations):
            tracker.update(detections)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        
        print(f"\n📊 Tracker update (10 dets): {avg_time_ms:.4f}ms avg")
        assert avg_time_ms < 5.0, f"Tracker update too slow: {avg_time_ms:.3f}ms > 5ms"
        
    @pytest.mark.performance
    def test_tracker_empty_detections(self, tracker):
        """Tracker with no detections should be very fast."""
        # Warmup
        for _ in range(100):
            tracker.update([])
        
        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            tracker.update([])
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        
        print(f"\n📊 Tracker update (0 dets): {avg_time_ms:.4f}ms avg")
        assert avg_time_ms < 0.5, f"Empty tracker update too slow: {avg_time_ms:.3f}ms"


class TestLockOnPerformance:
    """Performance tests for lock-on system."""
    
    @pytest.mark.performance
    def test_lock_sm_update_performance(self, lock_sm, sample_track):
        """Lock state machine update should be < 1ms."""
        tracks = [sample_track]
        
        # Warmup
        for i in range(100):
            lock_sm.update(tracks, sim_time=i * 0.016, dt=0.016)
        
        # Reset for benchmark
        lock_sm.reset()
        
        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for i in range(iterations):
            lock_sm.update(tracks, sim_time=i * 0.016, dt=0.016)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        
        print(f"\n📊 Lock SM update: {avg_time_ms:.4f}ms avg")
        assert avg_time_ms < 1.0, f"Lock SM update too slow: {avg_time_ms:.3f}ms"


class TestAutopilotPerformance:
    """Performance tests for autopilot."""
    
    @pytest.mark.performance
    def test_autopilot_update_performance(self, autopilot, sample_uav_state):
        """Autopilot update should be < 0.5ms."""
        from src.uav.autopilot import AutopilotMode
        
        autopilot.enable()
        autopilot.set_mode(AutopilotMode.ALTITUDE_HOLD)
        autopilot.set_target_altitude(100.0)
        
        uav_state = sample_uav_state
        
        # Warmup
        for _ in range(100):
            autopilot.update(uav_state, dt=0.016)
        
        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            autopilot.update(uav_state, dt=0.016)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        
        print(f"\n📊 Autopilot update: {avg_time_ms:.4f}ms avg")
        assert avg_time_ms < 0.5, f"Autopilot update too slow: {avg_time_ms:.3f}ms"


class TestIntegratedPerformance:
    """Performance tests for integrated simulation."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_full_simulation_fps(self, temp_output_dir):
        """Full simulation should maintain > 60 FPS equivalent."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        
        config = SimulationConfig(seed=42, duration=5.0)
        sim = SimulationCore(config)
        
        # Warmup
        for _ in range(60):
            sim.step()
        
        # Benchmark
        frame_times = []
        for _ in range(300):  # 5 seconds at 60fps
            start = time.perf_counter()
            sim.step()
            frame_times.append(time.perf_counter() - start)
        
        avg_ms = np.mean(frame_times) * 1000
        p50_ms = np.percentile(frame_times, 50) * 1000
        p95_ms = np.percentile(frame_times, 95) * 1000
        p99_ms = np.percentile(frame_times, 99) * 1000
        
        fps_equivalent = 1000 / avg_ms
        
        print(f"\n📊 Frame latency statistics:")
        print(f"   Avg:  {avg_ms:.2f}ms")
        print(f"   P50:  {p50_ms:.2f}ms")
        print(f"   P95:  {p95_ms:.2f}ms")
        print(f"   P99:  {p99_ms:.2f}ms")
        print(f"   FPS:  {fps_equivalent:.1f}")
        
        # 60 FPS = 16.67ms per frame
        assert avg_ms < 16.67, f"Cannot maintain 60 FPS: avg {avg_ms:.2f}ms"
        assert p95_ms < 33.33, f"P95 too high for 30 FPS: {p95_ms:.2f}ms"


class TestMemoryPerformance:
    """Memory-related performance tests."""
    
    @pytest.mark.performance
    def test_tracker_memory_stability(self, tracker):
        """Tracker should not accumulate memory over time."""
        import gc
        
        # Run many updates
        for i in range(1000):
            detections = [
                {
                    'bbox': (100 + (i % 5) * 50, 100, 150 + (i % 5) * 50, 150),
                    'confidence': 0.8,
                    'center': (125 + (i % 5) * 50, 125)
                }
            ]
            tracker.update(detections)
        
        # Force garbage collection
        gc.collect()
        
        # Track count should be bounded
        track_count = len(tracker.tracks)
        print(f"\n📊 Tracks after 1000 updates: {track_count}")
        
        # Should not have more than max_age worth of tracks
        assert track_count < 100, f"Too many tracks accumulated: {track_count}"
