"""
Phase 2 Tests - Noise/Latency Determinism and Effect Verification.
"""

import hashlib
import tempfile
from pathlib import Path

import pytest


class TestPhase2Determinism:
    """Verify Phase 2 noise and latency are deterministic."""
    
    def test_noise_determinism(self):
        """Run noise_bbox_5px twice with same seed, verify identical hashes."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        from src.core.frame_logger import FrameLogger
        
        hashes = []
        for _ in range(2):
            config = SimulationConfig(
                seed=42,
                duration=3.0,
                scenario='noise_bbox_5px',
            )
            sim = SimulationCore(config)
            states = sim.run()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                tmp_path = Path(f.name)
                
            with FrameLogger(tmp_path) as logger:
                logger.log_frames(states)
                
            content = tmp_path.read_bytes()
            hashes.append(hashlib.sha256(content).hexdigest())
            tmp_path.unlink()
            
        assert hashes[0] == hashes[1], f"Noise scenario not deterministic: {hashes[0][:16]} != {hashes[1][:16]}"
        
    def test_latency_determinism(self):
        """Run latency_50ms twice with same seed, verify identical hashes."""
        from src.core.simulation_core import SimulationCore, SimulationConfig
        from src.core.frame_logger import FrameLogger
        
        hashes = []
        for _ in range(2):
            config = SimulationConfig(
                seed=42,
                duration=3.0,
                scenario='latency_50ms',
            )
            sim = SimulationCore(config)
            states = sim.run()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                tmp_path = Path(f.name)
                
            with FrameLogger(tmp_path) as logger:
                logger.log_frames(states)
                
            content = tmp_path.read_bytes()
            hashes.append(hashlib.sha256(content).hexdigest())
            tmp_path.unlink()
            
        assert hashes[0] == hashes[1], f"Latency scenario not deterministic: {hashes[0][:16]} != {hashes[1][:16]}"


class TestPhase2NoiseEffect:
    """Verify noise actually has an effect on outputs."""
    
    def test_noise_effect_is_present(self):
        """
        Run easy_lock and noise_bbox_10px with same seed.
        Assert metrics differ in at least one key (does NOT assert exact values).
        """
        from src.core.simulation_core import SimulationCore, SimulationConfig
        from src.core.metrics import MetricsCalculator
        
        # Run baseline (easy_lock)
        config_baseline = SimulationConfig(
            seed=42,
            duration=3.0,
            scenario='easy_lock',
        )
        sim_baseline = SimulationCore(config_baseline)
        states_baseline = sim_baseline.run()
        frames_baseline = [{'t': s.t, 'detections': s.detections, 'lock': s.lock, 'score': s.score} for s in states_baseline]
        
        # Run noisy scenario
        config_noisy = SimulationConfig(
            seed=42,
            duration=3.0,
            scenario='noise_bbox_10px',
        )
        sim_noisy = SimulationCore(config_noisy)
        states_noisy = sim_noisy.run()
        frames_noisy = [{'t': s.t, 'detections': s.detections, 'lock': s.lock, 'score': s.score} for s in states_noisy]
        
        # Compute basic comparison (not full metrics, just detection counts)
        det_count_baseline = sum(len(f.get('detections', [])) for f in frames_baseline)
        det_count_noisy = sum(len(f.get('detections', [])) for f in frames_noisy)
        
        # Noise should cause some difference (either more FP or fewer due to FN)
        # We don't assert exact values, just that something changed
        # Note: In some seeds, they might be equal, so we check detection bbox positions instead
        baseline_bboxes = []
        noisy_bboxes = []
        for f in frames_baseline:
            for d in f.get('detections', []):
                baseline_bboxes.append(d.get('bbox'))
        for f in frames_noisy:
            for d in f.get('detections', []):
                noisy_bboxes.append(d.get('bbox'))
                
        # At least one bbox should differ (due to jitter)
        assert baseline_bboxes != noisy_bboxes, "Noise had no effect on bounding boxes"
