"""
Detection Noise Model - Phase 2

Adds configurable noise to detections for realistic simulation:
- Bounding box jitter (Gaussian)
- Confidence jitter (Gaussian)
- False negatives (drop with probability)
- False positives (inject spurious detections)

All noise is deterministic when using a seeded RNG.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class NoiseConfig:
    """Configuration for detection noise model."""
    # Bounding box noise (Gaussian std in pixels)
    bbox_sigma_px: float = 0.0
    
    # Confidence noise (Gaussian std, result clamped to [0,1])
    conf_sigma: float = 0.0
    
    # False negative probability (drop detection)
    p_fn: float = 0.0
    
    # False positive rate (per frame, not per second)
    p_fp: float = 0.0
    
    # Image bounds for false positives
    frame_width: int = 640
    frame_height: int = 480
    
    # Min/max size for false positive bboxes
    fp_min_size: int = 20
    fp_max_size: int = 80
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'NoiseConfig':
        """Create NoiseConfig from dictionary (scenario YAML)."""
        return cls(
            bbox_sigma_px=d.get('bbox_sigma_px', 0.0),
            conf_sigma=d.get('conf_sigma', 0.0),
            p_fn=d.get('p_fn', 0.0),
            p_fp=d.get('p_fp', 0.0),
            frame_width=d.get('frame_width', 640),
            frame_height=d.get('frame_height', 480),
            fp_min_size=d.get('fp_min_size', 20),
            fp_max_size=d.get('fp_max_size', 80),
        )
    
    def to_dict(self) -> Dict:
        """Export to dict for logging."""
        return {
            'bbox_sigma_px': self.bbox_sigma_px,
            'conf_sigma': self.conf_sigma,
            'p_fn': self.p_fn,
            'p_fp': self.p_fp,
        }


class DetectionNoiseModel:
    """
    Applies configurable noise to detection outputs.
    
    Must be initialized with a seeded RNG for determinism.
    """
    
    def __init__(self, config: NoiseConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        
    def apply(self, detections: List[Dict], frame_meta: Optional[Dict] = None) -> List[Dict]:
        """
        Apply noise to detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', etc.
            frame_meta: Optional frame metadata (unused currently)
            
        Returns:
            Noisy detections list (may be shorter or longer than input)
        """
        if not detections and self.config.p_fp == 0:
            return []
            
        noisy = []
        
        # Process existing detections
        for det in detections:
            # False negative check
            if self.config.p_fn > 0 and self.rng.random() < self.config.p_fn:
                continue  # Drop this detection
                
            # Copy detection
            noisy_det = det.copy()
            
            # Bbox jitter
            if self.config.bbox_sigma_px > 0:
                noisy_det = self._apply_bbox_noise(noisy_det)
                
            # Confidence jitter
            if self.config.conf_sigma > 0:
                noisy_det = self._apply_conf_noise(noisy_det)
                
            noisy.append(noisy_det)
            
        # False positives
        if self.config.p_fp > 0:
            fp_detections = self._generate_false_positives()
            noisy.extend(fp_detections)
            
        return noisy
    
    def _apply_bbox_noise(self, det: Dict) -> Dict:
        """Apply Gaussian noise to bounding box."""
        sigma = self.config.bbox_sigma_px
        
        if 'bbox' in det:
            x1, y1, x2, y2 = det['bbox']
            
            # Add noise to each corner
            x1 += self.rng.normal(0, sigma)
            y1 += self.rng.normal(0, sigma)
            x2 += self.rng.normal(0, sigma)
            y2 += self.rng.normal(0, sigma)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, self.config.frame_width - 1))
            y1 = max(0, min(y1, self.config.frame_height - 1))
            x2 = max(0, min(x2, self.config.frame_width - 1))
            y2 = max(0, min(y2, self.config.frame_height - 1))
            
            # Ensure valid box (x1 < x2, y1 < y2) - use midpoint if invalid
            if x1 >= x2:
                mid = (x1 + x2) / 2
                x1, x2 = mid - 1, mid + 1
            if y1 >= y2:
                mid = (y1 + y2) / 2
                y1, y2 = mid - 1, mid + 1
                
            det['bbox'] = (x1, y1, x2, y2)
            
            # Update center and dimensions
            det['center'] = ((x1 + x2) / 2, (y1 + y2) / 2)
            det['width'] = x2 - x1
            det['height'] = y2 - y1
            
        return det
    
    def _apply_conf_noise(self, det: Dict) -> Dict:
        """Apply Gaussian noise to confidence score."""
        if 'confidence' in det:
            conf = det['confidence']
            conf += self.rng.normal(0, self.config.conf_sigma)
            det['confidence'] = float(np.clip(conf, 0.0, 1.0))
        return det
    
    def _generate_false_positives(self) -> List[Dict]:
        """Generate false positive detections."""
        fps = []
        
        # Bernoulli trial for this frame
        if self.rng.random() < self.config.p_fp:
            # Generate one false positive
            w = self.rng.integers(self.config.fp_min_size, self.config.fp_max_size + 1)
            h = self.rng.integers(self.config.fp_min_size, self.config.fp_max_size + 1)
            
            # Random position within bounds
            x1 = self.rng.integers(0, max(1, self.config.frame_width - w))
            y1 = self.rng.integers(0, max(1, self.config.frame_height - h))
            x2 = x1 + w
            y2 = y1 + h
            
            # Random confidence (typically lower for FPs)
            conf = self.rng.uniform(0.3, 0.7)
            
            fps.append({
                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                'confidence': conf,
                'class': 'uav',
                'class_id': 0,
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'width': float(w),
                'height': float(h),
                'is_false_positive': True,  # Mark for debugging
            })
            
        return fps
    
    def get_config_dict(self) -> Dict:
        """Get config as dict for logging."""
        return self.config.to_dict()
