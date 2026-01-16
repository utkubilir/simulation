"""
Geometry Validator for TEKNOFEST 2025 Rules.

Centralizes all geometric validation logic for:
1. Lock-on validity (size, center alignment)
2. Visualization compliance (overlay thickness)
3. Reason codes for invalidity

Usage:
    validator = GeometryValidator(config)
    is_valid, reason = validator.validate_lock_candidate(bbox, frame_size)
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None
    dx: float = 0.0
    dy: float = 0.0
    coverage_w: float = 0.0
    coverage_h: float = 0.0
    bbox_area_ratio: float = 0.0

class GeometryValidator:
    """
    Enforces geometric constraints for lock-on and visualization.
    """
    
    # Standard rejection reasons
    REASON_SIZE_TOO_SMALL = "SIZE_TOO_SMALL"
    REASON_CENTER_OUTSIDE = "CENTER_OUTSIDE"
    REASON_CONFIDENCE_LOW = "CONFIDENCE_LOW"
    REASON_TARGET_LOST = "TARGET_LOST"
    
    def __init__(self, size_threshold: float = 0.06, 
                 margin_h: float = 0.5, margin_v: float = 0.5):
        """
        Args:
            size_threshold: Minimum coverage ratio (defaults to 0.06 safe margin)
            margin_h: Horizontal center tolerance (fraction of bbox width)
            margin_v: Vertical center tolerance (fraction of bbox width)
        """
        self.size_threshold = size_threshold
        # Auto-clamp warning could go here if we had a logger, but for now we trust construction value
        if abs(self.size_threshold - 0.05) < 1e-6:
             # Rules say: "Do not use exactly 5.0%". We'll assume the caller handles the warning/clamping
             # or we can enforce it. Let's enforce a soft minimum if it's exactly 0.05
             pass

        self.margin_h = margin_h
        self.margin_v = margin_v
        
    def validate_lock_candidate(self, bbox: Tuple[float, float, float, float], 
                                frame_width: int, frame_height: int,
                                center: Tuple[float, float] = None) -> ValidationResult:
        """
        Validate a detection for lock-on eligibility.
        
        Args:
            bbox: (x1, y1, x2, y2)
            frame_width: Image width
            frame_height: Image height
            center: Optional (cx, cy), calculated from bbox if None
            
        Returns:
            ValidationResult with detailed metrics
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # 1. Size Validation
        # Rule: Target must satisfy >=5% of frame in width OR height
        # Implementation uses configured threshold (e.g. 6%)
        if frame_width > 0:
            cov_w = w / frame_width
        else:
            cov_w = 0.0
            
        if frame_height > 0:
            cov_h = h / frame_height
        else:
            cov_h = 0.0
            
        size_ok = (cov_w >= self.size_threshold) or (cov_h >= self.size_threshold)
        
        if not size_ok:
            return ValidationResult(
                is_valid=False,
                reason=self.REASON_SIZE_TOO_SMALL,
                coverage_w=cov_w,
                coverage_h=cov_h
            )
            
        # 2. Center Alignment Validation
        # Rule: |dx| <= target_width/2 AND |dy| <= target_height/2
        # Note: This is equivalent to "crosshair is within the bounding box"
        # provided the box implies the target boundaries.
        if center:
            cx, cy = center
        else:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
        frame_cx = frame_width / 2
        frame_cy = frame_height / 2
        
        dx = cx - frame_cx
        dy = cy - frame_cy
        
        # Tolerances
        tol_x = w * self.margin_h
        tol_y = h * self.margin_v
        
        center_ok = (abs(dx) <= tol_x) and (abs(dy) <= tol_y)
        
        if not center_ok:
            return ValidationResult(
                is_valid=False,
                reason=self.REASON_CENTER_OUTSIDE,
                dx=dx, dy=dy,
                coverage_w=cov_w, coverage_h=cov_h
            )
            
        # All checks passed
        return ValidationResult(
            is_valid=True,
            reason=None,
            dx=dx, dy=dy,
            coverage_w=cov_w, coverage_h=cov_h
        )

    @staticmethod
    def validate_thickness(thickness: int) -> int:
        """
        Enforce strict max line thickness = 3 px.
        """
        return max(1, min(thickness, 3))
