"""
Kalman Filter Enhanced Tracker

Provides Kalman filter-based tracking for smoother position estimation
and better prediction during temporary occlusions.

Requires: filterpy (pip install filterpy)
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    KalmanFilter = None

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def create_kalman_filter() -> 'KalmanFilter':
    """
    Create a Kalman filter for tracking bounding box center and size.
    
    State vector: [x, y, s, r, vx, vy, vs]
        x, y: center coordinates
        s: scale (area)
        r: aspect ratio (constant)
        vx, vy: velocity
        vs: scale velocity
        
    Measurement vector: [x, y, s, r]
    """
    if not FILTERPY_AVAILABLE:
        raise ImportError("filterpy is required for Kalman filtering. Install with: pip install filterpy")
    
    kf = KalmanFilter(dim_x=7, dim_z=4)
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([
        [1, 0, 0, 0, 1, 0, 0],  # x = x + vx
        [0, 1, 0, 0, 0, 1, 0],  # y = y + vy
        [0, 0, 1, 0, 0, 0, 1],  # s = s + vs
        [0, 0, 0, 1, 0, 0, 0],  # r = r (constant)
        [0, 0, 0, 0, 1, 0, 0],  # vx = vx
        [0, 0, 0, 0, 0, 1, 0],  # vy = vy
        [0, 0, 0, 0, 0, 0, 1],  # vs = vs
    ])
    
    # Measurement matrix
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ])
    
    # Measurement noise covariance
    kf.R = np.diag([1, 1, 10, 10])
    
    # Process noise covariance
    kf.Q = np.eye(7)
    kf.Q[-1, -1] *= 0.01  # Scale velocity changes slowly
    kf.Q[4:, 4:] *= 0.01  # Velocities change slowly
    
    # Initial state covariance
    kf.P *= 10
    kf.P[4:, 4:] *= 1000  # High uncertainty in initial velocities
    
    return kf


def bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert bbox [x1, y1, x2, y2] to measurement [x, y, s, r]."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    s = w * h  # Scale (area)
    r = w / h if h > 0 else 1.0  # Aspect ratio
    return np.array([x, y, s, r])


def z_to_bbox(z: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert state [x, y, s, r, ...] back to bbox [x1, y1, x2, y2]."""
    x, y, s, r = z[:4]
    s = max(s, 1.0)  # Prevent negative/zero area
    r = max(r, 0.1)  # Prevent invalid aspect ratio
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 1.0
    return (x - w/2, y - h/2, x + w/2, y + h/2)


@dataclass
class KalmanTrack:
    """A track with Kalman filter state estimation."""
    id: int
    kf: 'KalmanFilter'
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    confidence: float
    class_name: str = 'uav'
    age: int = 0
    hits: int = 1
    misses: int = 0
    is_confirmed: bool = False
    world_id: str = None
    team: str = None
    histogram: Optional[np.ndarray] = None
    
    @property
    def velocity(self) -> np.ndarray:
        """Get velocity from Kalman state."""
        return self.kf.x[4:6].flatten()
    
    def predict(self):
        """Predict next state."""
        self.kf.predict()
        self.bbox = z_to_bbox(self.kf.x.flatten())
        self.center = (self.kf.x[0, 0], self.kf.x[1, 0])
        self.age += 1
        
    def update(self, bbox: Tuple[float, float, float, float], confidence: float, histogram: Optional[np.ndarray] = None):
        """Update with measurement."""
        z = bbox_to_z(bbox)
        self.kf.update(z)
        
        # Update Histogram (EMA)
        if histogram is not None:
            if self.histogram is None:
                self.histogram = histogram
            else:
                # Alpha = 0.5 (blend old and new)
                self.histogram = 0.5 * self.histogram + 0.5 * histogram
                cv2.normalize(self.histogram, self.histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        self.bbox = z_to_bbox(self.kf.x.flatten())
        self.center = (self.kf.x[0, 0], self.kf.x[1, 0])
        self.confidence = confidence
        self.hits += 1
        self.misses = 0


class KalmanTracker:
    """
    Multi-target tracker with Kalman filter state estimation.
    
    Provides smoother tracks and better handling of temporary occlusions
    compared to the basic IoU tracker.
    
    Requires: pip install filterpy
    """
    
    def __init__(self, config: dict = None):
        if not FILTERPY_AVAILABLE:
            raise ImportError(
                "filterpy is required for KalmanTracker. "
                "Install with: pip install filterpy"
            )
            
        config = config or {}
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        
        self.tracks: Dict[int, KalmanTrack] = {}
        self.next_id = 1
        
    def update(self, detections: List[Dict], frame=None) -> List[KalmanTrack]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', etc.
            frame: Optional frame (unused, for API compatibility)
            
        Returns:
            List of current tracks
        """
        # Predict existing tracks
        for track in self.tracks.values():
            track.predict()
            
        if not detections:
            for track in self.tracks.values():
                track.misses += 1
            self._remove_dead_tracks()
            return list(self.tracks.values())
            
        # Pre-compute histograms for appearance matching
        if frame is not None:
            for det in detections:
                det['histogram'] = compute_histogram(frame, det['bbox'])
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, det_idx in matched:
            det = detections[det_idx]
            track = self.tracks[track_id]
            track.update(det['bbox'], det['confidence'], det.get('histogram'))
            if 'world_id' in det:
                track.world_id = det['world_id']
            if 'team' in det:
                track.team = det['team']
            if track.hits >= self.min_hits:
                track.is_confirmed = True
                
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])
            
        # Increment misses for unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id].misses += 1
            
        self._remove_dead_tracks()
        return list(self.tracks.values())
    
    def _match_detections(self, detections):
        """Match detections to tracks using IoU.
        
        Uses Hungarian Algorithm (linear_sum_assignment) if scipy is available,
        otherwise falls back to Greedy matching.
        """
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        if not self.tracks or not detections:
            return matched, unmatched_dets, unmatched_tracks
            
        # Build IoU matrix using predicted track positions
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[tid].bbox for tid in track_ids])
        det_boxes = np.array([det['bbox'] for det in detections])
        
        iou_matrix = self._compute_iou_vectorized(track_boxes, det_boxes)
        
        if SCIPY_AVAILABLE:
            # Combine IoU and Appearance Cost
            iou_cost = 1.0 - iou_matrix
            
            # Appearance Cost
            appearance_cost = np.zeros_like(iou_cost)
            has_histogram = False
            
            for i, tid in enumerate(track_ids):
                track = self.tracks[tid]
                if track.histogram is None: continue
                
                for j, det in enumerate(detections):
                    if det.get('histogram') is None: continue
                    has_histogram = True
                    
                    # Correlation metric (1.0 = match, 0.0 = no match)
                    # Cost = 1 - correlation
                    score = cv2.compareHist(track.histogram, det['histogram'], cv2.HISTCMP_CORREL)
                    appearance_cost[i, j] = 1.0 - max(0, score)
            
            # Weighted Cost
            # If we have histograms, use 70% IoU + 30% Appearance
            final_cost = iou_cost
            if has_histogram:
                final_cost = 0.7 * iou_cost + 0.3 * appearance_cost
                
            # Gating: Set high cost for impossible matches
            final_cost[iou_matrix < self.iou_threshold] = 1000.0
            
            row_indices, col_indices = linear_sum_assignment(final_cost)
            
            for row, col in zip(row_indices, col_indices):
                if final_cost[row, col] < 1000.0:
                    matched.append((track_ids[row], col))
                    if col in unmatched_dets: unmatched_dets.remove(col)
                    if track_ids[row] in unmatched_tracks: unmatched_tracks.remove(track_ids[row])
        else:
            # Greedy matching (Fallback)

            while iou_matrix.size > 0 and iou_matrix.max() >= self.iou_threshold:
                idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                matched.append((track_ids[idx[0]], idx[1]))
                if idx[1] in unmatched_dets:
                    unmatched_dets.remove(idx[1])
                if track_ids[idx[0]] in unmatched_tracks:
                    unmatched_tracks.remove(track_ids[idx[0]])
                iou_matrix[idx[0], :] = 0
                iou_matrix[:, idx[1]] = 0
            
        return matched, unmatched_dets, unmatched_tracks
    
    def _compute_iou_vectorized(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1[:, None] + area2[None, :] - inter_area
        
        return inter_area / np.maximum(union_area, 1e-6)
        
    def _create_track(self, det: Dict):
        """Create a new Kalman track from detection."""
        kf = create_kalman_filter()
        
        # Initialize state from detection
        z = bbox_to_z(det['bbox'])
        kf.x[:4] = z.reshape(-1, 1)
        
        track = KalmanTrack(
            id=self.next_id,
            kf=kf,
            bbox=det['bbox'],
            center=det['center'],
            confidence=det['confidence'],
            world_id=det.get('world_id'),
            team=det.get('team'),
            histogram=det.get('histogram')
        )
        
        self.tracks[self.next_id] = track
        self.next_id += 1
        
    def _remove_dead_tracks(self):
        """Remove tracks that have been missing too long."""
        dead = [tid for tid, t in self.tracks.items() if t.misses > self.max_age]
        for tid in dead:
            del self.tracks[tid]
            
    def get_confirmed_tracks(self) -> List[KalmanTrack]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks.values() if t.is_confirmed]
    
    def get_track(self, tid: int) -> Optional[KalmanTrack]:
        """Get track by ID."""
        return self.tracks.get(tid)
    
    def reset(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.next_id = 1


def compute_histogram(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Compute HSV color histogram for appearance matching.
    Returns normalized histogram (32 bins for H, 32 for S).
    """
    if frame is None or bbox is None:
        return None
        
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    # Clip to frame
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
        
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Compute histogram (HS only, ignore V for lighting/shadow invariance)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    
    # Normalize
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return hist.flatten()
