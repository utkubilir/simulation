"""
Hedef Takip Sistemi
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Track:
    """Tek bir hedef izi (Fallback IoU Tracker için)"""
    id: int
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    confidence: float
    class_name: str = 'uav'
    age: int = 0
    hits: int = 1
    misses: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    is_confirmed: bool = False
    world_id: str = None
    team: str = None
    world_pos: np.ndarray = None  # Estimated or Ground Truth 3D position


class IoUTracker:
    """Yedek (Fallback) Tracker - Basit IoU tabanlı"""
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.01)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        
    def update(self, detections: List[Dict], frame=None) -> List[Track]:
        """Tespitlerle izleri güncelle"""
        for track in self.tracks.values():
            # Disable velocity prediction for stability with noisy detectors
            # if np.linalg.norm(track.velocity) > 0:
            #     dx, dy = track.velocity
            #     x1, y1, x2, y2 = track.bbox
            #     track.bbox = (x1+dx, y1+dy, x2+dx, y2+dy)
            #     track.center = (track.center[0]+dx, track.center[1]+dy)
            track.age += 1
            
        if not detections:
            for track in self.tracks.values():
                track.misses += 1
            self._remove_dead_tracks()
            return list(self.tracks.values())
            
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        for track_id, det_idx in matched:
            det = detections[det_idx]
            track = self.tracks[track_id]
            old_center = track.center
            track.bbox = det['bbox']
            track.center = det['center']
            track.confidence = det['confidence']
            track.hits += 1
            track.misses = 0
            track.velocity = np.array([det['center'][0]-old_center[0], det['center'][1]-old_center[1]])
            if 'world_id' in det: track.world_id = det['world_id']
            if 'team' in det: track.team = det['team']
            if 'world_pos' in det: track.world_pos = det['world_pos']
            if 'velocity' in det and np.any(det['velocity']): track.velocity = det['velocity'] # Use GT velocity if available
            if track.hits >= self.min_hits: track.is_confirmed = True
            
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])
            
        for track_id in unmatched_tracks:
            self.tracks[track_id].misses += 1
            
        self._remove_dead_tracks()
        return list(self.tracks.values())
        
    def _match_detections(self, detections):
        """Match detections to existing tracks using IoU.
        
        Uses vectorized IoU computation for O(n*m) -> O(1) per-element speedup.
        Greedy matching: iteratively assign highest IoU pairs above threshold.
        """
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        if not self.tracks or not detections:
            return matched, unmatched_dets, unmatched_tracks
        
        # Convert track bboxes to numpy array
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[tid].bbox for tid in track_ids])
        det_boxes = np.array([det['bbox'] for det in detections])
        
        # Vectorized IoU computation
        iou_matrix = self._compute_iou_vectorized(track_boxes, det_boxes)
        
        # Greedy matching
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
        """Compute IoU between all pairs of boxes using vectorized operations.
        
        Args:
            boxes1: (N, 4) array of boxes [x1, y1, x2, y2]
            boxes2: (M, 4) array of boxes [x1, y1, x2, y2]
            
        Returns:
            (N, M) IoU matrix
        """
        # Intersection coordinates
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        # Intersection area
        inter_width = np.maximum(0, x2 - x1)
        inter_height = np.maximum(0, y2 - y1)
        inter_area = inter_width * inter_height
        
        # Individual areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Union area
        union_area = area1[:, None] + area2[None, :] - inter_area
        
        # IoU with numerical stability
        return inter_area / np.maximum(union_area, 1e-6)
        
    def _compute_iou(self, b1, b2):
        """Compute IoU between two boxes (kept for backward compatibility)."""
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
        return inter / union if union > 0 else 0.0
        
    def _create_track(self, det):
        self.tracks[self.next_id] = Track(id=self.next_id, bbox=det['bbox'], center=det['center'],
            confidence=det['confidence'], world_id=det.get('world_id'), team=det.get('team'),
            world_pos=det.get('world_pos'), velocity=det.get('velocity', np.zeros(3)))
        self.next_id += 1
        
    def _remove_dead_tracks(self):
        dead = [tid for tid, t in self.tracks.items() if t.misses > self.max_age]
        for tid in dead: del self.tracks[tid]
        
    def get_confirmed_tracks(self): return [t for t in self.tracks.values() if t.is_confirmed]
    def get_track(self, tid): return self.tracks.get(tid)
    def reset(self): self.tracks.clear(); self.next_id = 1


# Tracker Seçimi (Kalman vs IoU)
try:
    from src.vision.kalman_tracker import KalmanTracker, FILTERPY_AVAILABLE
    if FILTERPY_AVAILABLE:
        # Kalman Filtresi mevcutsa kullan
        TargetTracker = KalmanTracker
        # logging.info("Vision: Kalman Tracking Active")
    else:
        # Fallback
        TargetTracker = IoUTracker
        # logging.warning("Vision: filterpy not found. Using basic IoU Tracker")
except ImportError:
    TargetTracker = IoUTracker
