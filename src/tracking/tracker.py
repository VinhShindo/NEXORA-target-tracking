"""
ByteTrack Tracker Implementation
"""
import numpy as np
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
from typing import List, Dict, Optional
import time

class TrackState:
    """Track state constants"""
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2

class Track:
    def __init__(self, detection, track_id: int, max_lost: int = 10):
        self.track_id = track_id
        self.bbox = detection['bbox']
        self.class_id = detection['class_id']
        self.confidence = detection['confidence']
        self.state = TrackState.TENTATIVE
        self.max_lost = max_lost
        self.lost_count = 0
        self.history = []
        self.last_update = time.time()
        
        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self._init_kalman_filter()
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter parameters"""
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        self.kf.R = np.eye(4) * 0.1
        self.kf.P = np.eye(8) * 100
        self.kf.Q = np.eye(8) * 0.01
    
    def update(self, detection: dict):
        """Update track with new detection"""
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.lost_count = 0
        self.last_update = time.time()
        self.history.append(self.bbox)
        
        # Update Kalman filter
        measurement = np.array([
            (self.bbox[0] + self.bbox[2]) / 2,  # center x
            (self.bbox[1] + self.bbox[3]) / 2,  # center y
            self.bbox[2] - self.bbox[0],         # width
            self.bbox[3] - self.bbox[1]          # height
        ])
        self.kf.update(measurement)
    
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        
        # Get predicted state
        prediction = self.kf.x[:4]
        x, y, w, h = prediction
        
        # Convert to bounding box
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        self.bbox = [int(x1), int(y1), int(x2), int(y2)]
        self.lost_count += 1

class ByteTracker:
    def __init__(self, max_lost: int = 10, iou_threshold: float = 0.3):
        """
        ByteTrack tracker implementation
        
        Args:
            max_lost: Maximum consecutive lost frames before track is deleted
            iou_threshold: IoU threshold for association
        """
        self.tracks = {}
        self.next_id = 0
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _associate(self, detections: List[dict]) -> dict:
        """Associate detections with existing tracks"""
        if not self.tracks or not detections:
            return {}
        
        # Build cost matrix based on IoU
        track_bboxes = [t.bbox for t in self.tracks.values()]
        det_bboxes = [d['bbox'] for d in detections]
        
        cost_matrix = np.zeros((len(track_bboxes), len(det_bboxes)))
        for i, t_bbox in enumerate(track_bboxes):
            for j, d_bbox in enumerate(det_bboxes):
                cost_matrix[i, j] = 1 - self._compute_iou(t_bbox, d_bbox)
        
        # Hungarian algorithm for assignment
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = {}
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < (1 - self.iou_threshold):
                track_id = list(self.tracks.keys())[i]
                matches[track_id] = detections[j]
        
        return matches
    
    def update(self, detections: List[dict]) -> List[dict]:
        """
        Update tracker with new detections
        
        Returns:
            List of track results: [{'bbox': [...], 'track_id': int, 'class_id': int}]
        """
        # Predict all tracks
        for track in self.tracks.values():
            track.predict()
        
        # Associate detections with tracks
        matches = self._associate(detections)
        
        # Update matched tracks
        matched_detections = []
        for track_id, detection in matches.items():
            self.tracks[track_id].update(detection)
            self.tracks[track_id].state = TrackState.CONFIRMED
            matched_detections.append(detection)
        
        # Create new tracks for unmatched detections
        unmatched_detections = [d for d in detections if d not in matched_detections]
        for detection in unmatched_detections:
            track = Track(detection, self.next_id, self.max_lost)
            self.tracks[self.next_id] = track
            self.next_id += 1
        
        # Remove lost tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.lost_count > track.max_lost:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        # Prepare results
        results = []
        for track in self.tracks.values():
            if track.state == TrackState.CONFIRMED:
                results.append({
                    'bbox': track.bbox,
                    'track_id': track.track_id,
                    'class_id': track.class_id,
                    'class_name': self._get_class_name(track.class_id)
                })
        
        return results
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        class_names = ['person', 'bottle', 'chair', 'ball', 'bag']
        return class_names[class_id] if class_id < len(class_names) else 'unknown'