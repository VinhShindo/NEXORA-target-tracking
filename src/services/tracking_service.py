"""
Tracking Service - ByteTrack nâng cao với Global Combined Cost Metric
FIXED: Giải quyết vấn đề mất ID khi đi ra/vào ở vị trí khác nhau.
Sử dụng: IoU + Center + Size + Feature Similarity trong cùng một vòng lặp Cost.
"""

import numpy as np
from typing import List, Dict, Optional
import logging
import cv2
import time

logger = logging.getLogger(__name__)


class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.A = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.05
        self.R = np.eye(2, dtype=np.float32) * 0.5
        self.P = np.eye(4, dtype=np.float32) * 100
        self.x = np.zeros((4, 1), dtype=np.float32)
    
    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x
    
    def update(self, z):
        self.z = np.array([[z[0]], [z[1]]], dtype=np.float32)
        y = self.z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x


class Track:
    def __init__(self, bbox, track_id, confidence=1.0, class_id=0):
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = confidence
        self.hits = 0
        self.lost = 0
        self.age = 1
        self.is_confirmed = False
        self.feature = None
        
        self.bbox = bbox
        self.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height
        
        self.kf = KalmanFilter()
        self.kf.x = np.array([[self.center[0]], [self.center[1]], [0], [0]], dtype=np.float32)
        self.kf.P = np.eye(4, dtype=np.float32) * 100
        self.velocity = [0.0, 0.0]
    
    def predict(self):
        self.kf.predict()
        x = self.kf.x.flatten()
        old_center = self.center.copy()
        self.center = [float(x[0]), float(x[1])]
        self.velocity = [self.center[0] - old_center[0], self.center[1] - old_center[1]]
        
        hw = self.width / 2
        hh = self.height / 2
        self.bbox = [self.center[0] - hw, self.center[1] - hh, self.center[0] + hw, self.center[1] + hh]
        self.lost += 1
        self.age += 1
    
    def update(self, bbox, confidence):
        self.bbox = bbox
        self.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height
        self.confidence = confidence
        self.hits += 1
        self.lost = 0
        self.kf.update(self.center)
        
        if self.hits >= 6:
            self.is_confirmed = True
    
    def is_valid(self):
        return self.is_confirmed and self.lost <= 30


class FeatureDatabase:
    # Giảm ngưỡng tương đồng xuống 0.55 để dễ dàng bắt lại ID khi đổi hướng/ánh sáng
    def __init__(self, max_age: float = 30.0, similarity_threshold: float = 0.55):
        self.features = {}
        self.max_age = max_age
        self.similarity_threshold = similarity_threshold
    
    def add_person(self, track_id: int, feature: np.ndarray, class_id: int = 0):
        if feature is None: return
        self.features[track_id] = {
            'feature': feature,
            'class_id': class_id,
            'last_seen': time.time()
        }
    
    def update_person(self, track_id: int, feature: np.ndarray, class_id: int = 0):
        if track_id in self.features and feature is not None:
            self.features[track_id]['feature'] = feature
            self.features[track_id]['last_seen'] = time.time()
    
    def find_match(self, feature: np.ndarray, class_id: int = 0) -> Optional[int]:
        if feature is None: return None
        best_match = None
        best_similarity = self.similarity_threshold
        
        for track_id, data in self.features.items():
            if data['class_id'] != class_id: continue
            if time.time() - data['last_seen'] > self.max_age: continue
            similarity = self._compute_similarity(feature, data['feature'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = track_id
        return best_match
    
    def _compute_similarity(self, f1, f2):
        try:
            return cv2.compareHist(f1.astype(np.float32), f2.astype(np.float32), cv2.HISTCMP_CORREL)
        except:
            return 0.0
    
    def clear_all(self):
        self.features.clear()
    
    def cleanup(self):
        expired = [tid for tid, data in self.features.items() if time.time() - data['last_seen'] > self.max_age]
        for tid in expired: del self.features[tid]


class ByteTrackOptimizedReID_Combined:
    """
    Tracker sử dụng cost metric kết hợp: IoU + Center Distance + Size Ratio + FEATURE.
    Đảm bảo tính liên tục cao cho ID, ngay cả khi di chuyển giữa các vùng khác nhau.
    """
    def __init__(self, track_thresh=0.5, high_thresh=0.6, match_thresh=0.6, track_buffer=30):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh  # Ngưỡng cost tối đa để coi là khớp
        self.track_buffer = track_buffer
        
        self.next_id = 1
        self.tracks: List[Track] = []
        self.feature_db = FeatureDatabase()
        logger.info("ByteTrackOptimizedReID_Combined initialized with Appearance Feature in Cost.")
    
    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.feature_db.clear_all()
    
    def _extract_feature(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        try:
            x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = min(x1, w-2), min(y1, h-2)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 <= x1 + 10 or y2 <= y1 + 10: return None
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: return None
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist = hist.flatten()
            if np.sum(hist) > 0: hist = hist / np.sum(hist)
            return hist
        except: return None
    
    def _compute_combined_cost(self, track, det_bbox, frame_size, det_feature=None):
        """
        Tính toán cost kết hợp: 50% Spatial (IoU/Dist/Size) + 50% Appearance (Feature Similarity).
        """
        # 1. Spatial Cost (IoU, Center Distance, Size Ratio)
        track_bbox = track.bbox
        x1_t, y1_t, x2_t, y2_t = track_bbox
        x1_d, y1_d, x2_d, y2_d = det_bbox
        
        inter_x1 = max(x1_t, x1_d)
        inter_y1 = max(y1_t, y1_d)
        inter_x2 = min(x2_t, x2_d)
        inter_y2 = min(y2_t, y2_d)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            iou = 0.0
        else:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area_t = (x2_t - x1_t) * (y2_t - y1_t)
            area_d = (x2_d - x1_d) * (y2_d - y1_d)
            union = area_t + area_d - inter_area
            iou = inter_area / union if union > 0 else 0.0
        
        center_t = ((x1_t + x2_t)/2, (y1_t + y2_t)/2)
        center_d = ((x1_d + x2_d)/2, (y1_d + y2_d)/2)
        dist = np.hypot(center_t[0] - center_d[0], center_t[1] - center_d[1])
        frame_diag = np.hypot(frame_size[0], frame_size[1])
        norm_dist = dist / frame_diag
        
        area_t = (x2_t - x1_t) * (y2_t - y1_t)
        area_d = (x2_d - x1_d) * (y2_d - y1_d)
        max_area = max(area_t, area_d)
        size_diff = abs(area_t - area_d) / max_area if max_area > 0 else 0.0
        
        # Spatial Cost = 0.5 IoU + 0.3 Dist + 0.2 Size
        spatial_cost = 0.5 * (1 - iou) + 0.3 * norm_dist + 0.2 * size_diff
        
        # 2. Appearance Cost (Feature)
        feature_sim = 0.0
        if track.feature is not None and det_feature is not None:
            # Tính độ tương tự màu sắc (0 đến 1)
            feature_sim = self.feature_db._compute_similarity(track.feature, det_feature)
        
        # Mặc định nếu không có feature, nó sẽ dựa hoàn toàn vào Spatial Cost
        if track.feature is None or det_feature is None:
            return spatial_cost
        
        # Kết hợp: 70% Spatial + 30% Appearance
        final_cost = 0.7 * spatial_cost + 0.3 * (1 - feature_sim)
        
        # Nếu khoảng cách xa quá mức (khác vùng hoàn toàn), vẫn chặn cứng bằng Spatial
        if norm_dist > 0.8: # Quá xa không thể match, kể cả giống nhau
            return 2.0
            
        return final_cost
    
    def _greedy_match(self, cost_matrix, threshold):
        matches = []
        if cost_matrix.size == 0: return matches
        pairs = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < threshold:
                    pairs.append((cost_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0]) # Cost càng nhỏ càng tốt
        
        matched_i = set()
        matched_j = set()
        for cost, i, j in pairs:
            if i not in matched_i and j not in matched_j:
                matches.append((i, j))
                matched_i.add(i)
                matched_j.add(j)
        return matches
    
    def update(self, detections: np.ndarray, frame=None) -> List[Track]:
        if detections is None or len(detections) == 0:
            for t in self.tracks: t.predict()
            self.tracks = [t for t in self.tracks if t.lost <= self.track_buffer]
            return self.tracks
        
        high_dets, low_dets = [], []
        for det in detections:
            if len(det) < 6: continue
            x1, y1, x2, y2, score, class_id = det[:6]
            if int(class_id) not in [0, 32]: continue
            if score >= self.high_thresh: high_dets.append([x1, y1, x2, y2, score, class_id])
            elif score >= self.track_thresh: low_dets.append([x1, y1, x2, y2, score, class_id])
        
        for t in self.tracks: t.predict()
        
        matched_track_ids = set()
        matched_det_indices = set()
        frame_size = (frame.shape[1], frame.shape[0]) if frame is not None else (640, 480)
        
        # === 1. Match High Dets sử dụng Combined Cost (Spatial + Feature) ===
        if len(self.tracks) > 0 and len(high_dets) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(high_dets)))
            # Pre-calc features để tiết kiệm thời gian
            det_features = [self._extract_feature(frame, d[:4]) if frame is not None else None for d in high_dets]
            
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(high_dets):
                    cost_matrix[i, j] = self._compute_combined_cost(t, d[:4], frame_size, det_features[j])
            
            matches = self._greedy_match(cost_matrix, self.match_thresh)
            for i, j in matches:
                det = high_dets[j]
                track = self.tracks[i]
                track.update(det[:4], det[4])
                # Cập nhật feature cho track
                if frame is not None and track.class_id == 0:
                    feat = self._extract_feature(frame, track.bbox)
                    if feat is not None:
                        track.feature = feat
                        self.feature_db.update_person(track.track_id, feat, track.class_id)
                matched_track_ids.add(i)
                matched_det_indices.add(j)
        
        # === 2. Match Low Dets (Chỉ dùng Spatial) ===
        if len(self.tracks) > 0 and len(low_dets) > 0:
            unmatched_tracks = [(i, t) for i, t in enumerate(self.tracks) if i not in matched_track_ids]
            if unmatched_tracks:
                cost_matrix = np.zeros((len(unmatched_tracks), len(low_dets)))
                for i, (idx, t) in enumerate(unmatched_tracks):
                    for j, d in enumerate(low_dets):
                        # Low dets không tính Feature để tiết kiệm CPU
                        cost_matrix[i, j] = self._compute_combined_cost(t, d[:4], frame_size, None)
                matches = self._greedy_match(cost_matrix, 0.55)
                for i, j in matches:
                    det = low_dets[j]
                    track = unmatched_tracks[i][1]
                    track.update(det[:4], det[4])
                    matched_track_ids.add(unmatched_tracks[i][0])
                    matched_det_indices.add(j + len(high_dets))
        
        # === 3. Re-ID cho các track đã mất hẳn và chưa được match ===
        unmatched_high = [(j, d) for j, d in enumerate(high_dets) if j not in matched_det_indices]
        if frame is not None:
            for j, det in unmatched_high:
                class_id = int(det[5]) if len(det) > 5 else 0
                if class_id == 0:
                    feature = self._extract_feature(frame, det[:4])
                    if feature is not None:
                        match_id = self.feature_db.find_match(feature, class_id)
                        if match_id is not None:
                            # Tìm track đã bị xóa hoặc chưa match
                            found_track = None
                            for t in self.tracks:
                                if t.track_id == match_id:
                                    found_track = t
                                    break
                            
                            if found_track is not None:
                                # Track bị treo, hồi sinh
                                found_track.update(det[:4], det[4])
                                found_track.feature = feature
                                self.feature_db.update_person(match_id, feature, class_id)
                                matched_det_indices.add(j)
                                logger.info(f"Re-animated lost Track ID {match_id}")
                                continue
                            else:
                                # Track đã bị xóa hoàn toàn khỏi self.tracks (do quá lâu), hồi sinh bằng ID cũ
                                new_track = Track(det[:4], match_id, det[4], class_id)
                                new_track.feature = feature
                                new_track.is_confirmed = True  # Đã từng confirmed nên không cần chờ nữa
                                self.tracks.append(new_track)
                                self.feature_db.update_person(match_id, feature, class_id)
                                matched_det_indices.add(j)
                                logger.info(f"Re-created Track ID {match_id} from database")
                                continue
        
        # === 4. Tạo Track Mới (Nếu chưa khớp gì cả) ===
        for j, det in enumerate(high_dets):
            if j not in matched_det_indices:
                class_id = int(det[5]) if len(det) > 5 else 0
                new_track = Track(det[:4], self.next_id, det[4], class_id)
                self.next_id += 1
                if frame is not None and class_id == 0:
                    feature = self._extract_feature(frame, det[:4])
                    if feature is not None:
                        new_track.feature = feature
                        self.feature_db.add_person(new_track.track_id, feature, class_id)
                self.tracks.append(new_track)
                logger.info(f"Created NEW Track ID {new_track.track_id}")
        
        self.tracks = [t for t in self.tracks if t.lost <= self.track_buffer]
        return self.tracks
    
    def _compute_iou(self, bbox1, bbox2):
        x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        x2, y2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1: return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
        return inter / (area1 + area2 - inter)


class TrackingService:
    def __init__(self):
        self.tracker = None
        self.tracks: List[Dict] = []
        self.track_history = {}
        self.locked_target_id = None
        self.locked_lost_count = 0
        self.max_locked_lost = 30
        self._init_tracker()
    
    def _init_tracker(self):
        self.tracker = ByteTrackOptimizedReID_Combined(
            track_thresh=0.4, high_thresh=0.6, match_thresh=0.6, track_buffer=30
        )
    
    def reset(self):
        self.tracks = []
        self.track_history = {}
        self.locked_target_id = None
        self.locked_lost_count = 0
        if self.tracker:
            self.tracker.reset()
    
    def lock_target(self, track_id: int, frame: np.ndarray = None):
        self.locked_target_id = track_id
        self.locked_lost_count = 0
    
    def unlock_target(self):
        self.locked_target_id = None
        self.locked_lost_count = 0
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        if self.tracker is None:
            return []
        if not detections:
            tracks = self.tracker.update(np.array([]), frame)
            self.tracks = self._convert_tracks(tracks)
            return self.tracks
        
        dets = []
        for d in detections:
            class_id = d.get('class_id', 0)
            if class_id not in [0, 32]:
                continue
            x1, y1, x2, y2 = d['bbox']
            dets.append([float(x1), float(y1), float(x2), float(y2), float(d['confidence']), int(class_id)])
        
        if not dets:
            tracks = self.tracker.update(np.array([]), frame)
            self.tracks = self._convert_tracks(tracks)
            return self.tracks
        
        tracks = self.tracker.update(np.array(dets), frame)
        self.tracks = self._convert_tracks(tracks)
        
        if self.locked_target_id is not None:
            found = False
            for t in self.tracks:
                if t['track_id'] == self.locked_target_id:
                    found = True
                    self.locked_lost_count = 0
                    break
            if not found:
                self.locked_lost_count += 1
                if self.locked_lost_count > self.max_locked_lost:
                    self.unlock_target()
        
        return self.tracks
    
    def _convert_tracks(self, tracks):
        result = []
        for t in tracks:
            if not t.is_valid():
                continue
            d = {
                'track_id': t.track_id,
                'bbox': t.bbox,
                'center': t.center,
                'confidence': t.confidence,
                'class_id': t.class_id,
                'class': 'person' if t.class_id == 0 else 'ball',
                'lost': t.lost,
                'hits': t.hits,
                'age': t.age,
                'velocity': t.velocity
            }
            result.append(d)
            self.track_history[t.track_id] = d.copy()
        return result
    
    def predict(self, frame: np.ndarray) -> List[Dict]:
        filtered = []
        for t in self.tracks:
            if t.get('lost', 0) < 4:
                filtered.append(t)
        return filtered
    
    def get_track_by_id(self, track_id):
        for t in self.tracks:
            if t['track_id'] == track_id:
                return t
        return None
    
    def get_tracks(self):
        return self.tracks
    
    def is_target_locked(self):
        return self.locked_target_id is not None
    
    def get_locked_target_id(self):
        return self.locked_target_id