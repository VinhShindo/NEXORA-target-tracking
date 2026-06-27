"""
Tracking Service - ByteTrack với Feature Database để giữ ID
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import cv2
from collections import deque
import time

logger = logging.getLogger(__name__)


class KalmanFilter:
    """Kalman Filter đơn giản cho tracking"""
    
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
        self.z = np.zeros((2, 1), dtype=np.float32)
    
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


class PersonFeature:
    """Lưu đặc điểm của một người để nhận diện lại"""
    
    def __init__(self, track_id: int, feature: np.ndarray, bbox: List[float], class_id: int = 0):
        self.track_id = track_id
        self.feature = feature
        self.bbox = bbox
        self.class_id = class_id
        self.last_seen = time.time()
        self.hits = 1
        self.confidence = 1.0
    
    def update(self, feature: np.ndarray, bbox: List[float]):
        """Cập nhật feature mới"""
        # Làm mới feature (moving average)
        self.feature = 0.7 * self.feature + 0.3 * feature
        self.feature = self.feature / (np.sum(self.feature) + 1e-6)
        self.bbox = bbox
        self.last_seen = time.time()
        self.hits += 1
    
    def is_expired(self, max_age: float = 30.0):
        """Kiểm tra feature có còn hiệu lực không"""
        return time.time() - self.last_seen > max_age


class FeatureDatabase:
    """Database lưu đặc điểm của tất cả người đã gặp"""
    
    def __init__(self, max_age: float = 30.0, similarity_threshold: float = 0.4):
        self.features: Dict[int, PersonFeature] = {}
        self.max_age = max_age
        self.similarity_threshold = similarity_threshold
        self.next_db_id = 1  # ID trong database (độc lập với track_id)
    
    def add_person(self, track_id: int, feature: np.ndarray, bbox: List[float], class_id: int = 0):
        """Thêm người mới vào database"""
        if feature is None:
            return
        self.features[track_id] = PersonFeature(track_id, feature, bbox, class_id)
        logger.info(f"Added person {track_id} to feature database")
    
    def update_person(self, track_id: int, feature: np.ndarray, bbox: List[float]):
        """Cập nhật feature của người đã có"""
        if track_id in self.features and feature is not None:
            self.features[track_id].update(feature, bbox)
    
    def find_match(self, feature: np.ndarray, class_id: int = 0) -> Optional[int]:
        """Tìm người khớp với feature này"""
        if feature is None:
            return None
        
        best_match = None
        best_similarity = self.similarity_threshold
        
        for track_id, person in self.features.items():
            # Chỉ so sánh cùng class
            if person.class_id != class_id:
                continue
            
            # Bỏ qua nếu đã hết hạn
            if person.is_expired(self.max_age):
                continue
            
            similarity = self._compute_similarity(feature, person.feature)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = track_id
        
        if best_match is not None:
            logger.debug(f"Found match: {best_match} with similarity {best_similarity:.2f}")
        return best_match
    
    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Tính similarity giữa 2 features"""
        try:
            # Histogram correlation
            hist1 = feat1[:256]
            hist2 = feat2[:256]
            corr = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            
            # Position similarity (trọng số thấp hơn)
            pos_sim = 1.0 - min(1.0, np.linalg.norm(feat1[256:258] - feat2[256:258]) * 2)
            size_sim = 1.0 - min(1.0, abs(feat1[258] - feat2[258]) * 3)
            
            # Weighted combination
            similarity = 0.7 * max(0, corr) + 0.2 * max(0, pos_sim) + 0.1 * max(0, size_sim)
            return similarity
            
        except Exception:
            return 0.0
    
    def cleanup(self):
        """Xóa các feature đã hết hạn"""
        expired = [tid for tid, p in self.features.items() if p.is_expired(self.max_age)]
        for tid in expired:
            del self.features[tid]
            logger.debug(f"Removed expired person {tid}")
    
    def get_stats(self) -> Dict:
        return {
            'total_persons': len(self.features),
            'max_age': self.max_age
        }


class Track:
    """Track object với ID ổn định"""
    
    def __init__(self, bbox, track_id, confidence=1.0, class_id=0):
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = confidence
        self.hits = 1
        self.no_losses = 0
        self.lost = 0
        self.age = 1
        
        # Bounding box [x1, y1, x2, y2]
        self.bbox = bbox
        self.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        
        # Kalman Filter
        self.kf = KalmanFilter()
        self.kf.x = np.array([[self.center[0]], [self.center[1]], [0], [0]], dtype=np.float32)
        self.kf.P = np.eye(4, dtype=np.float32) * 100
        
        # History
        self.bbox_history = [bbox]
        self.max_history = 30
        
        self.is_activated = True
        self.feature = None
        self.velocity = [0.0, 0.0]
        
        # Feature đã được đăng ký trong database chưa
        self.registered_in_db = False
    
    def predict(self):
        """Dự đoán vị trí mới"""
        self.kf.predict()
        x = self.kf.x.flatten()
        old_center = self.center.copy()
        self.center = [float(x[0]), float(x[1])]
        
        # Tính vận tốc
        self.velocity = [
            self.center[0] - old_center[0],
            self.center[1] - old_center[1]
        ]
        
        # Cập nhật bbox từ center
        hw = self.width / 2
        hh = self.height / 2
        self.bbox = [
            self.center[0] - hw,
            self.center[1] - hh,
            self.center[0] + hw,
            self.center[1] + hh
        ]
        
        self.lost += 1
        self.age += 1
    
    def update(self, bbox, confidence):
        """Cập nhật track với detection mới"""
        # Cập nhật bbox
        self.bbox = bbox
        self.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.confidence = confidence
        self.hits += 1
        self.lost = 0
        self.no_losses = 0
        
        # Cập nhật Kalman
        self.kf.update(self.center)
        
        # Lưu history
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > self.max_history:
            self.bbox_history.pop(0)
    
    def get_bbox(self):
        return self.bbox
    
    def get_center(self):
        return self.center
    
    def is_valid(self):
        """Kiểm tra track còn hợp lệ không"""
        return self.lost <= 30
    
    def get_state(self):
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'center': self.center,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'lost': self.lost,
            'hits': self.hits,
            'age': self.age,
            'velocity': self.velocity
            # KHÔNG bao gồm 'L:' trong label
        }


class ByteTrackWithFeature:
    """ByteTrack với Feature Database để giữ ID"""
    
    def __init__(self, track_thresh=0.5, high_thresh=0.6, match_thresh=0.8, 
                 frame_rate=30, track_buffer=30, feature_max_age=30.0):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        
        self.frame_id = 0
        self.next_id = 1
        self.tracks: List[Track] = []
        
        # Feature Database để giữ ID
        self.feature_db = FeatureDatabase(max_age=feature_max_age, similarity_threshold=0.4)
        
        logger.info("ByteTrackWithFeature initialized")
    
    def reset(self):
        """Reset tracker - KHÔNG reset next_id"""
        self.frame_id = 0
        # KHÔNG reset next_id để tránh trùng ID
        self.tracks = []
        # KHÔNG xóa feature database để giữ ID cũ
        # self.feature_db.cleanup()  # Có thể cleanup nhưng không xóa hết
        logger.info(f"Tracker reset, next_id: {self.next_id}, DB size: {len(self.feature_db.features)}")
    
    def _extract_feature_from_frame(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Trích xuất feature từ frame"""
        try:
            x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = min(x1, w-2), min(y1, h-2)
            x2, y2 = min(x2, w), min(y2, h)
            
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                return None
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            
            # Color histogram
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-6)
            
            # Thêm thông tin vị trí tương đối (trọng số thấp)
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            size = (x2 - x1) * (y2 - y1) / (w * h)
            
            feature = np.concatenate([hist, [cx * 0.1, cy * 0.1, size * 0.1]])
            return feature
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None
    
    def update(self, detections: np.ndarray, frame=None) -> List[Track]:
        """Update tracker với detections mới"""
        self.frame_id += 1
        
        # Cleanup database mỗi 100 frame
        if self.frame_id % 100 == 0:
            self.feature_db.cleanup()
        
        if detections is None or len(detections) == 0:
            # Predict tất cả tracks
            for track in self.tracks:
                track.predict()
            # Xóa track đã lost quá lâu
            self.tracks = [t for t in self.tracks if t.lost <= self.track_buffer]
            return self.tracks
        
        # Chia detections thành high và low
        high_dets = []
        low_dets = []
        
        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, score, class_id = det[:6]
            else:
                continue
            
            # Chỉ xử lý person (class_id=0) và ball (class_id=32)
            if int(class_id) not in [0, 32]:
                continue
            
            if score >= self.high_thresh:
                high_dets.append([x1, y1, x2, y2, score, class_id])
            elif score >= self.track_thresh:
                low_dets.append([x1, y1, x2, y2, score, class_id])
        
        # Predict tracks
        for track in self.tracks:
            track.predict()
        
        # ===== MATCH HIGH DETECTIONS =====
        matched_track_ids = set()
        matched_det_indices = set()
        
        if len(self.tracks) > 0 and len(high_dets) > 0:
            # Tính IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(high_dets)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(high_dets):
                    iou = self._compute_iou(track.bbox, det[:4])
                    iou_matrix[i, j] = iou
            
            # Greedy matching
            matched_pairs = self._greedy_match(iou_matrix, self.match_thresh)
            
            for i, j in matched_pairs:
                det = high_dets[j]
                track = self.tracks[i]
                track.update(det[:4], det[4])
                matched_track_ids.add(i)
                matched_det_indices.add(j)
                
                # Cập nhật feature trong database
                if frame is not None and track.class_id == 0:  # Chỉ person
                    feature = self._extract_feature_from_frame(frame, track.bbox)
                    if feature is not None:
                        self.feature_db.update_person(track.track_id, feature, track.bbox)
        
        # ===== MATCH LOW DETECTIONS =====
        if len(self.tracks) > 0 and len(low_dets) > 0:
            unmatched_tracks = []
            for i, track in enumerate(self.tracks):
                if i not in matched_track_ids:
                    unmatched_tracks.append((i, track))
            
            if len(unmatched_tracks) > 0:
                iou_matrix = np.zeros((len(unmatched_tracks), len(low_dets)))
                for i, (idx, track) in enumerate(unmatched_tracks):
                    for j, det in enumerate(low_dets):
                        iou = self._compute_iou(track.bbox, det[:4])
                        iou_matrix[i, j] = iou
                
                matched_pairs = self._greedy_match(iou_matrix, 0.5)
                
                for i, j in matched_pairs:
                    det = low_dets[j]
                    track = unmatched_tracks[i][1]
                    track.update(det[:4], det[4])
                    matched_track_ids.add(unmatched_tracks[i][0])
                    matched_det_indices.add(j + len(high_dets))
        
        # ===== CHECK UNMATCHED DETECTIONS WITH FEATURE DATABASE =====
        unmatched_high_dets = []
        for j, det in enumerate(high_dets):
            if j not in matched_det_indices:
                unmatched_high_dets.append((j, det))
        
        # Thử match với feature database cho person
        if frame is not None:
            for j, det in unmatched_high_dets:
                class_id = int(det[5]) if len(det) > 5 else 0
                if class_id == 0:  # Chỉ person
                    feature = self._extract_feature_from_frame(frame, det[:4])
                    if feature is not None:
                        # Tìm trong database
                        matched_db_id = self.feature_db.find_match(feature, class_id)
                        if matched_db_id is not None:
                            # Tìm track có ID này trong danh sách
                            found_track = None
                            for track in self.tracks:
                                if track.track_id == matched_db_id:
                                    found_track = track
                                    break
                            
                            if found_track is not None:
                                # Cập nhật track với detection mới
                                found_track.update(det[:4], det[4])
                                matched_track_ids.add(self.tracks.index(found_track))
                                # Cập nhật feature
                                self.feature_db.update_person(matched_db_id, feature, det[:4])
                                logger.info(f"Re-matched person {matched_db_id} from database")
                                continue
                            else:
                                # Track đã bị xóa nhưng feature vẫn còn
                                # Tạo track mới với ID cũ
                                new_track = Track(det[:4], matched_db_id, det[4], class_id)
                                new_track.feature = feature
                                new_track.registered_in_db = True
                                self.tracks.append(new_track)
                                # Cập nhật feature trong database
                                self.feature_db.update_person(matched_db_id, feature, det[:4])
                                logger.info(f"Re-created person {matched_db_id} from database")
                                matched_track_ids.add(len(self.tracks) - 1)
                                continue
        
        # ===== CREATE NEW TRACKS =====
        for j, det in enumerate(high_dets):
            if j not in matched_det_indices:
                class_id = int(det[5]) if len(det) > 5 else 0
                
                # Kiểm tra lại với database một lần nữa
                should_create = True
                if frame is not None and class_id == 0:
                    feature = self._extract_feature_from_frame(frame, det[:4])
                    if feature is not None:
                        matched_db_id = self.feature_db.find_match(feature, class_id)
                        if matched_db_id is not None:
                            # Đã có trong database, tạo track với ID cũ
                            new_track = Track(det[:4], matched_db_id, det[4], class_id)
                            new_track.feature = feature
                            new_track.registered_in_db = True
                            self.tracks.append(new_track)
                            self.feature_db.update_person(matched_db_id, feature, det[:4])
                            logger.info(f"Created track with existing ID {matched_db_id}")
                            should_create = False
                
                if should_create:
                    # Tạo track mới với ID mới
                    new_track = Track(det[:4], self.next_id, det[4], class_id)
                    self.next_id += 1
                    
                    # Lưu feature vào database nếu là person
                    if frame is not None and class_id == 0:
                        feature = self._extract_feature_from_frame(frame, det[:4])
                        if feature is not None:
                            new_track.feature = feature
                            self.feature_db.add_person(new_track.track_id, feature, det[:4], class_id)
                            new_track.registered_in_db = True
                    
                    self.tracks.append(new_track)
        
        # ===== REMOVE LOST TRACKS =====
        self.tracks = [t for t in self.tracks if t.lost <= self.track_buffer]
        
        return self.tracks
    
    def _compute_iou(self, bbox1, bbox2):
        """Tính IoU giữa 2 bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _greedy_match(self, iou_matrix, threshold):
        """Greedy matching đơn giản"""
        matches = []
        
        if iou_matrix.size == 0:
            return matches
        
        pairs = []
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i, j] > threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        matched_i = set()
        matched_j = set()
        
        for iou_val, i, j in pairs:
            if i not in matched_i and j not in matched_j:
                matches.append((i, j))
                matched_i.add(i)
                matched_j.add(j)
        
        return matches


class TrackingService:
    """Tracking Service với Feature Database để giữ ID"""
    
    def __init__(self):
        self.tracker = None
        self.tracks: List[Dict] = []
        self.frame_id = 0
        self.track_history: Dict[int, Dict] = {}
        
        # === RE-ID ===
        self.locked_target_id: Optional[int] = None
        self.locked_lost_count: int = 0
        self.max_locked_lost: int = 30
        self.locked_feature: Optional[np.ndarray] = None
        
        # Track params
        self.track_thresh = 0.4
        self.high_thresh = 0.6
        self.match_thresh = 0.7
        self.track_buffer = 30
        self.feature_max_age = 60.0  # Giữ feature 60 giây
        
        self._init_tracker()
        logger.info("Tracking Service initialized with Feature Database")
    
    def _init_tracker(self):
        """Khởi tạo ByteTrack với Feature Database"""
        self.tracker = ByteTrackWithFeature(
            track_thresh=self.track_thresh,
            high_thresh=self.high_thresh,
            match_thresh=self.match_thresh,
            frame_rate=30,
            track_buffer=self.track_buffer,
            feature_max_age=self.feature_max_age
        )
        logger.info("ByteTrackWithFeature initialized")
    
    def reset(self):
        """Reset tracker - GIỮ feature database"""
        self.tracks = []
        self.track_history = {}
        self.frame_id = 0
        self.locked_target_id = None
        self.locked_lost_count = 0
        self.locked_feature = None
        if self.tracker:
            self.tracker.reset()
        logger.info(f"Tracker reset, DB size: {self.tracker.feature_db.get_stats()['total_persons'] if self.tracker else 0}")
    
    def lock_target(self, track_id: int, frame: np.ndarray = None):
        """Lock target"""
        self.locked_target_id = track_id
        self.locked_lost_count = 0
        
        if frame is not None:
            track = self.get_track_by_id(track_id)
            if track and track.get('bbox'):
                self.locked_feature = self._extract_feature(frame, track['bbox'])
                logger.info(f"Target {track_id} LOCKED")
    
    def unlock_target(self):
        """Unlock target"""
        self.locked_target_id = None
        self.locked_lost_count = 0
        self.locked_feature = None
        logger.info("Target UNLOCKED")
    
    def _extract_feature(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Trích xuất đặc trưng cho re-ID"""
        try:
            x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = min(x1, w-2), min(y1, h-2)
            x2, y2 = min(x2, w), min(y2, h)
            
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                return None
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-6)
            
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            size = (x2 - x1) * (y2 - y1) / (w * h)
            
            feature = np.concatenate([hist, [cx * 0.1, cy * 0.1, size * 0.1]])
            return feature
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Update tracker với detections mới"""
        self.frame_id += 1
        
        if self.tracker is None:
            return []
        
        if not detections:
            tracks = self.tracker.update(np.array([]), frame)
            self.tracks = self._convert_tracks(tracks)
            return self.tracks
        
        # Format detections - CHỈ LỌC person và ball
        dets = []
        for d in detections:
            class_id = d.get('class_id', 0)
            if class_id not in [0, 32]:
                continue
            x1, y1, x2, y2 = d['bbox']
            dets.append([float(x1), float(y1), float(x2), float(y2), 
                        float(d['confidence']), int(class_id)])
        
        if not dets:
            tracks = self.tracker.update(np.array([]), frame)
            self.tracks = self._convert_tracks(tracks)
            return self.tracks
        
        dets = np.array(dets)
        
        try:
            tracks = self.tracker.update(dets, frame)
            self.tracks = self._convert_tracks(tracks)
            self._reid_check(frame)
            
        except Exception as e:
            logger.error(f"Tracker update error: {e}")
            for t in self.tracks:
                t['lost'] = t.get('lost', 0) + 1
        
        return self.tracks
    
    def _convert_tracks(self, tracks: List[Track]) -> List[Dict]:
        """Chuyển đổi Track objects thành dict - KHÔNG bao gồm L"""
        result = []
        for t in tracks:
            if not t.is_valid():
                continue
            track_dict = {
                'track_id': t.track_id,
                'bbox': t.bbox,
                'center': t.center,
                'confidence': t.confidence,
                'class_id': t.class_id,
                'class': 'person' if t.class_id == 0 else 'ball',
                'lost': t.lost,  # Vẫn lưu nhưng không hiển thị
                'hits': t.hits,
                'age': t.age,
                'frame_count': self.frame_id,
                'velocity': t.velocity
                # KHÔNG có 'L:' trong label
            }
            result.append(track_dict)
            self.track_history[t.track_id] = track_dict.copy()
        
        return result
    
    def _reid_check(self, frame: np.ndarray):
        """Kiểm tra re-ID cho target đã lock"""
        if self.locked_target_id is None:
            return
        
        found = False
        
        for t in self.tracks:
            if t['track_id'] == self.locked_target_id:
                found = True
                self.locked_lost_count = 0
                new_feat = self._extract_feature(frame, t['bbox'])
                if new_feat is not None:
                    self.locked_feature = new_feat
                break
        
        if not found and self.locked_feature is not None:
            best_match = None
            best_similarity = 0.5
            
            for t in self.tracks:
                if t['class_id'] != 0:
                    continue
                t_feat = self._extract_feature(frame, t['bbox'])
                if t_feat is not None:
                    # Sử dụng hàm similarity của tracker
                    similarity = self.tracker.feature_db._compute_similarity(self.locked_feature, t_feat)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = t
            
            if best_match is not None:
                old_id = self.locked_target_id
                new_id = best_match['track_id']
                logger.info(f"Re-ID: {old_id} -> {new_id} (sim: {best_similarity:.2f})")
                self.locked_target_id = new_id
                self.locked_lost_count = 0
                found = True
        
        if not found:
            self.locked_lost_count += 1
            if self.locked_lost_count > self.max_locked_lost:
                logger.info(f"Target {self.locked_target_id} lost, unlocking")
                self.unlock_target()
    
    def predict(self, frame: np.ndarray) -> List[Dict]:
        """Dự đoán khi không có detection"""
        self.frame_id += 1
        
        if not self.tracks:
            return []
        
        for t in self.tracks:
            t['lost'] = t.get('lost', 0) + 1
        
        return self.tracks
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict]:
        for t in self.tracks:
            if t['track_id'] == track_id:
                return t
        return self.track_history.get(track_id)
    
    def get_tracks(self) -> List[Dict]:
        return self.tracks
    
    def is_target_locked(self) -> bool:
        return self.locked_target_id is not None
    
    def get_locked_target_id(self) -> Optional[int]:
        return self.locked_target_id
    
    def get_feature_db_stats(self) -> Dict:
        """Lấy thống kê database"""
        if self.tracker:
            return self.tracker.feature_db.get_stats()
        return {}
    
    def set_params(self, track_thresh: float = 0.5, high_thresh: float = 0.6, 
                   match_thresh: float = 0.8, frame_rate: int = 30):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self._init_tracker()
        logger.info(f"Tracker params updated")