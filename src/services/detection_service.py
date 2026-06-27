"""
Detection Service - Tối ưu cho tracking ổn định
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available.")

logger = logging.getLogger(__name__)

class DetectionService:
    """Service for object detection using YOLOv8"""
    
    def __init__(self):
        self.model = None
        self.is_model_loaded = False
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.device = 'cpu'
        self.selected_target_id = None
        self.frame_counter = 0
        self.process_every_n_frames = 1
        self.project_root = Path(__file__).resolve().parents[2]
        self.weights_dir = self.project_root / "weights"
        self.default_model_path = str(self.weights_dir / "yolov8n_person.pt")
        
        self.last_detections = []
        self.inference_size = 320  # Tăng lên 320 cho accuracy tốt hơn
        
        self.class_names = {0: 'person', 32: 'ball'}
        self.allowed_classes = [0, 32]
        
        self.detection_times = []
        self.avg_detection_time = 0
        
        # NMS params
        self.nms_threshold = 0.4
        
        logger.info("Detection Service initialized")
    
    def set_selected_target(self, track_id: Optional[int]):
        self.selected_target_id = track_id
    
    def load_model(self, model_path: str = None, 
                   confidence_threshold: float = 0.5,
                   iou_threshold: float = 0.45,
                   device: str = "cpu") -> bool:
        try:
            if not YOLO_AVAILABLE:
                logger.error("YOLO not available")
                return False

            if model_path is None:
                model_path = self.default_model_path

            resolved_model_path = Path(model_path).expanduser()
            if not resolved_model_path.is_absolute():
                resolved_model_path = (self.project_root / resolved_model_path).resolve()

            if not resolved_model_path.exists():
                logger.warning(f"Model file not found: {resolved_model_path}")
                try:
                    self.model = YOLO('yolov8n.pt')
                    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    src = Path('yolov8n.pt')
                    if src.exists():
                        shutil.copy2(src, resolved_model_path)
                except Exception as e:
                    logger.error(f"Failed to download model: {e}")
                    return False
            else:
                self.model = YOLO(str(resolved_model_path))
            
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold
            self.device = device
            self.is_model_loaded = True
            self.frame_counter = 0
            self.last_detections = []
            self.detection_times = []
            
            logger.info(f"Model loaded from {resolved_model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run detection, return detections"""
        detections = []
        
        if not self.is_model_loaded or self.model is None:
            return detections
        
        self.frame_counter += 1
        
        if self.frame_counter % self.process_every_n_frames != 0:
            return self.last_detections
        
        try:
            start_time = time.perf_counter()
            
            # Tăng size inference để detect tốt hơn
            results = self.model(
                frame,
                imgsz=self.inference_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                classes=self.allowed_classes,
                verbose=False,
                half=False,
                max_det=30  # Giới hạn max detection
            )
            
            inference_time = (time.perf_counter() - start_time) * 1000
            self.detection_times.append(inference_time)
            if len(self.detection_times) > 30:
                self.detection_times.pop(0)
            self.avg_detection_time = sum(self.detection_times) / len(self.detection_times)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    # Lấy tất cả boxes
                    boxes = result.boxes
                    if len(boxes) > 0:
                        # Sắp xếp theo confidence giảm dần
                        confs = boxes.conf.cpu().numpy()
                        indices = np.argsort(-confs)
                        
                        # NMS thủ công để giảm detection trùng
                        kept_indices = []
                        bboxes = boxes.xyxy.cpu().numpy()
                        
                        for i in indices:
                            if len(kept_indices) == 0:
                                kept_indices.append(i)
                            else:
                                overlap = False
                                for j in kept_indices:
                                    iou = self._compute_iou(bboxes[i], bboxes[j])
                                    if iou > self.nms_threshold:
                                        overlap = True
                                        break
                                if not overlap:
                                    kept_indices.append(i)
                        
                        for idx in kept_indices:
                            try:
                                x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
                                confidence = float(boxes.conf[idx].cpu().numpy())
                                class_id = int(boxes.cls[idx].cpu().numpy())
                                
                                if class_id not in self.allowed_classes:
                                    continue
                                
                                class_name = self.class_names.get(class_id, f'class_{class_id}')
                                
                                detection = {
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'class': class_name,
                                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                                }
                                detections.append(detection)
                            except Exception:
                                continue
            
            self.last_detections = detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            detections = self.last_detections
        
        return detections
    
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
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Vẽ tracks lên frame với màu sắc ổn định - KHÔNG hiển thị L"""
        if frame is None or not tracks:
            return frame
        
        # Màu sắc cố định cho mỗi ID
        colors = {}
        
        for track in tracks:
            # Bỏ qua track đã lost quá lâu (không vẽ)
            if track.get('lost', 0) > 10:
                continue
            
            bbox = track.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            
            if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
                continue
            
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(1, min(w, x2))
            y2 = max(1, min(h, y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            track_id = track.get('track_id')
            confidence = track.get('confidence', 0)
            class_name = track.get('class', 'person')
            lost = track.get('lost', 0)
            
            # Tạo màu cố định cho mỗi ID
            if track_id not in colors:
                hash_val = hash(str(track_id))
                colors[track_id] = (
                    (hash_val & 0xFF),
                    ((hash_val >> 8) & 0xFF),
                    ((hash_val >> 16) & 0xFF)
                )
            
            color = colors[track_id]
            
            # Target được chọn: màu vàng nổi bật
            if self.selected_target_id is not None and track_id == self.selected_target_id:
                color = (0, 255, 255)
                thickness = 3
            else:
                thickness = 2
            
            # Vẽ bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Vẽ label - CHỈ HIỂN THỊ ID, KHÔNG HIỂN THỊ L
            if track_id is not None:
                label = f"ID:{track_id}"
                # KHÔNG thêm L vào label
            else:
                label = class_name
            
            # Label background
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(th + 5, y1 - 5)
            cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 4, label_y), color, -1)
            cv2.putText(frame, label, (x1 + 2, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Center dot
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
        return frame
    
    def reset_frame_counter(self):
        self.frame_counter = 0
        self.last_detections = []
    
    def set_process_interval(self, n_frames: int):
        if n_frames > 0:
            self.process_every_n_frames = n_frames
            logger.info(f"Processing interval set to {n_frames} frames")
    
    def set_inference_size(self, size: int):
        self.inference_size = size
        logger.info(f"Inference size set to {size}")
    
    def get_performance_stats(self) -> Dict:
        return {
            'avg_detection_time': self.avg_detection_time,
            'process_every_n_frames': self.process_every_n_frames,
            'inference_size': self.inference_size,
            'detections_count': len(self.last_detections)
        }