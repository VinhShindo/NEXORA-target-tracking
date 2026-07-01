"""
Detection Service - Hỗ trợ nhiều model (Person, Ball, Both)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import time
import threading
import queue

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available.")

logger = logging.getLogger(__name__)

class DetectionService:
    """Service for object detection với hỗ trợ nhiều model"""
    
    def __init__(self):
        self.models = {}
        self.current_mode = "both"
        self.current_model = None
        self.is_model_loaded = False
        self.is_loading = False
        self.loading_progress = 0
        
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.device = 'cpu'
        self.selected_target_id = None
        self.frame_counter = 0
        self.process_every_n_frames = 1
        
        self.last_detections = []
        self.inference_size = 320
        
        self.class_mapping = {}
        self.class_names = {0: 'person', 32: 'ball'}
        self.allowed_classes = [0, 32]
        
        self.detection_times = []
        self.avg_detection_time = 0
        self.nms_threshold = 0.4 
        
        self.project_root = Path(__file__).resolve().parents[2]
        self.weights_dir = self.project_root / "weights"
        
        self.model_status = {
            'person': {'loaded': False, 'path': None},
            'ball': {'loaded': False, 'path': None},
            'both': {'loaded': False, 'path': None}
        }
        
        self._load_default_config()
        logger.info("Detection Service initialized with multi-model support")
    
    def _load_default_config(self):
        self.model_paths = {
            'person': self.weights_dir / 'yolov8n_person_lr001.pt',
            'ball': self.weights_dir / 'yolov8n_ball.pt',
            'both': self.weights_dir / 'yolov8n_person.pt'
        }
        
        self.class_mapping = {
            'person': {0: 'person'},
            'ball': {32: 'ball'},  
            'both': {0: 'person', 32: 'ball'}
        }
    
    def set_selected_target(self, track_id: Optional[int]):
        self.selected_target_id = track_id
    
    def load_models(self, config: Dict):
        models_config = config.get('models', {})
        
        for mode, cfg in models_config.items():
            model_path = cfg.get('path')
            if model_path:
                path = Path(model_path)
                if not path.is_absolute():
                    path = self.project_root / path
                
                self.model_paths[mode] = path
                self.model_status[mode]['path'] = str(path)
                
                if path.exists():
                    self.model_status[mode]['loaded'] = False
                    logger.info(f"✓ Model found: {mode} at {path}")
                else:
                    logger.warning(f"✗ Model not found: {mode} at {path}")
        
        self.current_mode = config.get('default_mode', 'both')
        self.class_mapping = {
            'person': {0: 'person'},
            'ball': {32: 'ball'},
            'both': {0: 'person', 32: 'ball'}
        }
        
        self._load_model_async(self.current_mode)
    
    def _load_model_async(self, mode: str):
        if self.is_loading:
            logger.warning(f"Already loading a model, please wait")
            return
        
        def load_thread():
            self.is_loading = True
            self.loading_progress = 0
            logger.info(f"🔄 Loading {mode} model...")
            
            try:
                path = self.model_paths.get(mode)
                if not path or not path.exists():
                    logger.error(f"Model not found: {path}")
                    self.is_loading = False
                    return
                
                self.loading_progress = 30
                logger.info(f"📥 Loading from {path}")
                
                model = YOLO(str(path))
                self.loading_progress = 80
                
                self.models[mode] = {
                    'model': model,
                    'classes': self.class_mapping.get(mode, {}),
                    'path': str(path)
                }
                self.model_status[mode]['loaded'] = True
                
                self.current_model = model
                self.current_mode = mode
                self.is_model_loaded = True
                self.loading_progress = 100
                
                logger.info(f"✅ {mode} model loaded successfully!")
                
                self.frame_counter = 0
                self.last_detections = []
                
            except Exception as e:
                logger.error(f"Failed to load {mode} model: {e}")
                self.is_model_loaded = False
            finally:
                self.is_loading = False
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def switch_mode(self, mode: str) -> bool:
        if mode not in self.model_paths:
            logger.error(f"✗ Mode {mode} not available")
            return False
        
        if mode == self.current_mode and self.is_model_loaded:
            logger.info(f"✓ Already using {mode} mode")
            return True
        
        self._unload_model()
        self._load_model_async(mode)
        return True
    
    def _unload_model(self):
        if self.current_model is not None:
            logger.info(f"🔄 Unloading {self.current_mode} model...")
            self.current_model = None
            self.is_model_loaded = False
            import gc
            gc.collect()
            logger.info(f"✅ {self.current_mode} model unloaded")
    
    def get_loading_status(self) -> Dict:
        return {
            'is_loading': self.is_loading,
            'progress': self.loading_progress,
            'current_mode': self.current_mode,
            'is_loaded': self.is_model_loaded,
            'available_models': list(self.model_paths.keys())
        }
    
    def load_model(self, model_path: str = None, confidence_threshold: float = 0.5, iou_threshold: float = 0.45, device: str = "cpu") -> bool:
        return True
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        detections = []
        
        if not self.is_model_loaded or self.current_model is None:
            return detections
        
        self.frame_counter += 1
        
        if self.frame_counter % self.process_every_n_frames != 0:
            return self.last_detections
        
        try:
            start_time = time.perf_counter()
            
            classes = None
            if self.current_mode == 'person':
                classes = [0]
            elif self.current_mode == 'ball':
                classes = [32]
            else:  # both
                classes = [0, 32]
            
            results = self.current_model(
                frame,
                imgsz=self.inference_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,   
                device=self.device,
                classes=classes,
                verbose=False,
                half=False,
                max_det=30
            )
            
            inference_time = (time.perf_counter() - start_time) * 1000
            self.detection_times.append(inference_time)
            if len(self.detection_times) > 30:
                self.detection_times.pop(0)
            self.avg_detection_time = sum(self.detection_times) / len(self.detection_times)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        for i in range(len(boxes)):
                            try:
                                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                                confidence = float(boxes.conf[i].cpu().numpy())
                                class_id = int(boxes.cls[i].cpu().numpy())
                                
                                class_map = self.class_mapping.get(self.current_mode, {})
                                class_name = class_map.get(class_id, f'class_{class_id}')
                                
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
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        if frame is None or not tracks:
            return frame
        
        colors = {}
        COLOR_PALETTE = [
            (0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255),
            (255, 0, 0), (0, 128, 255), (255, 192, 203), (0, 255, 128)
        ]
        
        for track in tracks:
            # Sửa: Chỉ vẽ những track lost < 4 frame
            if track.get('lost', 0) > 3:
                continue
            
            bbox = track.get('bbox')
            if not bbox or len(bbox) < 4: continue
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1))
            x2 = max(1, min(w, x2)); y2 = max(1, min(h, y2))
            if x2 <= x1 or y2 <= y1: continue
            
            track_id = track.get('track_id')
            class_name = track.get('class', 'person')
            
            if track_id not in colors:
                colors[track_id] = COLOR_PALETTE[track_id % len(COLOR_PALETTE)]
            color = colors[track_id]
            
            if self.selected_target_id is not None and track_id == self.selected_target_id:
                color = (0, 255, 255)
                thickness = 3
            else:
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{class_name}:{track_id}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(th + 5, y1 - 5)
            
            cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 6, label_y + 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, label_y - th - 5), (x1 + tw + 6, label_y + 5), color, 1)
            cv2.putText(frame, label, (x1 + 3, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
        
        return frame
    
    def reset_frame_counter(self):
        self.frame_counter = 0; self.last_detections = []
    
    def set_process_interval(self, n_frames: int):
        if n_frames > 0: self.process_every_n_frames = n_frames
    
    def set_inference_size(self, size: int):
        self.inference_size = size
    
    def get_performance_stats(self) -> Dict:
        return {
            'avg_detection_time': self.avg_detection_time,
            'process_every_n_frames': self.process_every_n_frames,
            'inference_size': self.inference_size,
            'detections_count': len(self.last_detections),
            'current_mode': self.current_mode,
            'model_loaded': self.is_model_loaded,
            'is_loading': self.is_loading,
            'loading_progress': self.loading_progress
        }
    
    def get_available_modes(self) -> List[str]:
        return list(self.model_paths.keys())
    
    def get_model_status(self) -> Dict:
        status = {}
        for mode, path in self.model_paths.items():
            status[mode] = {
                'exists': path.exists(),
                'loaded': self.model_status.get(mode, {}).get('loaded', False),
                'path': str(path),
                'current': mode == self.current_mode
            }
        return status   