"""
Camera Service - Handles camera capture
OPTIMIZED: MJPEG support for higher FPS
"""

import cv2
import logging
from typing import Optional, Tuple
import time
import numpy as np

logger = logging.getLogger(__name__)

class CameraService:
    """Service for camera capture with MJPEG support"""
    
    def __init__(self):
        self.cap = None
        self.is_connected = False
        self.default_camera_id = 1
        self.default_width = 640
        self.default_height = 480
        self.default_fps = 30
        self.use_mjpeg = True
        self.frame_count = 0
        self.last_read_time = 0
        self.fps_counter = 0
        self.current_fps = 0
        
    def start(self, camera_id: int = 1, width: int = 640, height: int = 480) -> bool:
        """Start camera with DirectShow and MJPEG support"""
        try:
            # Stop existing camera
            self.stop()
            
            # Sử dụng DirectShow trên Windows
            if hasattr(cv2, 'CAP_DSHOW'):
                logger.info("Using DirectShow for camera capture")
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return False
            
            # BẬT MJPEG - TĂNG FPS ĐÁNG KỂ
            if self.use_mjpeg:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                logger.info("MJPEG mode enabled")
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.default_fps)
            
            # Kiểm tra FPS thực tế
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Camera FPS reported: {actual_fps}")
            
            # Warmup - đọc vài frame để ổn định
            for i in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    logger.info(f"Warmup frame {i+1} captured")
                time.sleep(0.05)
            
            # Test read
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Camera test read failed")
                self.stop()
                return False
            
            self.is_connected = True
            self.default_camera_id = camera_id
            self.default_width = width
            self.default_height = height
            self.frame_count = 0
            self.last_read_time = time.time()
            
            logger.info(f"Camera {camera_id} started successfully")
            logger.info(f"Resolution: {width}x{height}, MJPEG: {self.use_mjpeg}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            self.is_connected = False
            return False
    
    def stop(self):
        """Stop camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        logger.info("Camera stopped")
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame from camera"""
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_count += 1
                
                # Tính FPS thực tế
                current_time = time.time()
                if current_time - self.last_read_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_read_time = current_time
                else:
                    self.fps_counter += 1
                
                return frame
            else:
                # Try to reconnect
                logger.warning("Camera read failed, attempting reconnect...")
                self.is_connected = False
                self.start(self.default_camera_id, self.default_width, self.default_height)
                return None
                
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        if self.cap is None:
            return {}
        
        info = {
            'connected': self.is_connected,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC))
        }
        return info