"""
Robot Service - Cải tiến khoảng cách với bộ lọc phối cảnh (Perspective + Area + Y-Center)
GIẢI QUYẾT VẤN ĐỀ: Camera cao 50cm dẫn đến sai lệch khoảng cách khi người đi ngang.
Sử dụng diện tích bbox và sự thay đổi tâm Y để điều khiển tiến/lùi chính xác.
"""

import logging
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from collections import deque

from controllers.steering_pid import SteeringPID
from controllers.distance_pid import DistancePID

logger = logging.getLogger(__name__)

class RobotState(Enum):
    IDLE = "IDLE"
    FOLLOWING = "FOLLOWING"
    TURN_LEFT = "TURN LEFT"
    TURN_RIGHT = "TURN RIGHT"
    STOP = "STOP"
    SEARCHING = "SEARCHING"
    TARGET_LOST = "TARGET LOST"

@dataclass
class RobotTelemetry:
    target_id: Optional[int] = None
    tracking_state: str = "IDLE"
    error_x: float = 0.0
    distance_error: float = 0.0 # Sai số khoảng cách ước lượng
    servo_angle: float = 90.0
    motor_speed: float = 0.0
    bbox_height: float = 0.0
    bbox_width: float = 0.0
    bbox_area: float = 0.0
    target_center_x: float = 0.0
    target_center_y: float = 0.0
    frame_center_x: float = 320.0
    frame_center_y: float = 240.0
    pid_steering_output: float = 0.0
    pid_distance_output: float = 0.0
    estimated_distance: float = 0.0 # Khoảng cách thực tế tính ra mét
    fps: float = 0.0
    last_update: float = 0.0
    target_height: float = 150.0  # Giữ lại cho tương thích

class RobotService:
    """Service for robot control with PID steering and distance"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.steering_pid = SteeringPID()
        self.distance_pid = DistancePID()
        
        # ==== BIẾN MỚI CHO KHOẢNG CÁCH CHÍNH XÁC ====
        self.target_pixel_area = 4000.0   # Diện tích bbox mục tiêu (khi đạt khoảng cách mong muốn)
        self.distance_k_factor = 25000.0  # Hệ số chuyển đổi từ Pixel sang Met (Cần tinh chỉnh khi setup)
        self.target_real_distance = 2.0   # Khoảng cách mong muốn (mét)
        
        self.telemetry = RobotTelemetry()
        self.telemetry.target_height = self.target_pixel_area
        
        self.state = RobotState.IDLE
        self.direction = "STOP"
        self.is_target_lost = False
        
        self.estimated_distance = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.max_linear = 0.5
        self.max_angular = 1.0
        self.min_distance = 0.5
        self.max_distance = 5.0
        
        # ===== BỘ LỌC NHIỄU =====
        self.bbox_buffer: deque = deque(maxlen=5)
        self.area_buffer: deque = deque(maxlen=5)    # Mới: Lưu diện tích
        self.center_buffer: deque = deque(maxlen=5)
        
        # Deadzone mới dựa trên Area
        self.deadzone_area = 200.0    # Bỏ qua nhiễu diện tích nhỏ hơn 200 pixel
        
        self.smooth_factor = 0.3
        self.smoothed_area = 0.0
        self.smoothed_center_x = 0.0
        self.smoothed_center_y = 0.0
        self.smoothed_height = 0.0
        self.smoothed_width = 0.0
        
        # Lưu frame trước để tính delta Y (quan trọng cho camera 50cm)
        self.prev_center_y = 0.0
        self.delta_center_y = 0.0
        
        self.max_change_center = 50.0
        self.max_change_area = 5000.0  # Giới hạn thay đổi diện tích mỗi frame
        
        self.stability_mode = True
        self.stability_threshold = 0.5
        
        if config:
            self._apply_config(config)
    
    def _apply_config(self, config: Dict):
        steer_cfg = config.get('steering_pid', {})
        self.steering_pid.kp = steer_cfg.get('kp', 0.05)
        self.steering_pid.ki = steer_cfg.get('ki', 0.001)
        self.steering_pid.kd = steer_cfg.get('kd', 0.01)
        
        dist_cfg = config.get('distance_pid', {})
        self.distance_pid.kp = dist_cfg.get('kp', 0.1)
        self.distance_pid.ki = dist_cfg.get('ki', 0.002)
        self.distance_pid.kd = dist_cfg.get('kd', 0.02)
        
        # Lấy cấu hình mới nếu có
        self.target_pixel_area = config.get('target_pixel_area', 4000.0)
        self.distance_k_factor = config.get('distance_k_factor', 25000.0)
        self.target_real_distance = config.get('target_real_distance', 2.0)
        self.deadzone_area = config.get('deadzone_area', 200.0)
        
    def update_pid_params(self, steering=None, distance=None, target_height=None):
        if steering:
            if 'kp' in steering: self.steering_pid.kp = steering['kp']
            if 'ki' in steering: self.steering_pid.ki = steering['ki']
            if 'kd' in steering: self.steering_pid.kd = steering['kd']
            self.steering_pid.reset()
        if distance:
            if 'kp' in distance: self.distance_pid.kp = distance['kp']
            if 'ki' in distance: self.distance_pid.ki = distance['ki']
            if 'kd' in distance: self.distance_pid.kd = distance['kd']
            self.distance_pid.reset()
        if target_height is not None:
            # Có thể dùng target_height cũ để set area, nhưng ta giữ lại k_factor
            self.target_pixel_area = target_height * 1.2
            self.telemetry.target_height = target_height
    
    def start_following(self):
        self.state = RobotState.FOLLOWING
        self.telemetry.tracking_state = "FOLLOWING"
        self.steering_pid.reset()
        self.distance_pid.reset()
        self.bbox_buffer.clear()
        self.area_buffer.clear()
        self.center_buffer.clear()
    
    def _smooth_value(self, new_value: float, smoothed_value: float, max_change: float = None) -> float:
        if max_change is None:
            max_change = self.max_change_center
        
        if abs(new_value - smoothed_value) > max_change and smoothed_value > 0:
            if new_value > smoothed_value:
                new_value = smoothed_value + max_change
            else:
                new_value = smoothed_value - max_change
        
        if smoothed_value == 0:
            return new_value
        return smoothed_value * (1 - self.smooth_factor) + new_value * self.smooth_factor

    def update_target(self, track, frame_width, frame_height, fps=0.0):
        now = time.time()
        self.telemetry.last_update = now
        self.telemetry.fps = fps
        self.telemetry.frame_center_x = frame_width // 2
        self.telemetry.frame_center_y = frame_height // 2
        
        if track is None or not track.get('bbox'):
            self._set_lost_state()
            return
        
        bbox = track['bbox']
        x1, y1, x2, y2 = bbox
        
        raw_center_x = (x1 + x2) / 2.0
        raw_center_y = (y1 + y2) / 2.0
        raw_bbox_width = x2 - x1
        raw_bbox_height = y2 - y1
        raw_bbox_area = raw_bbox_width * raw_bbox_height
        
        # ===== LỌC NHIỄU & LÀM MỊN =====
        self.bbox_buffer.append(bbox)
        self.area_buffer.append(raw_bbox_area)
        self.center_buffer.append([raw_center_x, raw_center_y])
        
        # Median filter cho Area (ít nhạy cảm hơn height)
        if len(self.area_buffer) >= 3:
            areas = sorted(self.area_buffer)
            centers_x = sorted([c[0] for c in self.center_buffer])
            centers_y = sorted([c[1] for c in self.center_buffer])
            
            median_area = areas[len(areas)//2]
            median_center_x = centers_x[len(centers_x)//2]
            median_center_y = centers_y[len(centers_y)//2]
            median_height = sorted([self.height_buffer[i] for i in range(len(self.height_buffer))]) if self.height_buffer else raw_bbox_height
            # Dùng area cho mịn hơn
        else:
            median_area = raw_bbox_area
            median_center_x = raw_center_x
            median_center_y = raw_center_y
            median_height = raw_bbox_height
        
        # EMA Smooth
        if self.smoothed_area == 0:
            self.smoothed_area = median_area
            self.smoothed_center_x = median_center_x
            self.smoothed_center_y = median_center_y
            self.smoothed_height = median_height
            self.smoothed_width = raw_bbox_width
        else:
            self.smoothed_area = self._smooth_value(median_area, self.smoothed_area, self.max_change_area)
            self.smoothed_center_x = self._smooth_value(median_center_x, self.smoothed_center_x)
            self.smoothed_center_y = self._smooth_value(median_center_y, self.smoothed_center_y)
            self.smoothed_height = self._smooth_value(median_height, self.smoothed_height)
            self.smoothed_width = self._smooth_value(raw_bbox_width, self.smoothed_width)
        
        # ===== ƯỚC LƯỢNG KHOẢNG CÁCH THỰC TẾ (Perspective Transform) =====
        # D = K / sqrt(Area)
        if self.smoothed_area > 100:
            self.estimated_distance = self.distance_k_factor / np.sqrt(self.smoothed_area)
        else:
            self.estimated_distance = self.max_distance
        
        self.telemetry.target_id = track.get('track_id')
        self.telemetry.target_center_x = self.smoothed_center_x
        self.telemetry.target_center_y = self.smoothed_center_y
        self.telemetry.bbox_width = self.smoothed_width
        self.telemetry.bbox_height = self.smoothed_height
        self.telemetry.bbox_area = self.smoothed_area
        self.telemetry.estimated_distance = self.estimated_distance
        self.telemetry.target_height = self.target_pixel_area  # Cập nhật đồng bộ
        
        # ===== LẤY DELTA Y (ĐỂ XÁC ĐỊNH XU HƯỚNG TIẾN/LÙI) =====
        if self.prev_center_y > 0:
            self.delta_center_y = self.smoothed_center_y - self.prev_center_y
        self.prev_center_y = self.smoothed_center_y
        
        # ===== PID STEERING (ĐIỀU KHIỂN RẼ) =====
        error_x = self.smoothed_center_x - self.telemetry.frame_center_x
        if abs(error_x) < 2: error_x = 0.0 # Deadzone rẽ
        
        steering_output = self.steering_pid.update(error_x)
        self.telemetry.error_x = error_x
        self.telemetry.pid_steering_output = steering_output
        
        servo_angle = max(45.0, min(135.0, 90.0 + steering_output))
        self.telemetry.servo_angle = servo_angle
        
        # ===== PID DISTANCE (ĐIỀU KHIỂN TIẾN/LÙI) - DỰA VÀO DIỆN TÍCH =====
        # Sai số diện tích: Tiến nếu area < target_area, lùi nếu area > target_area
        error_area = self.target_pixel_area - self.smoothed_area
        
        # Điều chỉnh error dựa trên Delta Y (Camera cao 50cm sẽ giúp rất nhiều)
        # Nếu area nhỏ nhưng delta Y dương (người đang đi xuống dưới) -> đang tiến nhanh
        if self.delta_center_y > 5.0:
            error_area *= 1.5  # Tăng tốc tiến lên
        elif self.delta_center_y < -5.0:
            error_area *= 0.5  # Giảm tốc (vì người đang đi ngang/lùi)

        if abs(error_area) < self.deadzone_area:
            error_area = 0.0
            
        distance_output = self.distance_pid.update(error_area)
        self.telemetry.distance_error = error_area
        self.telemetry.pid_distance_output = distance_output
        
        # Điều chỉnh motor speed
        motor_speed = max(-100.0, min(100.0, distance_output))
        self.telemetry.motor_speed = motor_speed
        
        self.is_target_lost = False
        self.telemetry.tracking_state = "FOLLOWING"
        
        # ===== CHẾ ĐỘ ỔN ĐỊNH STABILITY FACTOR =====
        if abs(error_x) < 5 and abs(error_area) < 100:
            stability_factor = 0.3
            steering_output *= stability_factor
            motor_speed *= stability_factor
        
        # ===== XÁC ĐỊNH HƯỚNG DI CHUYỂN =====
        threshold = 15
        if abs(error_x) < threshold:
            self.direction = "FORWARD"
            self.state = RobotState.FOLLOWING
        elif error_x > 0:
            self.direction = "TURN RIGHT"
            self.state = RobotState.TURN_RIGHT
        else:
            self.direction = "TURN LEFT"
            self.state = RobotState.TURN_LEFT
            
        # ===== TÍNH VẬN TỐC XE =====
        if motor_speed > 0:
            self.linear_velocity = self.max_linear * (motor_speed / 100.0)
        elif motor_speed < 0:
            self.linear_velocity = -self.max_linear * (abs(motor_speed) / 100.0)
        else:
            self.linear_velocity = 0.0
            
        self.angular_velocity = max(-self.max_angular, min(self.max_angular, -steering_output * 0.01))
        
        # Cập nhật buffer chiều cao để phục vụ cho median filter
        self.height_buffer.append(self.smoothed_height)
        if len(self.height_buffer) > 5: self.height_buffer.popleft()
        self.width_buffer.append(self.smoothed_width)
        if len(self.width_buffer) > 5: self.width_buffer.popleft()

    def _set_lost_state(self):
        self.is_target_lost = True
        self.state = RobotState.TARGET_LOST
        self.telemetry.tracking_state = "TARGET LOST"
        self.telemetry.target_id = None
        self.telemetry.error_x = 0.0
        self.telemetry.distance_error = 0.0
        self.telemetry.servo_angle = 90.0
        self.telemetry.motor_speed = 0.0
        self.telemetry.target_center_x = 0.0
        self.telemetry.target_center_y = 0.0
        self.telemetry.bbox_area = 0.0
        self.telemetry.estimated_distance = 0.0
        self.telemetry.target_height = self.target_pixel_area
        self.direction = "STOP"
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.bbox_buffer.clear()
        self.area_buffer.clear()
        self.center_buffer.clear()
        self.height_buffer.clear()
        self.width_buffer.clear()
        self.smoothed_area = 0
        self.smoothed_height = 0
        self.smoothed_width = 0
        self.smoothed_center_x = 0
        self.smoothed_center_y = 0
        self.prev_center_y = 0
    
    def handle_target_lost(self):
        self.is_target_lost = True
        self.state = RobotState.SEARCHING
        self.telemetry.tracking_state = "SEARCHING"
        self.direction = "SEARCHING"
        self.linear_velocity = 0.1
        self.angular_velocity = 0.3
        self.telemetry.servo_angle = 90.0
        self.telemetry.motor_speed = 0.0
        self.steering_pid.reset()
        self.distance_pid.reset()
    
    def stop(self):
        self.state = RobotState.STOP
        self.direction = "STOP"
        self.telemetry.tracking_state = "STOP"
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.telemetry.servo_angle = 90.0
        self.telemetry.motor_speed = 0.0
        self.is_target_lost = True
        self.steering_pid.reset()
        self.distance_pid.reset()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'state': self.state.value,
            'direction': self.direction,
            'distance': round(self.estimated_distance, 2),
            'offset_x': round(self.telemetry.error_x, 2),
            'target_id': self.telemetry.target_id,
            'target_lost': self.is_target_lost,
            'linear_velocity': round(self.linear_velocity, 3),
            'angular_velocity': round(self.angular_velocity, 3),
            'pid_error': round(self.telemetry.error_x, 2),
            'servo_angle': round(self.telemetry.servo_angle, 1),
            'motor_speed': round(self.telemetry.motor_speed, 1)
        }
    
    def get_telemetry(self) -> RobotTelemetry:
        return self.telemetry
    
    def update_frame_center(self, width: int, height: int):
        self.telemetry.frame_center_x = width // 2
        self.telemetry.frame_center_y = height // 2
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if frame is None: return frame
        try:
            t = self.telemetry
            h, w = frame.shape[:2]
            cx = int(t.frame_center_x) if t.frame_center_x > 0 else w // 2
            cy = int(t.frame_center_y) if t.frame_center_y > 0 else h // 2
            
            cv2.line(frame, (cx, 0), (cx, h), (255, 0, 0), 1)
            cv2.line(frame, (0, cy), (w, cy), (255, 0, 0), 1)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            if t.target_id is not None and t.target_center_x > 0:
                target_cx = max(0, min(w-1, int(t.target_center_x)))
                target_cy = max(0, min(h-1, int(t.target_center_y)))
                
                cv2.circle(frame, (target_cx, target_cy), 5, (0, 255, 255), -1)
                cv2.line(frame, (cx, cy), (target_cx, target_cy), (255, 255, 0), 1)
                
                texts = [
                    f"ID: {t.target_id}",
                    f"ERR_X: {t.error_x:.1f}",
                    f"SERVO: {t.servo_angle:.1f}",
                    f"MOTOR: {t.motor_speed:.1f}",
                    f"DIST(m): {t.estimated_distance:.2f}", # Thêm khoảng cách thực tế
                    f"FPS: {t.fps:.1f}"
                ]
                for i, txt in enumerate(texts):
                    cv2.putText(frame, txt, (10, 30 + i*25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO TARGET", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            pass
        return frame