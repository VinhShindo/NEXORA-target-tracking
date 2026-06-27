"""
Robot Service - Thêm bộ lọc nhiễu cho bbox ổn định
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
    distance_error: float = 0.0
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
    fps: float = 0.0
    last_update: float = 0.0
    target_height: float = 150.0

class RobotService:
    """Service for robot control with PID steering and distance"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.steering_pid = SteeringPID()
        self.distance_pid = DistancePID()
        self.target_height = 150
        
        self.telemetry = RobotTelemetry()
        self.telemetry.target_height = self.target_height
        
        self.state = RobotState.IDLE
        self.direction = "STOP"
        self.is_target_lost = False
        
        self.estimated_distance = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.max_linear = 0.5
        self.max_angular = 1.0
        self.desired_distance = 2.0
        self.min_distance = 0.5
        self.max_distance = 5.0
        
        # ===== BỘ LỌC NHIỄU =====
        # Lọc trung bình động cho bbox
        self.bbox_buffer: deque = deque(maxlen=5)  # Lưu 5 frame gần nhất
        self.center_buffer: deque = deque(maxlen=5)
        self.height_buffer: deque = deque(maxlen=5)
        self.width_buffer: deque = deque(maxlen=5)
        
        # Ngưỡng thay đổi tối thiểu (deadzone)
        self.deadzone_steering = 2.0  # Pixels - bỏ qua thay đổi nhỏ hơn này
        self.deadzone_distance = 3.0  # Pixels - bỏ qua thay đổi nhỏ hơn này
        
        # Lọc Kalman đơn giản cho vị trí
        self.smooth_factor = 0.3  # Trọng số cho giá trị mới (0.1-0.5)
        self.smoothed_center_x = 0.0
        self.smoothed_center_y = 0.0
        self.smoothed_height = 0.0
        self.smoothed_width = 0.0
        
        # Giới hạn thay đổi tối đa mỗi frame (chống nhảy)
        self.max_change_center = 50.0  # Pixels
        self.max_change_height = 30.0  # Pixels
        
        # Chế độ ổn định
        self.stability_mode = True
        self.stability_threshold = 0.5  # Giảm tốc độ khi target gần ổn định
        
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
        
        self.target_height = config.get('target_height', 150)
        self.telemetry.target_height = self.target_height
        
        # Thêm config cho bộ lọc
        self.deadzone_steering = config.get('deadzone_steering', 2.0)
        self.deadzone_distance = config.get('deadzone_distance', 3.0)
        self.smooth_factor = config.get('smooth_factor', 0.3)
    
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
            self.target_height = target_height
            self.telemetry.target_height = target_height
    
    def start_following(self):
        self.state = RobotState.FOLLOWING
        self.telemetry.tracking_state = "FOLLOWING"
        self.steering_pid.reset()
        self.distance_pid.reset()
        # Reset buffers
        self.bbox_buffer.clear()
        self.center_buffer.clear()
        self.height_buffer.clear()
        self.width_buffer.clear()
    
    def _smooth_value(self, new_value: float, smoothed_value: float) -> float:
        """Lọc làm mịn giá trị"""
        # Giới hạn thay đổi tối đa
        if abs(new_value - smoothed_value) > self.max_change_center and smoothed_value > 0:
            if new_value > smoothed_value:
                new_value = smoothed_value + self.max_change_center
            else:
                new_value = smoothed_value - self.max_change_center
        
        # Lọc trung bình động (EMA)
        if smoothed_value == 0:
            return new_value
        return smoothed_value * (1 - self.smooth_factor) + new_value * self.smooth_factor
    
    def _apply_deadzone(self, error: float, deadzone: float) -> float:
        """Áp dụng deadzone để bỏ qua nhiễu nhỏ"""
        if abs(error) < deadzone:
            return 0.0
        # Giảm dần để chuyển mượt
        if abs(error) < deadzone * 2:
            return error - np.sign(error) * deadzone
        return error
    
    def update_target(self, track, frame_width, frame_height, fps=0.0):
        """Update target info với bộ lọc nhiễu"""
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
        
        # Raw values từ detection
        raw_center_x = (x1 + x2) / 2.0
        raw_center_y = (y1 + y2) / 2.0
        raw_bbox_width = x2 - x1
        raw_bbox_height = y2 - y1
        
        # ===== ÁP DỤNG BỘ LỌC =====
        # Lưu vào buffer
        self.bbox_buffer.append(bbox)
        self.center_buffer.append([raw_center_x, raw_center_y])
        self.height_buffer.append(raw_bbox_height)
        self.width_buffer.append(raw_bbox_width)
        
        # Lấy giá trị trung bình từ buffer (Median filter)
        if len(self.height_buffer) >= 3:
            # Dùng median để loại bỏ outlier
            heights = sorted(self.height_buffer)
            widths = sorted(self.width_buffer)
            centers_x = sorted([c[0] for c in self.center_buffer])
            centers_y = sorted([c[1] for c in self.center_buffer])
            
            # Lấy median (giá trị giữa)
            median_height = heights[len(heights)//2]
            median_width = widths[len(widths)//2]
            median_center_x = centers_x[len(centers_x)//2]
            median_center_y = centers_y[len(centers_y)//2]
        else:
            median_height = raw_bbox_height
            median_width = raw_bbox_width
            median_center_x = raw_center_x
            median_center_y = raw_center_y
        
        # EMA smoothing
        if self.smoothed_height == 0:
            self.smoothed_height = median_height
            self.smoothed_width = median_width
            self.smoothed_center_x = median_center_x
            self.smoothed_center_y = median_center_y
        else:
            self.smoothed_height = self._smooth_value(median_height, self.smoothed_height)
            self.smoothed_width = self._smooth_value(median_width, self.smoothed_width)
            self.smoothed_center_x = self._smooth_value(median_center_x, self.smoothed_center_x)
            self.smoothed_center_y = self._smooth_value(median_center_y, self.smoothed_center_y)
        
        # Lưu giá trị đã lọc
        target_center_x = self.smoothed_center_x
        target_center_y = self.smoothed_center_y
        bbox_width = self.smoothed_width
        bbox_height = self.smoothed_height
        
        # Cập nhật telemetry với giá trị đã lọc
        self.telemetry.target_id = track.get('track_id')
        self.telemetry.target_center_x = target_center_x
        self.telemetry.target_center_y = target_center_y
        self.telemetry.bbox_width = bbox_width
        self.telemetry.bbox_height = bbox_height
        self.telemetry.bbox_area = bbox_width * bbox_height
        
        # Tính error
        error_x = target_center_x - self.telemetry.frame_center_x
        distance_error = self.target_height - bbox_height
        
        # ===== ÁP DỤNG DEADZONE =====
        error_x_filtered = self._apply_deadzone(error_x, self.deadzone_steering)
        distance_error_filtered = self._apply_deadzone(distance_error, self.deadzone_distance)
        
        self.telemetry.error_x = error_x_filtered
        self.telemetry.distance_error = distance_error_filtered
        
        # ===== PID OUTPUT =====
        steering_output = self.steering_pid.update(error_x_filtered)
        distance_output = self.distance_pid.update(distance_error_filtered)
        
        self.telemetry.pid_steering_output = steering_output
        self.telemetry.pid_distance_output = distance_output
        
        # ===== CHẾ ĐỘ ỔN ĐỊNH =====
        # Nếu error nhỏ, giảm tốc độ để tránh rung
        if abs(error_x_filtered) < 5 and abs(distance_error_filtered) < 5:
            stability_factor = 0.3  # Giảm 70% tốc độ khi ổn định
            steering_output *= stability_factor
            distance_output *= stability_factor
        
        # ===== SERVO ANGLE =====
        servo_angle = max(45.0, min(135.0, 90.0 + steering_output))
        self.telemetry.servo_angle = servo_angle
        
        # ===== MOTOR SPEED =====
        motor_speed = max(-100.0, min(100.0, distance_output))
        self.telemetry.motor_speed = motor_speed
        
        self.is_target_lost = False
        self.telemetry.tracking_state = "FOLLOWING"
        
        # ===== DIRECTION STATE =====
        threshold = 15  # Giảm threshold để ổn định hơn
        if abs(error_x_filtered) < threshold:
            self.direction = "FORWARD"
            self.state = RobotState.FOLLOWING
        elif error_x_filtered > 0:
            self.direction = "TURN RIGHT"
            self.state = RobotState.TURN_RIGHT
        else:
            self.direction = "TURN LEFT"
            self.state = RobotState.TURN_LEFT
        
        # ===== VELOCITY =====
        if motor_speed > 0:
            self.linear_velocity = self.max_linear * (motor_speed / 100.0)
        elif motor_speed < 0:
            self.linear_velocity = -self.max_linear * (abs(motor_speed) / 100.0)
        else:
            self.linear_velocity = 0.0
        
        self.angular_velocity = max(-self.max_angular, min(self.max_angular, -steering_output * 0.01))
        
        if bbox_height > 0:
            self.estimated_distance = min(self.max_distance, max(self.min_distance, self.max_distance / bbox_height))
    
    def _set_lost_state(self):
        """Set robot to lost target state"""
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
        self.direction = "STOP"
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        # Reset buffers khi mất target
        self.bbox_buffer.clear()
        self.center_buffer.clear()
        self.height_buffer.clear()
        self.width_buffer.clear()
        self.smoothed_height = 0
        self.smoothed_width = 0
        self.smoothed_center_x = 0
        self.smoothed_center_y = 0
    
    def handle_target_lost(self):
        """Xử lý khi mất target"""
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
        if frame is None:
            return frame
        
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