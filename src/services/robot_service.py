"""
Robot Service - Kết hợp Area + Delta Y, tối ưu chống gai và thêm cơ chế phanh gấp
- Tăng Kp lên 50.0 để đạt tốc độ tối đa khi ở xa.
- Vô hiệu hóa Stability Score khi ở xa > 0.5m.
- Dùng Median Filter + Rate Limiter để diệt gai cho Motor Speed.
- Thêm cơ chế Emergency Brake dựa trên Delta Y và khoảng cách hiện tại.
"""

import logging
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from collections import deque
import os
from pathlib import Path

from controllers.steering_pid import SteeringPID
from controllers.distance_pid import DistancePID

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "robot_distance_log.txt"

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
    estimated_distance: float = 0.0
    fps: float = 0.0
    last_update: float = 0.0
    target_distance: float = 1.0

class RobotService:
    def __init__(self, config: Optional[Dict] = None):
        self.steering_pid = SteeringPID()
        self.distance_pid = DistancePID()
        
        # === CẤU HÌNH KHOẢNG CÁCH ===
        self.target_distance = 1.0
        self.distance_k_factor = 405.0
        self.distance_smooth_factor = 0.2
        self.smoothed_distance = 0.0
        self.max_distance = 10.0
        
        self.telemetry = RobotTelemetry()
        self.telemetry.target_distance = self.target_distance
        
        self.state = RobotState.IDLE
        self.direction = "STOP"
        self.is_target_lost = False
        
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.max_linear = 0.5
        self.max_angular = 1.0
        
        # === BỘ LỌC NHIỄU ===
        self.area_buffer = deque(maxlen=7)
        self.center_y_buffer = deque(maxlen=7)
        self.center_x_buffer = deque(maxlen=7)
        self.stability_score = 1.0
        
        self.smooth_factor = 0.4
        self.smoothed_area = 0.0
        self.smoothed_center_y = 0.0
        self.smoothed_center_x = 0.0
        
        # Bộ lọc median cho distance (buffer 10)
        self.distance_buffer = deque(maxlen=10)
        self.distance_median = 0.0
        
        self.prev_center_y = 0.0
        self.delta_center_y = 0.0
        
        self.max_change_area = 5000.0
        self.area_variance_threshold = 3000.0
        
        # === DEADZONE ===
        self.deadzone_steering_px = 20.0
        self.deadzone_distance_m = 0.1
        
        # === PID DISTANCE ===
        self.distance_pid.kp = 50.0     # Tăng lên 50.0 để khi ở xa bị chặn cứng ở 100%
        self.distance_pid.ki = 0.0
        self.distance_pid.kd = 0.0
        
        # === BỘ LỌC MOTOR SPEED MỚI (Median + Rate Limit) ===
        self.motor_speed_buffer = deque(maxlen=7)
        self.prev_motor_speed = 0.0
        self.max_motor_speed_change = 15.0  # Tối đa 15% thay đổi mỗi frame (mượt mà)
        
        # Rate limit cho error_dist
        self.prev_error_dist = 0.0
        self.error_dist_rate_limit = 0.5
        
        if config:
            self._apply_config(config)
    
    def _apply_config(self, config: Dict):
        if 'steering_pid' in config:
            steer = config['steering_pid']
            self.steering_pid.kp = steer.get('kp', 0.05)
            self.steering_pid.ki = steer.get('ki', 0.001)
            self.steering_pid.kd = steer.get('kd', 0.01)
            self.steering_pid.integral_limit = 5.0
        if 'distance_pid' in config:
            dist = config['distance_pid']
            if 'kp' in dist: self.distance_pid.kp = dist['kp']
            if 'ki' in dist: self.distance_pid.ki = dist['ki']
            if 'kd' in dist: self.distance_pid.kd = dist['kd']
        self.target_distance = config.get('target_distance', 1.0)
        self.distance_k_factor = config.get('distance_k_factor', 405.0)
        self.deadzone_steering_px = config.get('deadzone_steering_px', 20.0)
        self.deadzone_distance_m = config.get('deadzone_distance_m', 0.1)
    
    def update_pid_params(self, steering=None, distance=None, target_distance=None):
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
        if target_distance is not None:
            self.target_distance = target_distance
            self.telemetry.target_distance = target_distance
    
    def start_following(self):
        self.state = RobotState.FOLLOWING
        self.telemetry.tracking_state = "FOLLOWING"
        self.steering_pid.reset()
        self.distance_pid.reset()
        self.area_buffer.clear()
        self.center_y_buffer.clear()
        self.center_x_buffer.clear()
        self.distance_buffer.clear()
        self.motor_speed_buffer.clear()
        self.smoothed_distance = 0.0
        self.distance_median = 0.0
        self.smoothed_area = 0.0
        self.smoothed_center_x = 0.0
        self.smoothed_center_y = 0.0
        self.prev_center_y = 0.0
        self.prev_motor_speed = 0.0
        self.prev_error_dist = 0.0
    
    def _compute_stability_score(self, error_dist: float) -> float:
        # QUAN TRỌNG: Nếu đang ở xa mục tiêu > 0.5m, Không được kìm hãm tốc độ
        if abs(error_dist) > 0.5:
            return 1.0
        
        if len(self.area_buffer) < 3:
            return 1.0
        arr = np.array(self.area_buffer)
        variance = np.var(arr)
        if variance < self.area_variance_threshold:
            return 1.0
        else:
            score = 1.0 - min(0.5, variance / (self.area_variance_threshold * 3))
            return max(0.7, score)
    
    def _apply_rate_limit(self, new_value: float, prev_value: float, max_change: float) -> float:
        diff = new_value - prev_value
        if diff > max_change:
            return prev_value + max_change
        elif diff < -max_change:
            return prev_value - max_change
        else:
            return new_value
    
    def _write_log(self, target_dist, current_dist, error_dist, delta_y, stability, error_adjusted, pid_out, raw_motor, final_motor):
        action = "STOP"
        if final_motor > 0.5: action = "FORWARD"
        elif final_motor < -0.5: action = "BACKWARD"
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(f"{time.time():.3f},{target_dist:.3f},{current_dist:.3f},{error_dist:.3f},{delta_y:.1f},{stability:.3f},{error_adjusted:.3f},{pid_out:.3f},{raw_motor:.3f},{final_motor:.3f},{action}\n")
        except Exception:
            pass
    
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
        raw_width = x2 - x1
        raw_height = y2 - y1
        raw_area = raw_width * raw_height
        
        # === LỌC NHIỄU BẰNG AREA ===
        self.area_buffer.append(raw_area)
        self.center_y_buffer.append(raw_center_y)
        self.center_x_buffer.append(raw_center_x)
        
        if len(self.area_buffer) >= 3:
            areas = sorted(self.area_buffer)
            median_area = areas[len(areas)//2]
            centers_y = sorted(self.center_y_buffer)
            median_center_y = centers_y[len(centers_y)//2]
            centers_x = sorted(self.center_x_buffer)
            median_center_x = centers_x[len(centers_x)//2]
        else:
            median_area = raw_area
            median_center_y = raw_center_y
            median_center_x = raw_center_x
        
        # EMA Smoothing cho area và tâm
        if self.smoothed_area == 0:
            self.smoothed_area = median_area
            self.smoothed_center_y = median_center_y
            self.smoothed_center_x = median_center_x
        else:
            self.smoothed_area = self.smoothed_area * (1 - self.smooth_factor) + median_area * self.smooth_factor
            self.smoothed_center_y = self.smoothed_center_y * (1 - self.smooth_factor) + median_center_y * self.smooth_factor
            self.smoothed_center_x = self.smoothed_center_x * (1 - self.smooth_factor) + median_center_x * self.smooth_factor
        
        # === ƯỚC LƯỢNG KHOẢNG CÁCH ===
        if self.smoothed_area > 100:
            raw_distance = self.distance_k_factor / np.sqrt(self.smoothed_area)
        else:
            raw_distance = self.max_distance
        
        # Làm mịn khoảng cách bằng EMA
        if self.smoothed_distance == 0:
            self.smoothed_distance = raw_distance
        else:
            self.smoothed_distance = self.smoothed_distance * (1 - self.distance_smooth_factor) + raw_distance * self.distance_smooth_factor
        
        # Bộ lọc median cho distance
        self.distance_buffer.append(self.smoothed_distance)
        if len(self.distance_buffer) >= 5:
            sorted_vals = sorted(self.distance_buffer)
            self.distance_median = sorted_vals[len(sorted_vals)//2]
        else:
            self.distance_median = self.smoothed_distance
        
        # Sử dụng giá trị median làm distance cuối cùng
        final_distance = self.distance_median
        
        # === DELTA Y ===
        if self.prev_center_y > 0:
            self.delta_center_y = self.smoothed_center_y - self.prev_center_y
        self.prev_center_y = self.smoothed_center_y
        
        # === PID STEERING ===
        error_x = self.smoothed_center_x - self.telemetry.frame_center_x
        if abs(error_x) < self.deadzone_steering_px:
            error_x = 0.0
            self.steering_pid.reset()
        
        steering_output = self.steering_pid.update(error_x)
        self.telemetry.error_x = error_x
        self.telemetry.pid_steering_output = steering_output
        servo_angle = max(45.0, min(135.0, 90.0 + steering_output))
        self.telemetry.servo_angle = servo_angle
        
        # === PID DISTANCE ===
        error_dist = final_distance - self.target_distance  # Dương: xa hơn (tiến)
        
        # Rate limit cho error_dist
        if self.prev_error_dist != 0:
            error_dist = self._apply_rate_limit(error_dist, self.prev_error_dist, self.error_dist_rate_limit)
        self.prev_error_dist = error_dist
        
        delta_adjust = 1.0
        if self.delta_center_y > 5.0 and error_dist > 0:
            delta_adjust = 1.3
        elif self.delta_center_y < -5.0 and error_dist < 0:
            delta_adjust = 1.3
        elif abs(self.delta_center_y) < 2.0:
            delta_adjust = 0.8
        else:
            delta_adjust = 1.0
            
        # Tính Stability Score ưu tiên tốc độ khi ở xa
        self.stability_score = self._compute_stability_score(error_dist)
        
        error_dist_adjusted = error_dist * delta_adjust * self.stability_score
        
        if abs(error_dist) < self.deadzone_distance_m:
            error_dist_adjusted = 0.0
            self.distance_pid.reset()
        
        distance_output = self.distance_pid.kp * error_dist_adjusted
        self.telemetry.distance_error = error_dist
        self.telemetry.pid_distance_output = distance_output
        
        raw_motor_speed = max(-100.0, min(100.0, distance_output))
        
        # === CƠ CHẾ PHANH GẤP (EMERGENCY BRAKE) ===
        # Khi người lao nhanh về phía xe (delta_center_y > 8.0) và khoảng cách đã gần
        if self.delta_center_y > 8.0 and final_distance < 1.5:
            if final_distance > 0.9:
                raw_motor_speed = 0.0  # Phanh gấp, dừng khẩn cấp
            else:
                raw_motor_speed = -30.0  # Cực kỳ gần, lùi gấp để tránh va chạm
        
        # === BỘ LỌC CHỐNG GAI MẠNH MẼ (Median + Rate Limit) ===
        self.motor_speed_buffer.append(raw_motor_speed)
        # 1. Lấy Median làm giá trị tham chiếu
        if len(self.motor_speed_buffer) >= 5:
            sorted_vals = sorted(self.motor_speed_buffer)
            median_motor = sorted_vals[len(sorted_vals)//2]
        else:
            median_motor = raw_motor_speed
            
        # 2. Rate Limiter (Cưỡng chế tăng/giảm từ từ)
        if self.prev_motor_speed != 0:
            motor_speed = self._apply_rate_limit(median_motor, self.prev_motor_speed, self.max_motor_speed_change)
        else:
            motor_speed = median_motor
            
        self.prev_motor_speed = motor_speed
        motor_speed = max(-100.0, min(100.0, motor_speed))
        self.telemetry.motor_speed = motor_speed
        
        # === CHẾ ĐỘ ỔN ĐỊNH ===
        if abs(error_x) < 5 and abs(error_dist) < 0.2:
            motor_speed *= 0.8
        
        self.is_target_lost = False
        self.telemetry.tracking_state = "FOLLOWING"
        
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
        
        if motor_speed > 0:
            self.linear_velocity = self.max_linear * (motor_speed / 100.0)
        elif motor_speed < 0:
            self.linear_velocity = -self.max_linear * (abs(motor_speed) / 100.0)
        else:
            self.linear_velocity = 0.0
        self.angular_velocity = max(-self.max_angular, min(self.max_angular, -steering_output * 0.01))
        
        self.telemetry.target_id = track.get('track_id')
        self.telemetry.target_center_x = self.smoothed_center_x
        self.telemetry.target_center_y = self.smoothed_center_y
        self.telemetry.bbox_width = raw_width
        self.telemetry.bbox_height = raw_height
        self.telemetry.bbox_area = self.smoothed_area
        self.telemetry.estimated_distance = final_distance
    
    def _set_lost_state(self):
        self.is_target_lost = True
        self.state = RobotState.TARGET_LOST
        self.telemetry.tracking_state = "TARGET LOST"
        self.telemetry.target_id = None
        self.telemetry.error_x = 0.0
        self.telemetry.distance_error = 0.0
        self.telemetry.servo_angle = 90.0
        self.telemetry.motor_speed = 0.0
        self.telemetry.estimated_distance = 0.0
        self.direction = "STOP"
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.area_buffer.clear()
        self.center_y_buffer.clear()
        self.center_x_buffer.clear()
        self.distance_buffer.clear()
        self.motor_speed_buffer.clear()
        self.smoothed_distance = 0.0
        self.distance_median = 0.0
        self.smoothed_area = 0.0
        self.smoothed_center_x = 0.0
        self.smoothed_center_y = 0.0
        self.prev_center_y = 0.0
        self.prev_motor_speed = 0.0
        self.prev_error_dist = 0.0
        self.stability_score = 1.0
    
    def handle_target_lost(self):
        self.is_target_lost = True
        self.state = RobotState.SEARCHING
        self.telemetry.tracking_state = "SEARCHING"
        self.direction = "SEARCHING"
        self.linear_velocity = 0.1
        self.angular_velocity = 0.3
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
            'distance': round(self.telemetry.estimated_distance, 2),
            'offset_x': round(self.telemetry.error_x, 2),
            'target_id': self.telemetry.target_id,
            'target_lost': self.is_target_lost,
            'linear_velocity': round(self.linear_velocity, 3),
            'angular_velocity': round(self.angular_velocity, 3),
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
                    f"DIST(m): {t.estimated_distance:.2f}",
                    f"FPS: {t.fps:.1f}"
                ]
                for i, txt in enumerate(texts):
                    cv2.putText(frame, txt, (10, 30 + i*25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO TARGET", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception:
            pass
        return frame