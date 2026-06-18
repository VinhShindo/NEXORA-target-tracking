"""
PID Controller for Robot Navigation
"""
import time
import math
from typing import Tuple

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, 
                 max_output: float = 1.0, min_output: float = -1.0,
                 max_integral: float = 0.5):
        """
        PID Controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            max_output: Maximum output value
            min_output: Minimum output value
            max_integral: Integral windup limit
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        self.max_integral = max_integral
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
    
    def update(self, error: float) -> float:
        """
        Update PID controller
        
        Args:
            error: Current error value
        
        Returns:
            Control output
        """
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt <= 0:
            dt = 0.001
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Clamp output
        output = max(self.min_output, min(self.max_output, output))
        
        # Update state
        self.prev_error = error
        self.prev_time = current_time
        
        return output

class RobotController:
    def __init__(self):
        """Initialize robot controller with PID for lateral and distance control"""
        # Lateral control PID (steering)
        self.lateral_pid = PIDController(
            kp=0.3, ki=0.01, kd=0.05,
            max_output=1.0, min_output=-1.0,
            max_integral=0.5
        )
        
        # Distance control PID (speed)
        self.distance_pid = PIDController(
            kp=0.2, ki=0.005, kd=0.02,
            max_output=1.0, min_output=-1.0,
            max_integral=0.3
        )
        
        # Target parameters
        self.target_bbox = None
        self.target_width = 0
        self.target_center = 0
        
        # Desired state
        self.desired_distance = 0.5  # meters
        self.desired_center = 0.5   # normalized screen center
    
    def compute_distance(self, bbox_width: int, frame_width: int) -> float:
        """
        Estimate distance from bounding box width
        
        Args:
            bbox_width: Width of bounding box in pixels
            frame_width: Width of frame in pixels
        
        Returns:
            Estimated distance (normalized)
        """
        # Simple distance estimation based on bounding box size
        normalized_width = bbox_width / frame_width
        return 1.0 / (normalized_width + 0.1)  # Inverse proportional
    
    def compute_control(self, detection: dict, frame_width: int, 
                        frame_height: int) -> Tuple[float, float]:
        """
        Compute control commands
        
        Args:
            detection: Detection result with 'bbox' field
            frame_width: Width of frame
            frame_height: Height of frame
        
        Returns:
            (linear_speed, angular_speed) tuple
        """
        if not detection:
            return 0.0, 0.0
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Compute normalized center (0 to 1)
        norm_center_x = center_x / frame_width
        
        # Compute distance
        distance = self.compute_distance(bbox_width, frame_width)
        
        # Compute errors
        lateral_error = norm_center_x - self.desired_center
        distance_error = distance - self.desired_distance
        
        # Adjust error scaling
        lateral_error = max(-0.5, min(0.5, lateral_error))
        distance_error = max(-0.5, min(0.5, distance_error))
        
        # Compute control outputs
        angular_speed = self.lateral_pid.update(-lateral_error)  # Negative to turn towards target
        linear_speed = self.distance_pid.update(-distance_error)
        
        # Scale speeds
        linear_speed = max(-0.5, min(0.5, linear_speed))
        angular_speed = max(-0.5, min(0.5, angular_speed))
        
        return linear_speed, angular_speed