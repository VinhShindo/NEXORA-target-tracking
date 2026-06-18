"""
NEXORA Main Application
"""
import cv2
import yaml
import time
import numpy as np
from pathlib import Path
import sys

from detection.detector import ObjectDetector
from tracking.tracker import ByteTracker
from control.pid_controller import RobotController
from navigation.target_manager import TargetManager
from communication.esp32_comm import ESP32Comm
from ui.display import Display

class NEXORA:
    def __init__(self, config_path: str):
        """
        Initialize NEXORA system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.running = False
        
        # Initialize components
        self._init_components()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_components(self):
        """Initialize all system components"""
        print("Initializing NEXORA components...")
        
        # Camera
        self.camera = cv2.VideoCapture(
            self.config['camera']['device_id']
        )
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 
                       self.config['camera']['width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 
                       self.config['camera']['height'])
        self.camera.set(cv2.CAP_PROP_FPS, 
                       self.config['camera']['fps'])
        
        # Object Detector
        self.detector = ObjectDetector(
            model_path=self.config['detection']['model_path'],
            config_path=self.config['detection']['config_path'],
            use_onnx=True
        )
        self.detector.confidence_threshold = self.config['detection']['confidence_threshold']
        self.detector.nms_threshold = self.config['detection']['nms_threshold']
        
        # Tracker
        self.tracker = ByteTracker(
            max_lost=self.config['tracking']['max_lost'],
            iou_threshold=self.config['tracking']['iou_threshold']
        )
        
        # Target Manager
        self.target_manager = TargetManager(
            target_class=self.config['navigation']['target_class']
        )
        
        # Robot Controller
        self.robot_controller = RobotController()
        
        # Communication (ESP32)
        self.comm = ESP32Comm(
            port=self.config['communication']['port'],
            baudrate=self.config['communication']['baudrate'],
            timeout=self.config['communication']['timeout']
        )
        self.comm.connect()
        
        # Display
        self.display = Display(
            window_name=self.config['ui']['window_name'],
            show_fps=self.config['ui']['show_fps']
        )
        
        print("All components initialized!")
    
    def run(self):
        """Main loop"""
        self.running = True
        print("Starting NEXORA system...")
        print("Press 'q' to quit")
        
        while self.running:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Track objects
            tracks = self.tracker.update(detections)
            
            # Select target
            target = self.target_manager.select_target(tracks)
            
            # Compute control commands
            if target:
                linear_speed, angular_speed = self.robot_controller.compute_control(
                    target,
                    frame.shape[1],
                    frame.shape[0]
                )
                
                # Send motor commands
                left_speed = linear_speed - angular_speed
                right_speed = linear_speed + angular_speed
                self.comm.send_motor_command(left_speed, right_speed)
            else:
                # Stop robot if no target
                self.comm.send_stop()
            
            # Display
            display_frame = frame.copy()
            
            # Draw tracks
            display_frame = self.display.draw_tracks(display_frame, tracks)
            
            # Draw target
            if target:
                display_frame = self.display.draw_target(display_frame, target)
            
            # Draw info
            info = {
                'Tracks': len(tracks),
                'Detections': len(detections),
                'Target ID': target.get('track_id', 'None') if target else 'None',
                'Status': 'Following' if target else 'Searching'
            }
            display_frame = self.display.draw_info(display_frame, info)
            
            # Show frame
            if not self.display.show_frame(display_frame):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        
        # Stop robot
        self.comm.send_stop()
        
        # Disconnect ESP32
        self.comm.disconnect()
        
        # Release camera
        self.camera.release()
        
        # Close display
        self.display.close()
        
        print("NEXORA system shutdown complete")

def main():
    """Main entry point"""
    # Get config path
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Create and run NEXORA
    nexora = NEXORA(str(config_path))
    nexora.run()

if __name__ == "__main__":
    main()