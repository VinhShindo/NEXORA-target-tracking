"""
UI Display Module for Real-Time Visualization
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
import time

class Display:
    def __init__(self, window_name: str = "NEXORA", show_fps: bool = True):
        """
        Initialize display module
        
        Args:
            window_name: Name of display window
            show_fps: Show FPS counter
        """
        self.window_name = window_name
        self.show_fps = show_fps
        self.fps = 0.0
        self.last_time = time.time()
        self.frame_count = 0
        self.colors = self._generate_colors(100)
    
    def _generate_colors(self, num_colors: int) -> List[tuple]:
        """Generate distinct colors for different tracks"""
        colors = []
        for i in range(num_colors):
            hue = i * 360 / num_colors
            colors.append(self._hsv_to_rgb(hue, 0.8, 0.8))
        return colors
    
    def _hsv_to_rgb(self, hue: float, sat: float, val: float) -> tuple:
        """Convert HSV to RGB"""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360, sat, val)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def draw_bbox(self, image: np.ndarray, bbox: List[int], 
                  label: str = "", color: tuple = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """Draw bounding box with label"""
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if label:
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(image, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def draw_tracks(self, image: np.ndarray, tracks: List[dict],
                    show_id: bool = True) -> np.ndarray:
        """Draw tracked objects"""
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            class_name = track.get('class_name', 'unknown')
            class_id = track.get('class_id', 0)
            
            # Get color for this track
            color = self.colors[track_id % len(self.colors)]
            
            # Prepare label
            label = f"ID:{track_id}"
            if show_id and class_name:
                label += f" {class_name}"
            elif show_id:
                label = f"ID:{track_id}"
            elif class_name:
                label = class_name
            else:
                label = "Object"
            
            # Draw bounding box
            self.draw_bbox(image, bbox, label, color)
        
        return image
    
    def draw_target(self, image: np.ndarray, target: dict,
                    color: tuple = (0, 255, 255)) -> np.ndarray:
        """Draw target with special highlighting"""
        if not target:
            return image
        
        bbox = target['bbox']
        
        # Draw target with thicker border and different color
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Draw crosshairs at center
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        size = 10
        cv2.line(image, (center_x - size, center_y), (center_x + size, center_y), color, 2)
        cv2.line(image, (center_x, center_y - size), (center_x, center_y + size), color, 2)
        
        # Draw "TARGET" label
        label = "TARGET"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (bbox[0], bbox[1] - text_h - 15), 
                     (bbox[0] + text_w + 15, bbox[1]), color, -1)
        cv2.putText(image, label, (bbox[0] + 5, bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return image
    
    def draw_info(self, image: np.ndarray, info: dict) -> np.ndarray:
        """Draw information overlay"""
        y_offset = 30
        x_offset = 10
        line_height = 25
        
        # Create semi-transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (x_offset, y_offset - 20), 
                     (x_offset + 250, y_offset + len(info) * line_height + 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Draw info text
        y = y_offset
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(image, text, (x_offset + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += line_height
        
        return image
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        return self.fps
    
    def show_frame(self, image: np.ndarray) -> None:
        """Display frame"""
        if self.show_fps:
            fps_text = f"FPS: {self.update_fps():.1f}"
            cv2.putText(image, fps_text, (image.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, image)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
    
    def close(self):
        """Close display window"""
        cv2.destroyWindow(self.window_name)