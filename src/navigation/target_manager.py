"""
Target Manager for selecting and managing target objects
"""
from typing import List, Dict, Optional
from collections import defaultdict
import time

class TargetManager:
    def __init__(self, target_class: Optional[str] = None):
        """
        Initialize target manager
        
        Args:
            target_class: Preferred target class (e.g., 'person')
        """
        self.target_class = target_class or 'person'
        self.current_target = None
        self.target_history = []
        self.max_history = 50
        self.last_update = time.time()
        self.track_persistence = 0.5  # seconds
        self.priority_classes = ['person', 'bottle', 'chair', 'ball', 'bag']
    
    def select_target(self, tracks: List[dict]) -> Optional[dict]:
        """
        Select the best target from tracked objects
        
        Args:
            tracks: List of tracked objects
        
        Returns:
            Selected target or None
        """
        if not tracks:
            self.current_target = None
            return None
        
        # Filter tracks by confidence and class preference
        valid_tracks = []
        for track in tracks:
            # Only consider confirmed tracks
            if track.get('track_id', -1) == -1:
                continue
            
            # Prefer target class
            if track.get('class_name') == self.target_class:
                valid_tracks.append((track, 1))  # Highest priority
            else:
                valid_tracks.append((track, 2))
        
        if not valid_tracks:
            # No tracks with target class, use any track
            valid_tracks = [(tracks[0], 1)]
        
        # Sort by priority and then by track_id (for consistency)
        valid_tracks.sort(key=lambda x: (x[1], x[0].get('track_id', 0)))
        
        # Check if we should keep previous target
        if self.current_target:
            # Check if previous target still exists
            current_id = self.current_target.get('track_id')
            if current_id is not None:
                for track in tracks:
                    if track.get('track_id') == current_id:
                        # Previous target still exists
                        # Check if it's been a while since last update
                        if time.time() - self.last_update < self.track_persistence:
                            return self.current_target
                        else:
                            # Target lost for too long, switch to new one
                            pass
        
        # Select new target
        selected_track = valid_tracks[0][0]
        self.current_target = selected_track
        self.last_update = time.time()
        
        # Add to history
        self.target_history.append(selected_track)
        if len(self.target_history) > self.max_history:
            self.target_history.pop(0)
        
        return selected_track
    
    def get_target_priority(self, class_name: str) -> int:
        """Get priority level for a class"""
        if class_name in self.priority_classes:
            return self.priority_classes.index(class_name)
        return len(self.priority_classes)
    
    def is_target_lost(self) -> bool:
        """Check if target has been lost"""
        if not self.current_target:
            return True
        return (time.time() - self.last_update) > self.track_persistence
    
    def clear_target(self):
        """Clear current target"""
        self.current_target = None
        self.last_update = time.time()