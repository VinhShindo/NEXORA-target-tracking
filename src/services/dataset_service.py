"""
Dataset Service - Dataset collection and management
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import Dict, List, Optional
import shutil

logger = logging.getLogger(__name__)

class DatasetService:
    """Service for dataset collection and management"""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1] / "output"
        self.snapshot_dir = self.base_dir / "snapshots"
        self.recording_dir = self.base_dir / "recordings"
        self.dataset_dir = self.base_dir / "datasets"
        
        self.snapshot_count = 0
        self.recording_count = 0
        self.total_images = 0
        self.total_videos = 0
        
        self._load_stats()
    
    def _load_stats(self):
        """Load dataset statistics"""
        try:
            stats_file = self.dataset_dir / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    self.total_images = stats.get('images', 0)
                    self.total_videos = stats.get('videos', 0)
        except:
            pass
    
    def _save_stats(self):
        """Save dataset statistics"""
        try:
            stats_file = self.dataset_dir / "stats.json"
            stats = {
                'images': self.total_images,
                'videos': self.total_videos,
                'last_updated': datetime.now().isoformat()
            }
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
    
    def capture_frame(self, frame: np.ndarray) -> Path:
        """Capture single frame"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"capture_{timestamp}.jpg"
            filepath = self.snapshot_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            self.snapshot_count += 1
            self.total_images += 1
            self._save_stats()
            
            logger.info(f"📸 Frame captured: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            raise
    
    def save_recording(self, frames: List[np.ndarray]) -> Path:
        """Save recorded video"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            filepath = self.recording_dir / filename
            
            if frames:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(filepath), fourcc, 30.0, (width, height))
                
                for frame in frames:
                    out.write(frame)
                out.release()
                
                self.recording_count += 1
                self.total_videos += 1
                self._save_stats()
                
                logger.info(f"🎬 Recording saved: {filename}")
                return filepath
            
            raise ValueError("No frames to save")
            
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            raise
    
    def export_dataset(self) -> str:
        """Export dataset in YOLO format"""
        try:
            export_dir = self.dataset_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            images_dir = export_dir / "images"
            images_dir.mkdir(exist_ok=True)
            labels_dir = export_dir / "labels"
            labels_dir.mkdir(exist_ok=True)
            
            # Copy snapshots
            for img_file in self.snapshot_dir.glob("*.jpg"):
                shutil.copy2(img_file, images_dir / img_file.name)
            
            # Create data.yaml
            data_yaml = export_dir / "data.yaml"
            with open(data_yaml, 'w') as f:
                f.write(f"""
# NEXORA Dataset
path: {export_dir}
train: images
val: images

nc: 2
names: ['person', 'ball']
""")
            
            logger.info(f"📦 Dataset exported to: {export_dir}")
            return str(export_dir)
            
        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'images': self.total_images,
            'videos': self.total_videos,
            'snapshots': self.snapshot_count,
            'recordings': self.recording_count,
            'storage_used': self._get_storage_usage()
        }
    
    def _get_storage_usage(self) -> float:
        """Get storage usage in MB"""
        try:
            total_size = 0
            for path in [self.snapshot_dir, self.recording_dir, self.dataset_dir]:
                if path.exists():
                    for f in path.rglob('*'):
                        if f.is_file():
                            total_size += f.stat().st_size
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0