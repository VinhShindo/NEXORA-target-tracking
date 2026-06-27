
"""
Export YOLO model to ONNX for faster inference
Run: python scripts/export_onnx.py
"""

import sys
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_onnx():
    """Export YOLOv8 model to ONNX format"""
    model_path = BASE_DIR / "weights" / "yolov8n_person.pt"
    onnx_path = model_path.with_suffix('.onnx')
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Downloading default model...")
        model = YOLO('yolov8n.pt')
    else:
        logger.info(f"Loading model from {model_path}")
        model = YOLO(str(model_path))
    
    # Fuse model first
    logger.info("Fusing model...")
    model.fuse()
    
    # Export to ONNX
    logger.info(f"Exporting to ONNX: {onnx_path}")
    model.export(
        format='onnx',
        imgsz=416,  # Inference size
        half=False,
        int8=False,
        dynamic=False,
        simplify=True
    )
    
    # Move file
    import shutil
    src = Path('yolov8n.onnx')
    if src.exists():
        shutil.move(str(src), str(onnx_path))
        logger.info(f"ONNX model saved to {onnx_path}")
    
    logger.info("ONNX export completed!")

if __name__ == "__main__":
    export_onnx()