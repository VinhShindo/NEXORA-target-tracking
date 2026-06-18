"""
Object Detection Module using MobileNetV2
"""
import cv2
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from typing import List, Tuple, Optional
import time

class ObjectDetector:
    def __init__(self, model_path: str, config_path: str, use_onnx: bool = True):
        """
        Initialize object detector
        
        Args:
            model_path: Path to model file (.h5 or .onnx)
            config_path: Path to configuration file
            use_onnx: Use ONNX Runtime for inference
        """
        self.classes = ['person', 'bottle', 'chair', 'ball', 'bag']
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (224, 224)  # MobileNetV2 input size
        
        self.use_onnx = use_onnx
        if use_onnx:
            self.session = ort.InferenceSession(model_path)
        else:
            self.model = tf.keras.models.load_model(model_path)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize
        resized = cv2.resize(image, self.input_size)
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect objects in image
        
        Returns:
            List of detections: [{'bbox': [x1,y1,x2,y2], 'class': str, 'confidence': float}]
        """
        h, w = image.shape[:2]
        input_tensor = self.preprocess(image)
        
        # Run inference
        if self.use_onnx:
            outputs = self.session.run(None, {'input': input_tensor})
            boxes = outputs[0][0]
            scores = outputs[1][0]
            classes = outputs[2][0]
        else:
            predictions = self.model.predict(input_tensor, verbose=0)
            # Parse predictions (adjust based on your model output)
            boxes, scores, classes = self._parse_predictions(predictions)
        
        # Filter by confidence
        detections = []
        for i in range(len(boxes)):
            if scores[i] >= self.confidence_threshold:
                # Scale boxes back to original image size
                x1, y1, x2, y2 = boxes[i]
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': self.classes[int(classes[i])],
                    'class_id': int(classes[i]),
                    'confidence': float(scores[i])
                })
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[dict]) -> List[dict]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return detections
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []
    
    def _parse_predictions(self, predictions):
        """Parse model predictions - override based on model architecture"""
        # This is a placeholder - adjust based on your model
        return [], [], []