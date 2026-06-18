# NEXORA Model Repository

## Models Overview
This directory contains trained models for object detection.

## Model Files
```
models/
├── mobilenetv2/
│   ├── model.h5
│   ├── model.onnx
│   ├── model.tflite
│   └── config.yaml
├── weights/
│   ├── best_weights.h5
│   ├── checkpoint.h5
│   └── training_log.csv
└── README.md
```

## Model Specifications

### MobileNetV2 Object Detection
- **Architecture**: MobileNetV2 + SSD
- **Input Size**: 224x224
- **Classes**: 5 (person, bottle, chair, ball, bag)
- **Framework**: TensorFlow 2.x
- **Format**: H5 and ONNX

### Performance Metrics
| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.78 |
| mAP@0.5:0.95 | 0.52 |
| Inference Time (CPU) | 45ms |
| Inference Time (ONNX) | 35ms |
| Model Size | 14MB |

## Model Conversion

### Convert H5 to ONNX
```python
import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model('model.h5')

# Convert to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
```

### Convert ONNX to TensorRT
```bash
trtexec --onnx=model.onnx --saveEngine=model.trt
```

### Convert H5 to TFLite
```python
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Training

### Training Script
```python
# src/detection/train.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Load dataset
train_dataset = ...
val_dataset = ...

# Create model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Add detection head
...

# Train
model.compile(...)
model.fit(train_dataset, validation_data=val_dataset, epochs=50)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 100 |
| Optimizer | Adam |
| Loss | Focal Loss + L1 |

## Evaluation

### Run Evaluation
```python
# src/tests/test_model.py
from detection.detector import ObjectDetector

detector = ObjectDetector('models/model.onnx', 'configs/config.yaml')
metrics = detector.evaluate(dataset)
print(metrics)
```

### Expected Output
```
{
    'mAP@0.5': 0.78,
    'mAP@0.75': 0.62,
    'mAP@0.5:0.95': 0.52,
    'class_AP': {
        'person': 0.85,
        'bottle': 0.72,
        'chair': 0.68,
        'ball': 0.79,
        'bag': 0.64
    }
}
```

## Download Pre-trained Models

### Google Drive
```
https://drive.google.com/drive/folders/...
```

### Hugging Face
```
https://huggingface.co/nexora/mobilenetv2-detection
```

## Model Optimization

### Quantization
```python
# INT8 Quantization
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
quantized_model = converter.convert()
```

### Pruning
```python
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=2000,
        end_step=10000
    )
}

model = prune_low_magnitude(model, **pruning_params)
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | 2024-01 | Initial release |
| v1.1.0 | 2024-02 | Added ONNX support |
| v1.2.0 | 2024-03 | Quantization support |

## Troubleshooting

### Model Not Loading
```bash
# Check file permissions
ls -l models/
chmod 644 models/*

# Verify format
file models/model.onnx
```

### Poor Performance
1. Use ONNX or TensorRT for faster inference
2. Resize images to input size
3. Increase confidence threshold

### Memory Issues
```python
# Use smaller batch size
model.predict(x, batch_size=1)

# Use TF Lite
model = tf.lite.Interpreter(model_path='model.tflite')
```