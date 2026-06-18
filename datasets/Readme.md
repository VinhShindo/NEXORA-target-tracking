# Datasets for NEXORA Object Detection

## Dataset Overview
This directory contains training and validation datasets for the object detection model.

## Dataset Structure
```
datasets/
├── images/
│   ├── train/
│   │   ├── person/
│   │   ├── bottle/
│   │   ├── chair/
│   │   ├── ball/
│   │   └── bag/
│   └── val/
│       ├── person/
│       ├── bottle/
│       ├── chair/
│       ├── ball/
│       └── bag/
├── labels/
│   ├── train/
│   └── val/
├── annotations/
│   ├── train.json
│   └── val.json
├── data.yaml
└── README.md
```

## Classes
| ID | Class  |
|----|--------|
| 0  | person |
| 1  | bottle |
| 2  | chair  |
| 3  | ball   |
| 4  | bag    |

## Dataset Collection
- **Environment**: Indoor and outdoor
- **Lighting**: Various conditions (bright, dim, mixed)
- **Angles**: Multiple viewing angles
- **Format**: YOLO format (class x_center y_center width height)

## Data Augmentation
- Random horizontal flip
- Random brightness/contrast adjustment
- Random rotation (±15 degrees)
- Random scale (0.8-1.2)
- Mosaic augmentation

## Usage
```python
# Load dataset configuration
import yaml
with open('data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)
```

## Download Instructions
1. Collect images using webcam or mobile phone
2. Annotate using LabelImg or Roboflow
3. Split into train/val (80/20 ratio)
4. Save in YOLO format

## Data Statistics
- **Total images**: ~5000
- **Training images**: ~4000
- **Validation images**: ~1000
- **Objects per image**: 1-5
```