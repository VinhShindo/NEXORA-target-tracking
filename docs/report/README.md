# NEXORA - AI Target Following Robot System

## 🚀 Giới thiệu

NEXORA là hệ thống điều khiển robot theo dõi mục tiêu thời gian thực sử dụng AI, được thiết kế để chạy trên máy tính (PC/Laptop) và giao tiếp với vi điều khiển ESP32 để điều khiển robot vật lý. Hệ thống tích hợp YOLOv8 để phát hiện đối tượng, ByteTrack để theo dõi, và PID Controller để điều khiển robot.

## 📋 Tính năng chính

### 🤖 Điều khiển Robot
- **PID Steering Control**: Điều khiển hướng robot dựa trên vị trí mục tiêu
- **PID Distance Control**: Điều khiển khoảng cách robot với mục tiêu
- **ESP32 Integration**: Gửi lệnh điều khiển qua WebSocket đến ESP32
- **Real-time Telemetry**: Hiển thị servo angle, motor speed, error tracking

### 👁️ Computer Vision
- **YOLOv8 Detection**: Phát hiện người (person) và bóng (ball) với độ chính xác cao
- **ByteTrack Tracking**: Theo dõi nhiều đối tượng với ID ổn định
- **Feature Database**: Lưu đặc điểm của đối tượng để nhận diện lại sau khi mất
- **Kalman Filter**: Dự đoán vị trí khi đối tượng bị che khuất

### 📊 Dashboard Real-time
- **Live Video Stream**: Hiển thị video với bounding box và confidence
- **Robot Control Panel**: Điều khiển và giám sát trạng thái robot
- **PID Tuning**: Điều chỉnh PID parameters trực tiếp trên dashboard
- **Robot Telemetry**: Hiển thị chi tiết các thông số steering, distance, system
- **Dataset Management**: Capture, record, export dataset cho training

### 📡 Kết nối IoT
- **ESP32 WebSocket**: Giao tiếp real-time với ESP32 để điều khiển motor và servo
- **Manual Control**: Điều khiển robot thủ công qua dashboard
- **Emergency Stop**: Dừng khẩn cấp robot
- **Command History**: Lưu lịch sử lệnh đã gửi

## 🛠️ Công nghệ sử dụng

### Backend
- **FastAPI** - Web Framework hiệu năng cao
- **OpenCV** - Xử lý video và image
- **Ultralytics YOLOv8** - Object Detection
- **ByteTrack (self-implemented)** - Multi-object tracking với Kalman Filter
- **WebSocket** - Real-time communication
- **PID Controller** - Điều khiển robot

### Frontend
- **HTML5 + CSS3** - Giao diện Dark Theme
- **JavaScript** - WebSocket client và UI logic
- **Bootstrap 5** - Responsive layout

### Hardware Interface
- **ESP32** - Vi điều khiển nhận lệnh điều khiển
- **Motor Driver** - Điều khiển động cơ DC
- **Servo** - Điều khiển hướng

## 📁 Cấu trúc thư mục

```
NEXORA-target-tracking/
├── apps/
│   ├── app.py                      # Main FastAPI application
│   ├── configs/
│   │   └── config.yaml             # System configuration
│   ├── controllers/
│   │   ├── steering_pid.py         # Steering PID controller
│   │   └── distance_pid.py         # Distance PID controller
│   ├── services/
│   │   ├── camera_service.py       # Camera management
│   │   ├── detection_service.py    # YOLOv8 detection
│   │   ├── tracking_service.py     # ByteTrack + Feature Database
│   │   ├── robot_service.py        # Robot control + PID
│   │   └── dataset_service.py      # Dataset management
│   ├── templates/
│   │   └── index.html              # Dashboard UI
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css           # Custom styles
│   │   └── js/
│   │       └── dashboard.js        # Frontend logic
│   ├── weights/
│   │   └── yolov8n_person.pt       # YOLO model weights
│   └── output/
│       ├── snapshots/              # Captured frames
│       ├── recordings/             # Recorded videos
│       ├── datasets/               # Exported datasets
│       └── logs/                   # System logs
└── logs/
    └── telemetry.csv               # Telemetry data logging
```

## 🚀 Cài đặt và chạy

### 1. Clone repository
```bash
git clone <repository-url>
cd NEXORA-target-tracking
cd apps
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Tải model YOLOv8
```bash
# Tải model pretrained
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Copy vào thư mục weights
cp yolov8n.pt apps/weights/yolov8n_person.pt
```

### 5. Cấu hình (configs/config.yaml)
```yaml
camera:
  id: 0  # 0 cho camera laptop, 1 cho webcam USB
  width: 640
  height: 480
  fps: 30
  use_mjpeg: true

detection:
  model_path: weights/yolov8n_person.pt
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: cpu  # hoặc 'cuda' nếu có GPU
  process_every_n_frames: 1
  inference_size: 320

tracking:
  track_thresh: 0.4
  high_thresh: 0.6
  match_thresh: 0.7

robot:
  frame_width: 640
  frame_height: 480
  pid:
    kp: 0.05   # Steering Proportional
    ki: 0.001  # Steering Integral
    kd: 0.01   # Steering Derivative
  distance_pid:
    kp: 0.1    # Distance Proportional
    ki: 0.002  # Distance Integral
    kd: 0.02   # Distance Derivative
  target_height: 150  # Target height in pixels

esp32:
  send_interval: 0.1  # Send command every 100ms
  timeout: 1.0
```

### 6. Chạy ứng dụng
```bash
cd apps
python app.py
```

### 7. Truy cập dashboard
Mở trình duyệt và truy cập: `http://localhost:8000`

## 📡 API Endpoints

### System
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard trang chính |
| `/api/status` | GET | Trạng thái hệ thống |
| `/api/config` | GET | Cấu hình hiện tại |

### Camera
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/camera/start` | POST | Khởi động camera |
| `/api/camera/stop` | POST | Dừng camera |

### Detection & Tracking
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detection/start` | POST | Bắt đầu detection |
| `/api/detection/stop` | POST | Dừng detection |
| `/api/target/select` | POST | Chọn target để theo dõi |
| `/api/target/release` | POST | Giải phóng target |

### Robot Control
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/robot/status` | GET | Lấy trạng thái robot |
| `/api/robot/pid` | POST | Cập nhật PID parameters |
| `/api/follow/start` | POST | Bắt đầu bám target |
| `/api/follow/stop` | POST | Dừng bám target |

### ESP32 Communication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/esp32/status` | GET | Trạng thái kết nối ESP32 |
| `/api/esp32/command` | GET | Lệnh cuối cùng đã gửi |
| `/api/esp32/command/manual` | POST | Gửi lệnh điều khiển thủ công |
| `/api/esp32/command/stop` | POST | Dừng khẩn cấp |

### Dataset Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dataset/capture` | POST | Chụp ảnh |
| `/api/dataset/record/start` | POST | Bắt đầu ghi hình |
| `/api/dataset/record/stop` | POST | Dừng ghi hình |
| `/api/dataset/export` | POST | Xuất dataset |
| `/api/dataset/stats` | GET | Thống kê dataset |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `/ws/frame` | Video stream + tracking data |
| `/ws/metrics` | System metrics |
| `/ws/esp32` | ESP32 control commands |

## 🔌 WebSocket Communication

### ESP32 Command JSON (Tối ưu hóa)
```json
{
  "cmd": "ctrl",
  "ts": 1700000000123,
  "motor": 25.5,      // Tốc độ motor (-100 -> 100)
  "servo": 92.5,      // Góc servo (45 -> 135)
  "dir": "FORWARD",   // Hướng: FORWARD, TURN_LEFT, TURN_RIGHT, STOP
  "state": "FOLLOWING",
  "target": 1,        // Target ID
  "err_x": 2.5,       // Lỗi hướng (pixels)
  "err_d": -3.0,      // Lỗi khoảng cách (pixels)
  "dist": 1.25        // Khoảng cách ước tính (m)
}
```

### Frame Data JSON (Dashboard)
```json
{
  "type": "frame",
  "fps": 12,
  "inference_time": 47.0,
  "tracks": [
    {
      "track_id": 1,
      "bbox": [120, 150, 520, 545],
      "class": "person",
      "confidence": 0.85,
      "lost": 0
    }
  ],
  "selected_target": 1,
  "target": {
    "id": 1,
    "class": "person",
    "confidence": 0.88,
    "status": "ACTIVE",
    "bbox": [120, 150, 520, 545]
  },
  "robot": {
    "state": "TURN RIGHT",
    "direction": "TURN RIGHT",
    "servo_angle": 93.3,
    "motor_speed": -30.5
  },
  "robot_telemetry": {
    "error_x": 51.34,
    "distance_error": -244.86,
    "pid_steering_output": 3.297,
    "pid_distance_output": -30.539
  },
  "esp32_connected": true
}
```

## 🎯 Hướng dẫn sử dụng

### 1. Kết nối Camera
- Nhấn **"START CAMERA"** để mở camera
- Kiểm tra video stream hiển thị

### 2. Khởi tạo Detection
- Nhấn **"DETECT"** để load model và bắt đầu phát hiện
- YOLO sẽ detect person và ball

### 3. Chọn Target
- **Click vào bounding box** trên video để chọn target
- Hoặc chọn từ danh sách targets trong dashboard
- Target được chọn sẽ có màu vàng nổi bật

### 4. Bắt đầu Follow
- Nhấn **"FOLLOW"** để robot bắt đầu bám target
- Robot sẽ tự động điều chỉnh hướng và khoảng cách

### 5. Điều chỉnh PID (nếu cần)
- Điều chỉnh các thông số KP, KI, KD trong bảng PID TUNING
- Nhấn **"APPLY PID"** để cập nhật

### 6. Quản lý Dataset
- **CAPTURE**: Chụp ảnh frame hiện tại
- **RECORD**: Ghi lại video
- **EXPORT**: Xuất dataset để training

### 7. Kết nối ESP32
- ESP32 kết nối WebSocket đến: `ws://<PC_IP>:8000/ws/esp32`
- Robot sẽ nhận lệnh điều khiển từ dashboard

## 📊 Hiểu các thông số

### Robot Control
| Thông số | Ý nghĩa | Giá trị mong muốn |
|----------|---------|-------------------|
| **STATE** | Trạng thái robot | FOLLOWING, TURN LEFT, TURN RIGHT, IDLE |
| **DIRECTION** | Hướng di chuyển | FORWARD, BACKWARD, TURN LEFT, TURN RIGHT |
| **SERVO ANGLE** | Góc servo | 90° (thẳng), <90° (trái), >90° (phải) |
| **MOTOR SPEED** | Tốc độ motor | 0-100 (tiến), -100-0 (lùi) |
| **ERROR X** | Lỗi hướng (pixels) | Càng gần 0 càng tốt |
| **DIST ERROR** | Lỗi khoảng cách (pixels) | Càng gần 0 càng tốt |

### Telemetry
| Thông số | Ý nghĩa |
|----------|---------|
| **Target Center X** | Vị trí X của target trong khung hình |
| **Frame Center X** | Tâm khung hình (320) |
| **BBox Height/Width** | Kích thước bounding box |
| **PID Output** | Giá trị đầu ra của PID controller |
| **FPS** | Số frame xử lý mỗi giây |

## 🐛 Debugging & Troubleshooting

### Camera không mở
```bash
# Kiểm tra camera ID
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# Thay đổi camera_id trong config.yaml
camera:
  id: 1  # Thử 0, 1, 2...
```

### Model không load
```bash
# Kiểm tra file weights tồn tại
ls apps/weights/yolov8n_person.pt

# Download lại model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### ESP32 không kết nối
```bash
# Kiểm tra IP của PC
ipconfig  # Windows
ifconfig  # Linux

# ESP32 kết nối đến: ws://192.168.1.100:8000/ws/esp32
```

### FPS thấp
```yaml
# Trong config.yaml, giảm inference_size
detection:
  inference_size: 288  # Giảm từ 320 xuống 288
  process_every_n_frames: 2  # Process mỗi 2 frames
```

### Logs
```bash
# Xem logs real-time
tail -f apps/output/logs/app.log

# Xem telemetry data
cat logs/telemetry.csv
```

## 🔧 PID Tuning Guide

### Steering PID (Điều khiển hướng)
| Thông số | Tăng | Giảm |
|----------|------|------|
| **KP** | Phản hồi nhanh hơn | Mượt hơn, ít rung |
| **KI** | Giảm sai số dư | Ít overshoot hơn |
| **KD** | Giảm dao động | Phản hồi nhanh hơn |

### Distance PID (Điều khiển khoảng cách)
| Thông số | Tăng | Giảm |
|----------|------|------|
| **KP** | Bám nhanh hơn | Mượt hơn |
| **KI** | Giảm sai số | Ít overshoot |
| **KD** | Giảm dao động | Phản hồi nhanh hơn |

### Target Height
- **Tăng** (200): Robot giữ khoảng cách xa hơn
- **Giảm** (100): Robot giữ khoảng cách gần hơn

## 📈 Performance Metrics

- **Inference Speed**: 10-15 FPS (CPU, inference_size=320)
- **Inference Time**: 45-60ms (CPU)
- **Latency**: < 100ms (end-to-end)
- **Memory Usage**: ~1.5GB RAM
- **Network Bandwidth**: ~500KB/s (video stream)

## 🚧 Future Improvements

- [ ] Optimize inference with TensorRT
- [ ] Add Re-identification (Re-ID) with Deep Learning
- [ ] Multi-camera support
- [ ] ROS2 integration
- [ ] Mobile app for remote control
- [ ] Cloud-based training pipeline
- [ ] Automatic PID tuning with AI

## 📝 License

MIT License

## 👥 Contributors

- NEXORA Team

## 📞 Contact & Support

- **Email**: support@nexora.ai
- **Website**: https://nexora.ai
- **GitHub**: https://github.com/nexora/nexora-robot

---

**⚠️ Lưu ý quan trọng**: Hệ thống này yêu cầu camera hoạt động tốt và ánh sáng đủ để detection chính xác. Đảm bảo ESP32 kết nối cùng mạng WiFi với PC.