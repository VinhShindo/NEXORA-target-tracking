# NEXORA - AI Target Following Robot

> NEXORA là hệ thống robot di động thông minh theo dõi và bám mục tiêu thời gian thực ứng dụng Trí tuệ nhân tạo, Thị giác máy tính và Robot tự hành.

---

## 📌 Tổng quan

NEXORA là dự án nghiên cứu và phát triển robot di động thông minh có khả năng nhận diện, theo dõi và bám theo mục tiêu trong thời gian thực.

Hệ thống sử dụng camera USB kết hợp mô hình YOLOv8n được tối ưu cho nền tảng Edge AI nhằm triển khai trực tiếp trên Raspberry Pi 4 hoặc PC.

NEXORA hỗ trợ hai chế độ hoạt động chính:

- **Human Following Mode** - Theo dõi người
- **Ball Following Mode** - Theo dõi quả bóng

Dự án hướng tới việc nghiên cứu, đánh giá và tối ưu các mô hình Object Detection trên phần cứng nhúng có tài nguyên hạn chế.

---

## 🎯 Mục tiêu nghiên cứu

### Mục tiêu tổng quát

Thiết kế và phát triển hệ thống robot di động có khả năng:

- Nhận diện mục tiêu thời gian thực
- Theo dõi mục tiêu liên tục
- Điều khiển robot bám theo mục tiêu
- Duy trì khoảng cách an toàn
- Hoạt động ổn định trên Raspberry Pi 4

### Mục tiêu cụ thể

#### Computer Vision
- Xây dựng bộ dữ liệu Person Dataset
- Xây dựng bộ dữ liệu Ball Dataset
- Huấn luyện và đánh giá nhiều kiến trúc phát hiện đối tượng

#### Edge AI
- Tối ưu mô hình cho Raspberry Pi 4
- Chuyển đổi sang ONNX
- Quantization INT8
- Triển khai bằng ONNX Runtime

#### Tracking
- Tích hợp ByteTrack (self-implemented)
- Duy trì ID đối tượng ổn định
- Hạn chế mất dấu mục tiêu
- Feature Database cho Re-ID

#### Robot Control
- Xác định vị trí mục tiêu
- Ước lượng khoảng cách
- Điều khiển robot bằng PID Controller
- Lọc nhiễu bbox (Median + EMA)

---

## 🏗️ Kiến trúc hệ thống

![Sơ đồ kiến trúc hệ thống](<docs/diagrams/Sơ đồ kiến trúc hệ thống.png>)

---

## 🧠 Logic điều khiển Robot

### Sơ đồ quyết định tổng quan

```
                            ┌─────────────────────┐
                            │   Camera Frame      │
                            │   640x480 @ 30fps   │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   YOLO Detection    │
                            │   Person / Ball     │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   ByteTrack         │
                            │   Gán ID tracking   │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   Có target?        │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
         ┌─────────────────────┐              ┌─────────────────────┐
         │   KHÔNG             │              │   CÓ                │
         │   TARGET_LOST       │              │   Tính toán lỗi     │
         │   Dừng / Tìm kiếm   │              └──────────┬──────────┘
         └─────────────────────┘                         │
                                       ┌─────────────────┴─────────────────┐
                                       │                                   │
                                       ▼                                   ▼
                            ┌─────────────────────┐          ┌─────────────────────┐
                            │  Lỗi hướng          │          │  Lỗi khoảng cách    │
                            │  error_x =          │          │  distance_error =   │
                            │  target_cx - 320    │          │  150 - bbox_height  │
                            └──────────┬──────────┘          └──────────┬──────────┘
                                       │                                   │
                                       ▼                                   ▼
                            ┌─────────────────────┐          ┌─────────────────────┐
                            │  Lọc nhiễu          │          │  Lọc nhiễu          │
                            │  - Deadzone 2.0px   │          │  - Deadzone 3.0px   │
                            │  - Median 5 frame   │          │  - Median 5 frame   │
                            │  - EMA 0.3          │          │  - EMA 0.3          │
                            └──────────┬──────────┘          └──────────┬──────────┘
                                       │                                   │
                                       ▼                                   ▼
                            ┌─────────────────────┐          ┌─────────────────────┐
                            │  Steering PID       │          │  Distance PID       │
                            │  KP=0.05 KI=0.001   │          │  KP=0.1 KI=0.002    │
                            │  KD=0.01            │          │  KD=0.02            │
                            └──────────┬──────────┘          └──────────┬──────────┘
                                       │                                   │
                                       │                    ┌──────────────┴──────────────┐
                                       │                    │                             │
                                       ▼                    ▼                             ▼
                            ┌─────────────────────┐  ┌─────────────┐          ┌─────────────┐
                            │  steering_output    │  │ motor_speed │          │ motor_speed │
                            │  = 3.297            │  │  > 0        │          │  < 0        │
                            └──────────┬──────────┘  └──────┬──────┘          └──────┬──────┘
                                       │                    │                         │
                                       ▼                    ▼                         ▼
                            ┌─────────────────────┐  ┌─────────────┐          ┌─────────────┐
                            │  servo_angle =      │  │  TIẾN      │          │  LÙI        │
                            │  90 + steering_out  │  │  FORWARD   │          │  BACKWARD   │
                            └──────────┬──────────┘  └─────────────┘          └─────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
         ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
         │  error_x > 30   │  │  |error_x|<=30  │  │  error_x < -30  │
         │  RẼ PHẢI        │  │  ĐI THẲNG       │  │  RẼ TRÁI         │
         │  TURN RIGHT     │  │  FORWARD        │  │  TURN LEFT       │
         └─────────────────┘  └─────────────────┘  └─────────────────┘
                                       │
                                       ▼
                            ┌─────────────────────────────────────┐
                            │   Gửi lệnh điều khiển xuống ESP32   │
                            │   {motor: -30.5, servo: 93.3,       │
                            │    dir: "TURN RIGHT"}               │
                            └─────────────────────────────────────┘
```

### Chi tiết quyết định Rẽ trái / Rẽ phải / Đi thẳng

```
                      Frame Center X = 320
                             │
                  error_x = 0 | (Đi thẳng)
                             │
          error_x < 0        |        error_x > 0
          (Rẽ trái)          |        (Rẽ phải)
                             │
      ←----[TARGET]----------|----------[TARGET]----→
      ←-----268.7-----------|-----------371.3-------→
      ←----(-51.34)---------|---------(+51.34)------→
                             │
                  Threshold = 30 pixels
```

**Công thức:**
```python
error_x = target_center_x - frame_center_x  # frame_center_x = 320

if error_x > 30:      # Target ở bên phải
    direction = "TURN RIGHT"
elif error_x < -30:   # Target ở bên trái
    direction = "TURN LEFT"
else:                 # Target ở giữa
    direction = "FORWARD"
```

### Chi tiết quyết định Tiến / Lùi / Dừng

```
                    target_height = 150 (pixels)
                           |
                     distance_error = 0 | (Dừng)
                           |
        distance_error > 0 |        distance_error < 0
        (TIẾN)             |        (LÙI)
                           |
    [NHỎ]                  |                  [LỚN]
    bbox_height=80         |         bbox_height=394.9
    distance_error=+70     |         distance_error=-244.86
    motor_speed=+70        |         motor_speed=-30.5
    → TIẾN                 |         → LÙI
```

**Công thức:**
```python
distance_error = target_height - bbox_height  # target_height = 150

# PID Distance
motor_speed = PID_Distance.update(distance_error)

if motor_speed > 5:      # Sai số dương → target xa
    linear_velocity = + (TIẾN)
elif motor_speed < -5:   # Sai số âm → target gần
    linear_velocity = - (LÙI)
else:                    # Sai số ≈ 0 → đúng khoảng cách
    linear_velocity = 0  # DỪNG
```

### Bảng tổng hợp quyết định

| Điều kiện | error_x | distance_error | motor_speed | Hướng | Trạng thái |
|-----------|---------|---------------|-------------|-------|------------|
| Target ở giữa + đúng khoảng cách | \|error_x\| < 30 | ≈ 0 | ≈ 0 | FORWARD | FOLLOWING |
| Target ở giữa + quá xa | \|error_x\| < 30 | > 0 | > 0 | FORWARD | FOLLOWING |
| Target ở giữa + quá gần | \|error_x\| < 30 | < 0 | < 0 | FORWARD (lùi) | FOLLOWING |
| Target bên phải + đúng khoảng cách | > 30 | ≈ 0 | ≈ 0 | TURN RIGHT | TURN_RIGHT |
| Target bên trái + đúng khoảng cách | < -30 | ≈ 0 | ≈ 0 | TURN LEFT | TURN_LEFT |
| Target bên phải + quá xa | > 30 | > 0 | > 0 | TURN RIGHT (rẽ + tiến) | TURN_RIGHT |
| Target bên phải + quá gần | > 30 | < 0 | < 0 | TURN RIGHT (rẽ + lùi) | TURN_RIGHT |
| Không có target | - | - | 0 | STOP | TARGET_LOST |

---

## 🔬 Tham số điều khiển

### 1. Steering PID (Điều khiển hướng)

| Tham số | Giá trị | Ý nghĩa | Ảnh hưởng |
|---------|---------|---------|-----------|
| **KP** | 0.05 | Hệ số tỉ lệ | Phản hồi theo lỗi hướng. Tăng → rẽ nhanh hơn, Giảm → rẽ mượt hơn |
| **KI** | 0.001 | Hệ số tích phân | Xử lý sai số dư tích lũy. Tăng → giảm sai số, Giảm → ít overshoot |
| **KD** | 0.01 | Hệ số vi phân | Giảm dao động. Tăng → ổn định hơn, Giảm → phản hồi nhanh hơn |

**Công thức:** `steering_output = KP * error + KI * integral + KD * derivative`

### 2. Distance PID (Điều khiển khoảng cách)

| Tham số | Giá trị | Ý nghĩa | Ảnh hưởng |
|---------|---------|---------|-----------|
| **KP** | 0.1 | Hệ số tỉ lệ | Phản hồi theo lỗi khoảng cách. Tăng → bám nhanh hơn |
| **KI** | 0.002 | Hệ số tích phân | Xử lý sai số khoảng cách tích lũy |
| **KD** | 0.02 | Hệ số vi phân | Giảm dao động khoảng cách |

**Công thức:** `distance_output = KP * error + KI * integral + KD * derivative`

### 3. Bộ lọc nhiễu

| Tham số | Giá trị | Ý nghĩa | Ảnh hưởng |
|---------|---------|---------|-----------|
| **deadzone_steering** | 2.0 | Vùng chết steering | Bỏ qua nhiễu < 2px. Tăng → ổn định hơn, Giảm → nhạy hơn |
| **deadzone_distance** | 3.0 | Vùng chết distance | Bỏ qua nhiễu < 3px. Tăng → ổn định hơn, Giảm → nhạy hơn |
| **smooth_factor** | 0.3 | Hệ số làm mịn EMA | Trọng số giá trị mới. Tăng → phản hồi nhanh, Giảm → mượt hơn |
| **stability_factor** | 0.3 | Hệ số ổn định | Giảm tốc độ khi ổn định. Tăng → chậm hơn, Giảm → nhanh hơn |

### 4. Tham số tracking

| Tham số | Giá trị | Ý nghĩa | Ảnh hưởng |
|---------|---------|---------|-----------|
| **track_thresh** | 0.4 | Ngưỡng tracking | Thấp → bám nhiều hơn, Cao → bám chính xác hơn |
| **high_thresh** | 0.6 | Ngưỡng high confidence | Phân loại detection chất lượng cao |
| **match_thresh** | 0.7 | Ngưỡng match IoU | Thấp → match nhiều hơn, Cao → match chính xác hơn |
| **track_buffer** | 30 | Số frame giữ track | Lớn → giữ track lâu hơn khi mất |

---

## 📁 Cấu trúc thư mục (Cập nhật)

```text
NEXORA-target-tracking/
│
├── src/                              # Source code chính
│   ├── app.py                        # FastAPI server
│   ├── configs/
│   │   └── config.yaml               # Cấu hình hệ thống
│   ├── controllers/
│   │   ├── steering_pid.py           # PID Steering
│   │   └── distance_pid.py           # PID Distance
│   ├── services/
│   │   ├── camera_service.py         # Camera capture
│   │   ├── detection_service.py      # YOLO detection
│   │   ├── tracking_service.py       # ByteTrack + Feature DB
│   │   ├── robot_service.py          # Robot control logic
│   │   └── dataset_service.py        # Dataset management
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css             # Dashboard styles
│   │   └── js/
│   │       └── dashboard.js          # Dashboard JavaScript
│   ├── templates/
│   │   └── index.html                # Dashboard HTML
│   ├── logs/
│   │   └── telemetry.csv             # Telemetry data log
│   └── output/
│       ├── logs/
│       │   └── app.log               # Application logs
│       ├── snapshots/                # Captured images
│       ├── recordings/               # Recorded videos
│       └── datasets/                 # Exported datasets
│
├── weights/                          # Model weights
│   ├── yolov8n_person.pt             # PyTorch model
│   └── yolov8n_person.onnx           # ONNX model
│
├── datasets/                         # Training datasets
│   ├── Readme.md
│   └── data.yaml                     # Dataset config
│
├── hardware/                         # Hardware files
│   ├── esp32/
│   │   ├── esp32_motor_control/
│   │   │   └── esp32_motor_control.ino  # ESP32 firmware
│   │   └── README.md
│   └── wiring/
│       └── README.md                 # Wiring guide
│
├── docs/                             # Documentation
│   ├── diagrams/
│   │   └── Sơ đồ kiến trúc hệ thống.png
│   └── report/
│       └── README.md
│
├── scripts/                          # Utility scripts
│   └── export_onnx.py                # Export to ONNX
│
├── check_folder.py                   # Folder structure checker
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🛠️ Công nghệ sử dụng

### AI & Computer Vision
- Python 3.12
- OpenCV 4.8+
- Ultralytics YOLOv8
- ONNX Runtime

### Tracking
- ByteTrack (self-implemented)
- Kalman Filter
- Feature Database (Re-ID)

### Backend & Dashboard
- FastAPI
- WebSocket
- Jinja2 Templates

### Embedded Systems
- Raspberry Pi 4
- ESP32 (WebSocket Client)
- USB Camera 1080P

### Control
- PID Controller
- Median Filter
- EMA Smoothing
- Deadzone

---

## 📊 Hiệu năng

| Thành phần | Mục tiêu | Thực tế |
|------------|----------|---------|
| Camera | 30 FPS | 30 FPS |
| Detection (YOLOv8n) | 15-25 FPS | 10-15 FPS |
| Tracking | >100 FPS | >100 FPS |
| Toàn hệ thống | 12-20 FPS | 10-15 FPS |
| Inference Time | <50ms | 40-60ms |

---

## 🔧 Cài đặt và chạy

### Yêu cầu hệ thống
- Python 3.12+
- Raspberry Pi 4 hoặc PC (Windows/Linux)
- Camera USB 1080P
- ESP32 (tùy chọn)

### Cài đặt

```bash
# Clone repository
git clone https://github.com/yourusername/NEXORA-target-tracking.git
cd NEXORA-target-tracking

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy ứng dụng

```bash
# Chạy server
cd src
python app.py

# Mở trình duyệt
# Truy cập: http://localhost:8000
```

---

## 🎮 Hướng dẫn sử dụng

### 1. Kết nối Camera
- Nhấn **START CAMERA** để mở camera
- Kiểm tra luồng video hiển thị

### 2. Bắt đầu Detection
- Nhấn **DETECT** để khởi động YOLO
- Hệ thống sẽ bắt đầu phát hiện người và bóng

### 3. Chọn Target
- Click trực tiếp lên khung người cần theo dõi
- Hoặc nhấn **SELECT** để chọn target hiện tại

### 4. Bắt đầu Follow
- Nhấn **FOLLOW** để robot bắt đầu bám theo target
- Robot sẽ tự động điều chỉnh hướng và khoảng cách

### 5. Dừng
- Nhấn **STOP** để dừng theo dõi
- Nhấn **X RELEASE** để giải phóng target

### 6. Điều chỉnh PID
- Điều chỉnh các tham số KP, KI, KD
- Nhấn **APPLY PID** để cập nhật

---

## 📡 Giao thức ESP32

ESP32 kết nối qua WebSocket đến server:

```
ws://<IP>:8000/ws/esp32
```

### JSON Command Format

```json
{
  "type": "control",
  "timestamp": 1700000000123,
  "command_id": 1700000000123,
  "is_following": true,
  "target_detected": true,
  "target_id": 1,
  "targets_count": 1,
  "control": {
    "motor_speed": 25.5,
    "linear_velocity": 0.127,
    "angular_velocity": 0.015,
    "servo_angle": 92.5,
    "direction": "FORWARD",
    "state": "FOLLOWING"
  },
  "error": {
    "steering_error": 2.5,
    "distance_error": -3.0,
    "pid_steering": 0.025,
    "pid_distance": 0.03
  },
  "target": {
    "id": 1,
    "class": "person",
    "confidence": 0.85,
    "bbox_center": [260, 240],
    "bbox_height": 155.0,
    "bbox_width": 80.0
  },
  "distance": 1.25,
  "system": {
    "fps": 12.0,
    "inference_time": 47.0,
    "esp32_connected": true
  }
}
```

### ESP32 Response

```json
{
  "status": "ok",
  "motor": 25.5,
  "servo": 92.5,
  "timestamp": 1700000000123
}
```

---

## 📈 Quy trình xử lý

```
1. Camera → Thu nhận hình ảnh
   └── USB Camera truyền frame 640x480 @ 30fps

2. YOLO Detection → Nhận diện đối tượng
   ├── Phát hiện person và ball
   ├── Sinh Bounding Box
   └── Confidence Score

3. ByteTrack → Theo dõi mục tiêu
   ├── Gán ID duy nhất
   ├── Theo dõi nhiều đối tượng
   ├── Kalman Filter dự đoán
   └── Feature Database Re-ID

4. Robot Service → Điều khiển
   ├── Tính lỗi hướng (error_x)
   ├── Tính lỗi khoảng cách (distance_error)
   ├── Lọc nhiễu (Median + EMA + Deadzone)
   ├── PID Steering và Distance
   └── Tạo lệnh điều khiển

5. ESP32 → Thực thi
   ├── Điều khiển Motor
   ├── Điều khiển Servo
   └── Phản hồi trạng thái
```

---

## 🧪 Các chế độ hoạt động

### Human Following Mode
- Robot phát hiện người trong khung hình
- Người dùng lựa chọn đối tượng cần theo dõi
- Robot tự động bám theo

### Ball Following Mode
- Robot phát hiện quả bóng
- Tự động xác định vị trí và khoảng cách
- Điều khiển chuyển động để bám theo

---

## 🔮 Hướng phát triển tương lai

- [ ] Re-Identification (ReID) nâng cao
- [ ] Voice Command
- [ ] SLAM Navigation
- [ ] Obstacle Avoidance
- [ ] Multi-Robot Collaboration
- [ ] Edge AI Optimization (TensorRT, OpenVINO)
- [ ] Autonomous Patrol Mode
- [ ] Mobile App Control

---

## 👥 Development Team

NEXORA Development Team

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 📧 Contact

- GitHub: [NEXORA](https://github.com/yourusername/NEXORA-target-tracking)
- Email: your.email@example.com