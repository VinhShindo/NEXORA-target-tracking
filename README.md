# NEXORA

> **NEXORA – Hệ thống robot di động thông minh theo dõi và bám mục tiêu thời gian thực ứng dụng Trí tuệ nhân tạo và Thị giác máy tính**

## Tổng quan

NEXORA là dự án nghiên cứu và phát triển hệ thống robot di động tự hành có khả năng nhận diện, theo dõi và bám theo mục tiêu trong thời gian thực thông qua công nghệ Trí tuệ nhân tạo (Artificial Intelligence), Thị giác máy tính (Computer Vision) và Robot tự hành (Mobile Robotics).

Hệ thống sử dụng camera USB làm cảm biến thị giác chính, kết hợp mô hình nhận diện đối tượng được huấn luyện trên bộ dữ liệu tự xây dựng, thuật toán theo dõi đa đối tượng ByteTrack và bộ điều khiển PID để điều hướng robot di chuyển theo mục tiêu đã lựa chọn.

Dự án hướng đến khả năng triển khai trên phần cứng chi phí thấp như Raspberry Pi 4 nhưng vẫn đảm bảo hiệu suất xử lý thời gian thực và khả năng mở rộng cho các ứng dụng robot thông minh.

---

# Mục tiêu nghiên cứu

## Mục tiêu tổng quát

Thiết kế và xây dựng hệ thống robot di động có khả năng:

* Nhận diện nhiều loại đối tượng trong môi trường thực tế.
* Theo dõi liên tục mục tiêu bằng thuật toán Tracking.
* Điều khiển robot bám theo mục tiêu được chỉ định.
* Hoạt động thời gian thực trên Raspberry Pi 4.
* Duy trì khoảng cách an toàn với đối tượng.

---

## Mục tiêu cụ thể

### Thị giác máy tính

* Xây dựng bộ dữ liệu riêng gồm 5 lớp đối tượng.
* Huấn luyện mô hình MobileNetV2 Object Detection.
* Tối ưu mô hình bằng ONNX Runtime.

### Theo dõi mục tiêu

* Tích hợp ByteTrack.
* Duy trì ID đối tượng xuyên suốt nhiều khung hình.
* Hạn chế mất dấu khi đối tượng di chuyển.

### Điều khiển robot

* Tính toán vị trí tương đối của mục tiêu.
* Ước lượng khoảng cách từ kích thước Bounding Box.
* Điều khiển động cơ bằng PID Controller.

### Hệ thống nhúng

* Triển khai trên Raspberry Pi 4.
* Giao tiếp với ESP32 điều khiển động cơ.
* Tối ưu tốc độ xử lý thời gian thực.

---

# Kiến trúc hệ thống

![Sơ đồ kiến trúc hệ thống](<docs/diagrams/Sơ đồ kiến trúc hệ thống.png>)

---

# Công nghệ sử dụng

## AI & Computer Vision

* Python
* OpenCV
* TensorFlow
* MobileNetV2
* ONNX Runtime

## Multi Object Tracking

* ByteTrack

## Robotics

* ROS2 Humble
* Gazebo
* RViz

## Embedded Systems

* Raspberry Pi 4
* ESP32
* USB Camera 1080P

## Control

* PID Controller
* Kalman Filter

---

# Bộ dữ liệu

## Các lớp đối tượng

| ID | Class  |
| -- | ------ |
| 0  | person |
| 1  | bottle |
| 2  | chair  |
| 3  | ball   |
| 4  | bag    |

## Nguồn dữ liệu

* Thu thập bằng webcam thực tế.
* Môi trường trong nhà.
* Môi trường ngoài trời.
* Nhiều điều kiện ánh sáng.
* Nhiều góc quan sát.

## Định dạng dữ liệu

```text
YOLO Format
```

---

# Quy trình xử lý

## Bước 1: Thu nhận hình ảnh

Camera USB truyền khung hình về Raspberry Pi.

## Bước 2: Nhận diện đối tượng

Mô hình MobileNetV2 thực hiện:

* Phát hiện đối tượng.
* Phân loại lớp.
* Sinh Bounding Box.

## Bước 3: Theo dõi mục tiêu

ByteTrack:

* Gán ID.
* Theo dõi nhiều đối tượng.
* Duy trì ID qua các frame.

## Bước 4: Chọn mục tiêu

Target Manager:

* Chọn đối tượng cần bám.
* Ưu tiên theo loại đối tượng.
* Ưu tiên theo ID.

## Bước 5: Điều khiển robot

PID Controller:

* Tính sai số ngang.
* Tính sai số khoảng cách.
* Sinh lệnh điều khiển động cơ.

## Bước 6: Điều khiển động cơ

ESP32 nhận lệnh:

* Tiến
* Lùi
* Rẽ trái
* Rẽ phải
* Dừng

---

# Tính năng chính

## Object Detection

* Nhận diện 5 lớp đối tượng.
* Hỗ trợ thời gian thực.
* Hoạt động trên Raspberry Pi.

## Multi Object Tracking

* Theo dõi nhiều mục tiêu.
* Duy trì ID ổn định.
* Hạn chế mất dấu.

## Target Following

* Robot tự động bám theo mục tiêu.
* Điều chỉnh hướng di chuyển liên tục.
* Duy trì khoảng cách an toàn.

## Real-Time Monitoring

* Hiển thị Bounding Box.
* Hiển thị ID đối tượng.
* Hiển thị FPS.
* Hiển thị trạng thái robot.

---

# Cấu trúc thư mục

```text
NEXORA/
│
├── assets/
│   ├── images/
│   └── videos/
│
├── configs/
│
├── datasets/
│
├── docs/
│   ├── proposal/
│   ├── report/
│   └── diagrams/
│
├── hardware/
│   ├── esp32/
│   └── wiring/
│
├── models/
│
├── notebooks/
│
├── ros2_ws/
│
├── scripts/
│
├── simulations/
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── control/
│   ├── navigation/
│   ├── communication/
│   └── utils/
│
├── tests/
│
├── weights/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# Hiệu năng mục tiêu

| Thành phần    | Mục tiêu  |
| ------------- | --------- |
| Camera        | 30 FPS    |
| Detection     | 15–25 FPS |
| Tracking      | >100 FPS  |
| Toàn hệ thống | 12–20 FPS |
| Độ trễ        | <100 ms   |

---


Đây là tất cả các file cần thiết cho các folder trống. Bạn đã có đầy đủ cấu trúc để upload lên GitHub với:

1. **datasets/** - Dữ liệu huấn luyện
2. **hardware/esp32/** - Firmware ESP32
3. **hardware/wiring/** - Hướng dẫn đấu dây
4. **models/** - Model AI và tài liệu
5. **ros2_ws/** - ROS2 workspace
6. **simulations/** - Mô phỏng Gazebo

---

# Hướng phát triển tương lai

* Re-Identification (ReID)
* Voice Command
* SLAM Navigation
* Obstacle Avoidance
* Multi-Robot System
* Edge AI Optimization
* TensorRT Deployment
* Autonomous Patrol Mode

---

# Thành viên phát triển

NEXORA Development Team

---

# License

MIT License
