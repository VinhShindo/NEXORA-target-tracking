# NEXORA

## Giới thiệu

NEXORA là hệ thống robot di động thông minh có khả năng nhận diện, theo dõi và bám mục tiêu trong thời gian thực bằng công nghệ Trí tuệ nhân tạo (AI), Thị giác máy tính (Computer Vision) và Robot tự hành (Mobile Robotics).

Hệ thống cho phép robot tự động phát hiện đối tượng từ luồng hình ảnh camera, xác định vị trí mục tiêu, ước lượng khoảng cách và điều khiển chuyển động để duy trì khả năng theo dõi liên tục trong môi trường động.

---

## Mục tiêu dự án

- Xây dựng hệ thống robot di động tự hành.
- Phát hiện và nhận diện mục tiêu bằng AI.
- Theo dõi mục tiêu theo thời gian thực.
- Điều khiển robot bám theo mục tiêu.
- Duy trì khoảng cách an toàn với đối tượng.
- Đảm bảo hiệu suất xử lý thời gian thực.

---

## Kiến trúc hệ thống

Camera
↓
Object Detection (YOLO)
↓
Object Tracking (ByteTrack/DeepSORT)
↓
Target Localization
↓
Motion Controller (PID)
↓
Mobile Robot

---

## Công nghệ sử dụng

### AI & Computer Vision

- Python
- OpenCV
- PyTorch
- YOLO
- ONNX Runtime

### Robotics

- ROS2 Humble
- Gazebo
- RViz

### Embedded Systems

- Raspberry Pi 5
- NVIDIA Jetson Orin Nano
- ESP32

### Control

- PID Controller
- Kalman Filter

---

## Chức năng chính

### Nhận diện đối tượng

- Phát hiện người
- Phát hiện vật thể
- Nhận diện theo lớp đối tượng

### Theo dõi mục tiêu

- Duy trì ID mục tiêu
- Theo dõi liên tục khi di chuyển

### Điều khiển robot

- Tiến
- Lùi
- Rẽ trái
- Rẽ phải

### Giám sát thời gian thực

- Hiển thị video trực tiếp
- Hiển thị Bounding Box
- Hiển thị trạng thái robot

---

## Cấu trúc thư mục

NEXORA/
│
├── docs/
├── datasets/
├── models/
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── control/
│   ├── navigation/
│   └── utils/
│
├── ros2_ws/
├── simulations/
├── tests/
├── requirements.txt
├── README.md
└── LICENSE

---

## Lộ trình phát triển

### Giai đoạn 1

- Xây dựng môi trường phát triển
- Tích hợp camera
- Thu thập dữ liệu

### Giai đoạn 2

- Huấn luyện mô hình phát hiện đối tượng
- Tối ưu tốc độ suy luận

### Giai đoạn 3

- Tích hợp thuật toán theo dõi mục tiêu
- Xây dựng bộ điều khiển

### Giai đoạn 4

- Tích hợp ROS2
- Mô phỏng trên Gazebo

### Giai đoạn 5

- Triển khai trên robot thực tế
- Kiểm thử và đánh giá

---

## Ứng dụng

- Robot đồng hành
- Robot hỗ trợ trong nhà
- Robot giao hàng
- Robot giám sát thông minh
- Robot hỗ trợ cứu hộ

---

## Thành viên phát triển

NEXORA Development Team

---

## Giấy phép

MIT License
