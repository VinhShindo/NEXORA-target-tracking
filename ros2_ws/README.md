# NEXORA ROS2 Workspace

## Overview
ROS2 integration for NEXORA robot system.

## Workspace Structure
```
ros2_ws/
├── src/
│   ├── nexora_description/
│   │   ├── urdf/
│   │   ├── meshes/
│   │   └── launch/
│   ├── nexora_bringup/
│   │   └── launch/
│   ├── nexora_control/
│   │   ├── config/
│   │   └── launch/
│   └── nexora_vision/
│       ├── src/
│       └── launch/
├── build/
├── install/
└── log/
```

## Setup

### Install ROS2 Humble
```bash
# Setup sources
sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2
sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions
```

### Build Workspace
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Packages

### nexora_description
Robot description package with URDF models.

### nexora_bringup
Launch files for starting the robot system.

### nexora_control
Controller configurations for robot navigation.

### nexora_vision
Vision processing nodes for object detection and tracking.

## Launch Files

### Start Robot
```bash
ros2 launch nexora_bringup robot.launch.py
```

### Start Vision
```bash
ros2 launch nexora_vision vision.launch.py
```

### Start Navigation
```bash
ros2 launch nexora_control control.launch.py
```

## Topics

### Published Topics
| Topic | Type | Description |
|-------|------|-------------|
| /camera/image_raw | sensor_msgs/Image | Camera feed |
| /detections | nexora_msgs/Detection | Object detections |
| /tracks | nexora_msgs/Track | Tracked objects |
| /cmd_vel | geometry_msgs/Twist | Velocity commands |

### Subscribed Topics
| Topic | Type | Description |
|-------|------|-------------|
| /cmd_vel | geometry_msgs/Twist | Velocity commands |
| /target_select | nexora_msgs/Target | Target selection |

## Services

| Service | Type | Description |
|---------|------|-------------|
| /set_target | nexora_srv/SetTarget | Set target object |
| /get_status | nexora_srv/GetStatus | Get robot status |
| /start_following | std_srvs/SetBool | Start/stop following |

## Testing

### Test Camera
```bash
ros2 run rqt_image_view rqt_image_view
```

### Test Motor Control
```bash
ros2 topic pub --once /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}, angular: {z: 0.0}}"
```

### Test Detection
```bash
ros2 run nexora_vision detector_node --ros-args -p model_path:=models/model.onnx
```

## Integration with Python
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
    def send_command(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)
```

## Troubleshooting

### Build Errors
```bash
# Clean build
rm -rf build/ install/ log/
colcon build --symlink-install
```

### Communication Issues
```bash
# Check ROS2 status
ros2 node list
ros2 topic list
ros2 service list

# Check DDS
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### Permission Issues
```bash
sudo usermod -aG dialout $USER
newgrp dialout
```

## References
- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [Gazebo Tutorials](http://gazebosim.org/tutorials)
- [ROS2 Control](https://control.ros.org/)