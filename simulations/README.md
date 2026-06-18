# NEXORA Simulation Environment

## Overview
Gazebo/ROS2 simulation environment for NEXORA robot.

## Simulation Components
```
simulations/
├── worlds/
│   ├── empty.world
│   ├── office.world
│   └── outdoor.world
├── models/
│   ├── nexora/
│   │   ├── model.sdf
│   │   ├── model.config
│   │   └── meshes/
│   └── objects/
├── launch/
│   ├── simulation.launch.py
│   └── navigation.launch.py
├── config/
│   ├── navigation.yaml
│   └── sensors.yaml
└── README.md
```

## Installation

### Install Gazebo
```bash
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-ros2-control
```

### Install Simulation Dependencies
```bash
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-slam-toolbox
```

## Launch Simulation

### Start Empty World
```bash
ros2 launch simulations simulation.launch.py world:=empty
```

### Start Office Environment
```bash
ros2 launch simulations simulation.launch.py world:=office
```

### Start with Navigation
```bash
ros2 launch simulations navigation.launch.py
```

## World Environments

### Empty World
Basic world with flat ground for testing.

### Office Environment
Office environment with desks, chairs, and people.

### Outdoor Environment
Outdoor environment with varied terrain and obstacles.

## Robot Models

### NEXORA Robot
- Differential drive
- USB camera
- LIDAR (optional)
- IMU (optional)

### Object Models
- Person (animated)
- Bottle
- Chair
- Ball
- Bag

## Testing

### Test Robot Movement
```bash
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}, angular: {z: 0.0}}"
```

### Test Camera
```bash
ros2 run rqt_image_view rqt_image_view
```

### Test Detection
```bash
ros2 run nexora_vision detector_node
```

## Customization

### Add New Object
1. Create model in `models/objects/`
2. Add to world file
3. Spawn in simulation

### Modify Robot
1. Edit `models/nexora/model.sdf`
2. Update URDF
3. Rebuild package

## Performance

### System Requirements
- CPU: Intel i5 or equivalent
- RAM: 8GB minimum
- GPU: Optional (for advanced rendering)
- Storage: 10GB

### Optimization
```bash
# Reduce graphics quality
export GAZEBO_HEADLESS=1

# Limit FPS
gz physics -r 0.5

# Use simplified models
```