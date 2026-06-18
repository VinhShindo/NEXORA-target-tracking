# ESP32 Motor Controller Firmware

## Overview
This firmware controls DC motors for the NEXORA robot using ESP32 microcontroller.

## Pin Configuration
| Pin | Function |
|-----|----------|
| 12  | Left Motor PWM |
| 14  | Left Motor IN1 |
| 27  | Left Motor IN2 |
| 13  | Right Motor PWM |
| 26  | Right Motor IN1 |
| 25  | Right Motor IN2 |

## Communication Protocol

### Serial Commands
```
# Set motor speeds (range: -255 to 255)
[left_speed] [right_speed]

# Stop motors
stop

# Get status
status

# Reset motors
reset
```

### Command Format Examples
```
# Move forward
100 100

# Turn left
-50 50

# Turn right
50 -50

# Stop
stop
```

## Installation

### 1. Install Arduino IDE
Download and install Arduino IDE from https://www.arduino.cc/

### 2. Install ESP32 Board
1. File → Preferences → Additional Boards Manager URLs
2. Add: https://dl.espressif.com/dl/package_esp32_index.json
3. Tools → Board → Boards Manager → Search "ESP32" → Install

### 3. Upload Firmware
1. Open `esp32_motor_control.ino`
2. Select Tools → Board → ESP32 Dev Module
3. Select correct COM port
4. Click Upload

## Testing

### Basic Test
```python
import serial
import time

ser = serial.Serial('/dev/ttyUSB0', 115200)

# Test forward
ser.write(b'100 100\n')
time.sleep(2)

# Stop
ser.write(b'stop\n')
time.sleep(1)

ser.close()
```

### Motor Test Sequence
1. Forward: `100 100`
2. Backward: `-100 -100`
3. Turn Left: `-50 50`
4. Turn Right: `50 -50`
5. Stop: `stop`

## Troubleshooting

### No Response
- Check USB connection
- Verify correct COM port
- Check serial baud rate (115200)

### Motors Not Moving
- Check power supply (min 5V, 2A)
- Verify motor connections
- Check PWM pins

### Erratic Movement
- Reduce speed values
- Check for EMI interference
- Verify motor driver connections