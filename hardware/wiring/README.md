# NEXORA Hardware Wiring Guide

## System Overview
```
[Camera] ─── [Raspberry Pi 4] ─── [ESP32] ─── [Motor Driver] ─── [Motors]
                   │
                   └── [Power Supply]
```

## Wiring Diagram

### Raspberry Pi 4 Connections
```
Raspberry Pi 4           ESP32
─────────────────────────────────
GPIO 14 (TX)     ────── GPIO 16 (RX)
GPIO 15 (RX)     ────── GPIO 17 (TX)
5V               ────── VIN
GND              ────── GND
```

### Camera Connections
```
Raspberry Pi 4           USB Camera
─────────────────────────────────
USB Port          ────── USB Connector
```

### ESP32 to Motor Driver
```
ESP32                    L298N Motor Driver
────────────────────────────────────────────
GPIO 12 (PWM_L)   ────── ENA
GPIO 14 (IN1_L)   ────── IN1
GPIO 27 (IN2_L)   ────── IN2
GPIO 13 (PWM_R)   ────── ENB
GPIO 26 (IN1_R)   ────── IN3
GPIO 25 (IN2_R)   ────── IN4
5V                ────── +5V
GND               ────── GND
```

### Motor Driver to Motors
```
L298N                    DC Motors
─────────────────────────────────
OUT1                     Motor Left (+)
OUT2                     Motor Left (-)
OUT3                     Motor Right (+)
OUT4                     Motor Right (-)
```

### Power Distribution
```
12V Battery
    │
    ├── L298N (12V IN)
    │
    ├── DC-DC Converter (12V → 5V)
    │     ├── Raspberry Pi (5V)
    │     └── ESP32 (5V via VIN)
    │
    └── USB Camera (via Raspberry Pi USB)
```

## Component Specifications

### Raspberry Pi 4
- **Model**: Raspberry Pi 4 Model B
- **RAM**: 4GB or 8GB
- **Power**: 5V 3A
- **OS**: Raspberry Pi OS or Ubuntu

### ESP32
- **Model**: ESP32 Dev Kit V1
- **Power**: 5V (via VIN)
- **Communication**: UART (115200 baud)

### Motor Driver
- **Model**: L298N or L293D
- **Power**: 12V (for motors)
- **Logic**: 5V

### DC Motors
- **Type**: DC Geared Motor
- **Voltage**: 6-12V
- **RPM**: 100-300 RPM

### Camera
- **Type**: USB Webcam
- **Resolution**: 1080p
- **FPS**: 30

## Connection Checklist

### Power Connections
- [ ] 12V battery connected to motor driver
- [ ] DC-DC converter connected to battery
- [ ] Raspberry Pi powered via USB-C
- [ ] ESP32 powered via VIN (5V)

### Signal Connections
- [ ] UART connections (RX/TX) between Pi and ESP32
- [ ] PWM connections from ESP32 to motor driver
- [ ] Control pins from ESP32 to motor driver

### Motor Connections
- [ ] Left motor connected to OUT1/OUT2
- [ ] Right motor connected to OUT3/OUT4
- [ ] Motors securely fastened to chassis

### Camera Connections
- [ ] USB camera connected to Pi
- [ ] Camera detected by system

## Safety Guidelines

1. **Power Off Before Wiring**: Always disconnect power before making connections
2. **Double Check Polarity**: Reverse polarity can damage components
3. **Use Fuses**: Add fuses for overcurrent protection
4. **Heat Management**: Ensure proper ventilation for motor driver
5. **Secure Connections**: Use heat shrink or electrical tape
6. **Label Wires**: Label all connections for easy troubleshooting

## Troubleshooting

### Raspberry Pi Not Detecting Camera
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

### ESP32 Not Responding
```bash
# Check serial port
ls /dev/ttyUSB*
# Test connection
echo "status" > /dev/ttyUSB0
```

### Motors Not Moving
1. Check power supply (12V)
2. Verify PWM signals
3. Check motor driver enable pins
4. Test motors directly with battery

### Communication Issues
1. Verify baud rate (115200)
2. Check RX/TX connections
3. Ensure common ground
4. Test with loopback