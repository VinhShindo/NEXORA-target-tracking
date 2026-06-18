"""
ESP32 Communication Module for Motor Control
"""
import serial
import time
import struct
from typing import Optional
import threading
import queue

class ESP32Comm:
    def __init__(self, port: str = '/dev/ttyUSB0', 
                 baudrate: int = 115200,
                 timeout: float = 0.1):
        """
        Initialize ESP32 serial communication
        
        Args:
            port: Serial port
            baudrate: Baud rate
            timeout: Read timeout
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.connected = False
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def connect(self) -> bool:
        """Connect to ESP32"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.connected = True
            print(f"Connected to ESP32 on {self.port}")
            
            # Start communication thread
            self.running = True
            self.thread = threading.Thread(target=self._communication_loop)
            self.thread.daemon = True
            self.thread.start()
            
            return True
        except Exception as e:
            print(f"Failed to connect to ESP32: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.serial:
            self.serial.close()
            self.serial = None
        
        self.connected = False
        print("Disconnected from ESP32")
    
    def send_command(self, command: dict) -> bool:
        """Send command to ESP32"""
        if not self.connected or not self.serial:
            return False
        
        try:
            # Format command
            # Example: {"type": "motor", "left": 0.5, "right": 0.5}
            cmd_type = command.get('type', 'motor')
            
            if cmd_type == 'motor':
                left = command.get('left', 0.0)
                right = command.get('right', 0.0)
                
                # Convert to motor speeds (-1.0 to 1.0)
                left_speed = max(-1.0, min(1.0, left))
                right_speed = max(-1.0, min(1.0, right))
                
                # Pack into bytes
                # Format: [start_byte, left_byte, right_byte, end_byte]
                left_byte = int((left_speed + 1.0) * 127.5)  # 0-255
                right_byte = int((right_speed + 1.0) * 127.5)  # 0-255
                
                packet = struct.pack('>BBB', 0xAA, left_byte, right_byte)
                self.serial.write(packet)
                return True
            else:
                # Send raw command
                self.serial.write(command.get('data', b'').encode())
                return True
                
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def read_response(self) -> Optional[str]:
        """Read response from ESP32"""
        if not self.connected or not self.serial:
            return None
        
        try:
            if self.serial.in_waiting > 0:
                response = self.serial.readline().decode('utf-8', errors='ignore').strip()
                return response
        except Exception as e:
            print(f"Error reading response: {e}")
        
        return None
    
    def _communication_loop(self):
        """Communication thread loop"""
        while self.running:
            try:
                # Process command queue
                try:
                    command = self.command_queue.get_nowait()
                    self.send_command(command)
                except queue.Empty:
                    pass
                
                # Read responses
                response = self.read_response()
                if response:
                    self.response_queue.put(response)
                
                time.sleep(0.01)
            except Exception as e:
                print(f"Communication error: {e}")
                time.sleep(0.1)
    
    def send_motor_command(self, left_speed: float, right_speed: float):
        """Send motor command to ESP32"""
        command = {
            'type': 'motor',
            'left': left_speed,
            'right': right_speed
        }
        self.command_queue.put(command)
    
    def send_stop(self):
        """Send stop command"""
        self.send_motor_command(0.0, 0.0)
    
    def check_connection(self) -> bool:
        """Check if connection is still alive"""
        if not self.connected or not self.serial:
            return False
        
        try:
            # Send ping command
            self.serial.write(b'ping\n')
            response = self.serial.readline().decode('utf-8', errors='ignore').strip()
            return response == 'pong'
        except:
            return False