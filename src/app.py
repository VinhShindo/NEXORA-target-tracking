"""
NEXORA - AI Target Following Robot
Professional Robotics Dashboard
ULTRA OPTIMIZED - 26 FPS target
FIXED: Data broadcasting
UPDATED: PID Steering, PID Distance, Robot Telemetry, Logging
MULTI-MODEL SUPPORT: Person, Ball, Both
"""

import os
import sys
import json
import asyncio
import logging
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import psutil
import csv
import dataclasses
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yaml

# Import services
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.tracking_service import TrackingService
from services.robot_service import RobotService, RobotState
from services.dataset_service import DatasetService

# Import controllers
from controllers.steering_pid import SteeringPID
from controllers.distance_pid import DistancePID

# Setup
BASE_DIR = Path(__file__).parent.absolute()
log_dir = BASE_DIR / "output" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# ============= FIX UNICODE ENCODING =============
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# Custom logging handler
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                msg = msg.encode('ascii', 'ignore').decode('ascii')
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except:
                self.handleError(record)
        except Exception:
            self.handleError(record)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log", encoding='utf-8'),
        SafeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============= LOAD CONFIG =============
def load_config():
    """Load configuration from config.yaml"""
    config_paths = [
        BASE_DIR / "configs" / "config.yaml",
        BASE_DIR.parent / "src" / "configs" / "config.yaml"
    ]
    
    default_config = {
        'camera': {
            'id': 1,
            'width': 640,
            'height': 480,
            'fps': 30,
            'use_mjpeg': True
        },
        'display': {
            'width': 1152,
            'height': 864,
            'scale_up': True,
            'quality': 85,
            'preserve_aspect_ratio': True
        },
        'detection': {
            'default_mode': 'both',
            'models': {
                'person': {
                    'path': 'weights/yolov8n_person_lr001.pt',
                    'classes': [0],
                    'names': ['person'],
                    'color': '#00ff88',
                    'icon': '👤'
                },
                'ball': {
                    'path': 'weights/yolov8n_ball.pt',
                    'classes': [0],
                    'names': ['ball'],
                    'color': '#ff8800',
                    'icon': '⚽'
                },
                'both': {
                    'path': 'weights/yolov8n_person.pt',
                    'classes': [0, 1],
                    'names': ['person', 'ball'],
                    'color': '#00d4ff',
                    'icon': '🎯'
                }
            },
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'device': 'cpu',
            'process_every_n_frames': 1,
            'inference_size': 320
        },
        'tracking': {
            'track_thresh': 0.4,
            'high_thresh': 0.6,
            'match_thresh': 0.7,
            'frame_rate': 30
        },
        'robot': {
            'frame_width': 640,
            'frame_height': 480,
            'pid': {
                'kp': 0.05,
                'ki': 0.001,
                'kd': 0.01
            },
            'distance_pid': {
                'kp': 0.1,
                'ki': 0.002,
                'kd': 0.02
            },
            'target_height': 150
        },
        'performance': {
            'websocket': {
                'frame_quality': 80,
                'metrics_fps': 1
            },
            'jpeg_quality': 85,
            'resize_interpolation': 'linear'
        },
        'esp32': {
            'send_interval': 0.1,
            'timeout': 1.0
        }
    }
    
    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break
    
    if config_file:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config:
                    for key in default_config:
                        if key not in config:
                            config[key] = default_config[key]
                        elif isinstance(default_config[key], dict):
                            for sub_key in default_config[key]:
                                if sub_key not in config[key]:
                                    config[key][sub_key] = default_config[key][sub_key]
                else:
                    config = default_config
                logger.info(f"[OK] Config loaded from: {config_file}")
                return config
        except Exception as e:
            logger.warning(f"[WARN] Error loading config: {e}, using default config")
            return default_config
    else:
        logger.warning(f"[WARN] Config file not found, using default config")
        return default_config

# Load config
config = load_config()

# Create directories
(BASE_DIR / "output" / "snapshots").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "output" / "recordings").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "output" / "datasets").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "weights").mkdir(parents=True, exist_ok=True)

# Create logs directory for CSV logging
telemetry_log_dir = BASE_DIR / "logs"
telemetry_log_dir.mkdir(parents=True, exist_ok=True)
telemetry_log_file = telemetry_log_dir / "telemetry.csv"

# ============= SERVICES INITIALIZATION =============
camera_service = CameraService()
detection_service = DetectionService()
tracking_service = TrackingService()
dataset_service = DatasetService()

# ===== LOAD MODELS FROM CONFIG =====
detection_service.load_models(config.get('detection', {}))

# Configure robot service with PID settings
robot_config = config.get('robot', {})
pid_config = {
    'steering_pid': robot_config.get('pid', {}),
    'distance_pid': robot_config.get('distance_pid', {}),
    'target_height': robot_config.get('target_height', 150)
}
robot_service = RobotService(pid_config)

# Apply config to services
camera_config = config.get('camera', {})
camera_service.default_camera_id = camera_config.get('id', 1)
camera_service.default_width = camera_config.get('width', 640)
camera_service.default_height = camera_config.get('height', 480)
camera_service.default_fps = camera_config.get('fps', 30)
camera_service.use_mjpeg = camera_config.get('use_mjpeg', True)

# Display config
display_config = config.get('display', {})
display_width = display_config.get('width', 1152)
display_height = display_config.get('height', 864)
display_scale_up = display_config.get('scale_up', True)
display_quality = display_config.get('quality', 85)
preserve_aspect_ratio = display_config.get('preserve_aspect_ratio', True)

detection_config = config.get('detection', {})
detection_service.default_model_path = detection_config.get('model_path', str(BASE_DIR.parent / "weights" / "yolov8n_person.pt"))
detection_service.default_confidence = detection_config.get('confidence_threshold', 0.5)
detection_service.default_iou = detection_config.get('iou_threshold', 0.45)
detection_service.default_device = detection_config.get('device', 'cpu')
detection_service.process_every_n_frames = detection_config.get('process_every_n_frames', 1)
detection_service.inference_size = detection_config.get('inference_size', 320)

tracking_config = config.get('tracking', {})
tracking_service.track_thresh = tracking_config.get('track_thresh', 0.4)
tracking_service.high_thresh = tracking_config.get('high_thresh', 0.6)
tracking_service.match_thresh = tracking_config.get('match_thresh', 0.7)

robot_frame_width = robot_config.get('frame_width', 640)
robot_frame_height = robot_config.get('frame_height', 480)
robot_service.update_frame_center(robot_frame_width, robot_frame_height)

# Performance config
jpeg_quality = config.get('performance', {}).get('jpeg_quality', 85)

# ESP32 config
esp32_config = config.get('esp32', {})
esp32_send_interval = esp32_config.get('send_interval', 0.1)
esp32_timeout = esp32_config.get('timeout', 1.0)

logger.info(f"[CONFIG] Camera: {camera_service.default_width}x{camera_service.default_height} @ {camera_service.default_fps}fps")
logger.info(f"[CONFIG] Display: {display_width}x{display_height}, Quality: {display_quality}")
logger.info(f"[CONFIG] Detection interval: {detection_service.process_every_n_frames} frames")
logger.info(f"[CONFIG] Inference size: {detection_service.inference_size}")
logger.info(f"[CONFIG] JPEG quality: {jpeg_quality}")
logger.info(f"[CONFIG] Steering PID: kp={robot_service.steering_pid.kp}, ki={robot_service.steering_pid.ki}, kd={robot_service.steering_pid.kd}")
logger.info(f"[CONFIG] Distance PID: kp={robot_service.distance_pid.kp}, ki={robot_service.distance_pid.ki}, kd={robot_service.distance_pid.kd}")
logger.info(f"[CONFIG] Target Height: {robot_service.get_telemetry().target_height}")  # Sửa lỗi tại đây
logger.info(f"[CONFIG] Detection models: {list(detection_service.model_paths.keys())}")
logger.info(f"[CONFIG] Current detection mode: {detection_service.current_mode}")

# ============= GLOBAL STATE =============
class AppState:
    def __init__(self):
        self.is_detecting = False
        self.is_recording = False
        self.is_following = False
        self.current_frame = None
        self.detections = []
        self.tracks = []
        self.fps = 0
        self.inference_time = 0
        self.total_persons = 0
        self.total_balls = 0
        self.current_target = None
        self.selected_target_id = None
        self.recording_frames = []
        self.frame_clients = set()
        self.metrics_clients = set()
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.processing_task = None
        self.metrics_task = None
        self.esp32_task = None
        self.system_metrics = {
            'cpu': 0,
            'ram': 0,
            'storage': 0
        }
        self.detection_updated = False
        self.last_tracks = []
        
        # Frame buffer
        self.latest_frame = None
        self.latest_tracks = []
        self.latest_target = None
        self.frame_counter = 0
        self.display_frame = None
        
        # ESP32 clients
        self.esp32_clients = set()
        self.last_esp32_command = None
        self.esp32_command_time = 0
        
        # Performance
        self.frame_times = deque(maxlen=30)
        self.target_fps = 26

state = AppState()

# ============= PYDANTIC MODELS =============
class CameraConfig(BaseModel):
    camera_id: int = 1
    width: int = 640
    height: int = 480
    fps: int = 30

class DetectionConfig(BaseModel):
    model_path: str = str(BASE_DIR.parent / "weights" / "yolov8n_person.pt")
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"

class TargetSelection(BaseModel):
    track_id: int

class ESP32Command(BaseModel):
    linear: float = 0.0
    angular: float = 0.0

class PIDConfig(BaseModel):
    kp: float = 0.05
    ki: float = 0.001
    kd: float = 0.01

class RobotPIDUpdate(BaseModel):
    steering: Optional[PIDConfig] = None
    distance: Optional[PIDConfig] = None
    target_height: Optional[float] = None

# ============= TELEMETRY LOGGING =============
def log_telemetry(telemetry):
    """Ghi dữ liệu telemetry vào file CSV"""
    try:
        file_exists = telemetry_log_file.exists()
        with open(telemetry_log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'target_id', 'error_x', 'distance_error', 
                               'servo_angle', 'motor_speed', 'pid_steering_output', 
                               'pid_distance_output', 'fps'])
            writer.writerow([
                datetime.now().isoformat(),
                telemetry.target_id if telemetry.target_id else '',
                f"{telemetry.error_x:.2f}",
                f"{telemetry.distance_error:.2f}",
                f"{telemetry.servo_angle:.1f}",
                f"{telemetry.motor_speed:.1f}",
                f"{telemetry.pid_steering_output:.3f}",
                f"{telemetry.pid_distance_output:.3f}",
                f"{telemetry.fps:.1f}"
            ])
    except Exception as e:
        logger.error(f"Telemetry logging error: {e}")

# ============= LIFESPAN =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[START] Starting NEXORA Robotics Dashboard...")
    state.processing_task = asyncio.create_task(process_and_broadcast())
    state.metrics_task = asyncio.create_task(update_metrics())
    state.esp32_task = asyncio.create_task(esp32_broadcast_loop())
    
    try:
        camera_id = camera_service.default_camera_id
        camera_service.start(camera_id)
        logger.info(f"[OK] Camera {camera_id} initialized")
    except Exception as e:
        logger.warning(f"[WARN] Camera error: {e}")
    
    yield
    
    logger.info("[STOP] Shutting down NEXORA Dashboard...")
    if state.processing_task:
        state.processing_task.cancel()
    if state.metrics_task:
        state.metrics_task.cancel()
    if state.esp32_task:
        state.esp32_task.cancel()
    camera_service.stop()
    for client in list(state.frame_clients):
        try:
            await client.close()
        except Exception:
            pass
    for client in list(state.metrics_clients):
        try:
            await client.close()
        except Exception:
            pass
    for client in list(state.esp32_clients):
        try:
            await client.close()
        except Exception:
            pass

# ============= FASTAPI APP =============
app = FastAPI(
    title="NEXORA Robotics Dashboard",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# ============= ROUTES =============

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "camera": camera_service.is_connected,
        "detection": state.is_detecting,
        "following": state.is_following,
        "recording": state.is_recording,
        "fps": state.fps,
        "targets": len(state.tracks),
        "persons": state.total_persons,
        "balls": state.total_balls,
        "selected_target": state.selected_target_id,
        "esp32_connected": len(state.esp32_clients) > 0,
        "detection_mode": detection_service.current_mode
    }

@app.get("/api/robot/status")
async def get_robot_status():
    """API endpoint trả về robot telemetry đầy đủ"""
    telemetry = robot_service.get_telemetry()
    return {
        "tracking": robot_service.state == RobotState.FOLLOWING,
        "target_id": telemetry.target_id,
        "servo_angle": round(telemetry.servo_angle, 1),
        "motor_speed": round(telemetry.motor_speed, 1),
        "error_x": round(telemetry.error_x, 2),
        "distance_error": round(telemetry.distance_error, 2),
        "bbox_height": round(telemetry.bbox_height, 2),
        "pid_steering_output": round(telemetry.pid_steering_output, 3),
        "pid_distance_output": round(telemetry.pid_distance_output, 3),
        "fps": round(telemetry.fps, 1)
    }

@app.post("/api/robot/pid")
async def update_pid_params(pid_update: RobotPIDUpdate):
    """Cập nhật PID parameters realtime"""
    try:
        steering = pid_update.steering.dict() if pid_update.steering else None
        distance = pid_update.distance.dict() if pid_update.distance else None
        robot_service.update_pid_params(steering, distance, pid_update.target_height)
        return {"success": True, "message": "PID parameters updated"}
    except Exception as e:
        logger.error(f"PID update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/camera/start")
async def start_camera(config: CameraConfig):
    try:
        success = camera_service.start(config.camera_id, config.width, config.height)
        if success:
            return {"success": True, "message": "Camera started"}
        raise HTTPException(status_code=500, detail="Failed to start camera")
    except Exception as e:
        logger.error(f"Camera error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/camera/stop")
async def stop_camera():
    camera_service.stop()
    state.is_detecting = False
    return {"success": True, "message": "Camera stopped"}

@app.post("/api/detection/start")
async def start_detection(config: DetectionConfig):
    try:
        if not camera_service.is_connected:
            raise HTTPException(status_code=400, detail="Camera not connected")
        
        success = detection_service.load_model(
            config.model_path,
            config.confidence_threshold,
            config.iou_threshold,
            config.device
        )
        
        if success:
            state.is_detecting = True
            state.detection_updated = True
            tracking_service.reset()
            return {"success": True, "message": "Detection active"}
        raise HTTPException(status_code=500, detail="Model loading failed")
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detection/stop")
async def stop_detection():
    """Dừng detection và xóa tất cả tracks + target"""
    state.is_detecting = False
    state.is_following = False
    state.tracks = []
    state.detections = []
    state.current_target = None
    state.selected_target_id = None
    state.total_persons = 0
    state.total_balls = 0
    robot_service.stop()
    tracking_service.reset()
    detection_service.set_selected_target(None)
    logger.info("[STOP] Detection stopped, all tracks cleared")
    return {"success": True, "message": "Detection stopped, all cleared"}

@app.post("/api/target/select")
async def select_target(selection: TargetSelection):
    """Chọn target và LOCK để theo dõi cố định"""
    state.selected_target_id = selection.track_id
    state.is_following = True
    detection_service.set_selected_target(selection.track_id)
    
    # LOCK target với fingerprint
    frame = state.current_frame
    tracking_service.lock_target(selection.track_id, frame)
    
    robot_service.start_following()
    logger.info(f"[TARGET] Target LOCKED: ID {selection.track_id}")
    return {"success": True, "track_id": selection.track_id, "locked": True}

@app.post("/api/target/release")
async def release_target():
    """Hủy target và xóa lock"""
    state.selected_target_id = None
    state.is_following = False
    robot_service.stop()
    detection_service.set_selected_target(None)
    tracking_service.unlock_target()
    logger.info("[RELEASE] Target UNLOCKED")
    return {"success": True, "message": "Target released"}

@app.post("/api/follow/start")
async def start_follow():
    if state.selected_target_id is None:
        raise HTTPException(status_code=400, detail="No target selected")
    state.is_following = True
    robot_service.start_following()
    return {"success": True, "message": "Following started"}

@app.post("/api/follow/stop")
async def stop_follow():
    state.is_following = False
    robot_service.stop()
    return {"success": True, "message": "Following stopped"}

@app.post("/api/dataset/capture")
async def capture_image():
    if state.current_frame is None:
        raise HTTPException(status_code=400, detail="No frame available")
    frame_copy = state.current_frame.copy()
    filepath = dataset_service.capture_frame(frame_copy)
    return {
        "success": True,
        "message": "Image captured",
        "path": str(filepath)
    }

@app.post("/api/dataset/record/start")
async def start_recording():
    if state.is_recording:
        return {"success": False, "message": "Already recording"}
    state.is_recording = True
    state.recording_frames = []
    return {"success": True, "message": "Recording started"}

@app.post("/api/dataset/record/stop")
async def stop_recording():
    if not state.is_recording:
        return {"success": False, "message": "Not recording"}
    state.is_recording = False
    if state.recording_frames:
        filepath = dataset_service.save_recording(state.recording_frames)
        state.recording_frames = []
        return {
            "success": True,
            "message": "Recording saved",
            "path": str(filepath)
        }
    return {"success": False, "message": "No frames recorded"}

@app.post("/api/dataset/export")
async def export_dataset():
    try:
        result = dataset_service.export_dataset()
        return {
            "success": True,
            "message": "Dataset exported",
            "path": result
        }
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/stats")
async def get_dataset_stats():
    return dataset_service.get_stats()

@app.get("/api/config")
async def get_config():
    return {
        "camera": {
            "id": camera_service.default_camera_id,
            "width": camera_service.default_width,
            "height": camera_service.default_height,
            "fps": camera_service.default_fps,
            "use_mjpeg": camera_service.use_mjpeg
        },
        "display": {
            "width": display_width,
            "height": display_height,
            "scale_up": display_scale_up,
            "quality": display_quality,
            "preserve_aspect_ratio": preserve_aspect_ratio
        },
        "detection": {
            "current_mode": detection_service.current_mode,
            "available_modes": detection_service.get_available_modes(),
            "model_status": detection_service.get_model_status()
        },
        "robot": {
            "frame_width": robot_frame_width,
            "frame_height": robot_frame_height,
            "pid": {
                "kp": robot_service.steering_pid.kp,
                "ki": robot_service.steering_pid.ki,
                "kd": robot_service.steering_pid.kd
            },
            "distance_pid": {
                "kp": robot_service.distance_pid.kp,
                "ki": robot_service.distance_pid.ki,
                "kd": robot_service.distance_pid.kd
            },
            "target_height": robot_service.get_telemetry().target_height
        },
        "esp32": {
            "send_interval": esp32_send_interval,
            "timeout": esp32_timeout
        }
    }

# ============= ESP32 API ENDPOINTS =============
@app.get("/api/esp32/status")
async def get_esp32_status():
    return {
        'connected': len(state.esp32_clients) > 0,
        'clients_count': len(state.esp32_clients),
        'is_following': state.is_following,
        'target_selected': state.selected_target_id is not None,
        'tracks_count': len(state.tracks),
        'last_command_time': state.esp32_command_time
    }

@app.get("/api/esp32/command")
async def get_esp32_command():
    if state.last_esp32_command is None:
        return {'error': 'No command available'}
    return {
        'timestamp': state.esp32_command_time,
        'command': state.last_esp32_command
    }

@app.post("/api/esp32/command/manual")
async def send_manual_command(cmd: ESP32Command):
    if not state.esp32_clients:
        raise HTTPException(status_code=400, detail="No ESP32 connected")
    
    command = {
        'type': 'control',
        'timestamp': time.time(),
        'command_id': int(time.time() * 1000),
        'is_following': False,
        'target_detected': False,
        'target_selected': False,
        'targets_count': 0,
        'robot': {
            'state': 'MANUAL',
            'direction': 'FORWARD' if cmd.linear > 0 else ('BACKWARD' if cmd.linear < 0 else 'STOP'),
            'linear_velocity': cmd.linear,
            'angular_velocity': cmd.angular,
            'distance': 0,
            'offset_x': 0
        },
        'target': None
    }
    
    message = json.dumps(command)
    for client in list(state.esp32_clients):
        try:
            await client.send_text(message)
        except Exception as e:
            logger.error(f"[ESP32] Manual send error: {e}")
            state.esp32_clients.discard(client)
    
    state.last_esp32_command = command
    state.esp32_command_time = time.time()
    return {'success': True, 'message': 'Command sent'}

@app.post("/api/esp32/command/stop")
async def send_stop_command():
    if not state.esp32_clients:
        raise HTTPException(status_code=400, detail="No ESP32 connected")
    
    command = {
        'type': 'control',
        'timestamp': time.time(),
        'command_id': int(time.time() * 1000),
        'is_following': False,
        'target_detected': False,
        'target_selected': False,
        'targets_count': 0,
        'robot': {
            'state': 'EMERGENCY_STOP',
            'direction': 'STOP',
            'linear_velocity': 0.0,
            'angular_velocity': 0.0,
            'distance': 0,
            'offset_x': 0
        },
        'target': None
    }
    
    message = json.dumps(command)
    for client in list(state.esp32_clients):
        try:
            await client.send_text(message)
        except Exception as e:
            logger.error(f"[ESP32] Stop command error: {e}")
            state.esp32_clients.discard(client)
    
    robot_service.stop()
    state.last_esp32_command = command
    state.esp32_command_time = time.time()
    return {'success': True, 'message': 'Emergency stop sent'}

# ============= DETECTION MODE API =============
@app.get("/api/detection/modes")
async def get_detection_modes():
    """Lấy danh sách các mode có sẵn và trạng thái"""
    return {
        "current_mode": detection_service.current_mode,
        "available_modes": detection_service.get_available_modes(),
        "model_status": detection_service.get_model_status(),
        "loading_status": detection_service.get_loading_status(),
        "performance": detection_service.get_performance_stats()
    }

@app.post("/api/detection/mode")
async def set_detection_mode(request: Request):
    """Chuyển đổi chế độ detection: person, ball, both"""
    try:
        data = await request.json()
        mode = data.get('mode')
        
        if not mode:
            raise HTTPException(status_code=400, detail="Mode is required")
        
        if mode not in detection_service.get_available_modes():
            raise HTTPException(status_code=400, detail=f"Mode {mode} not available")
        
        success = detection_service.switch_mode(mode)
        if success:
            # Reset tracker khi đổi mode
            tracking_service.reset()
            return {
                "success": True, 
                "message": f"Switched to {mode} mode",
                "current_mode": mode
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to switch mode")
            
    except Exception as e:
        logger.error(f"Switch mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detection/loading")
async def get_detection_loading():
    """Lấy trạng thái loading model"""
    return detection_service.get_loading_status()

# ============= WEBSOCKET ENDPOINTS =============
@app.websocket("/ws")
async def websocket_legacy(websocket: WebSocket):
    await websocket.accept()
    state.frame_clients.add(websocket)
    try:
        while True:
            try:
                message = await websocket.receive()
                if message['type'] == 'websocket.receive':
                    if 'text' in message:
                        try:
                            data = json.loads(message['text'])
                            if data.get('type') == 'ping':
                                await websocket.send_text(json.dumps({'type': 'pong'}))
                        except:
                            pass
            except WebSocketDisconnect:
                break
            except Exception as e:
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.frame_clients.discard(websocket)

@app.websocket("/ws/frame")
async def websocket_frame(websocket: WebSocket):
    await websocket.accept()
    state.frame_clients.add(websocket)
    logger.info(f"[CONN] Frame WebSocket connected. Clients: {len(state.frame_clients)}")
    try:
        while True:
            try:
                message = await websocket.receive()
                if message['type'] == 'websocket.receive':
                    if 'text' in message:
                        try:
                            data = json.loads(message['text'])
                            if data.get('type') == 'ping':
                                await websocket.send_text(json.dumps({'type': 'pong'}))
                        except:
                            pass
            except WebSocketDisconnect:
                break
            except Exception as e:
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.frame_clients.discard(websocket)
        logger.info(f"[CONN] Frame WebSocket disconnected. Clients: {len(state.frame_clients)}")

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    state.metrics_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            metrics_data = {
                'type': 'metrics',
                'system': state.system_metrics,
                'fps': state.fps,
                'inference_time': state.inference_time,
                'total_persons': state.total_persons,
                'total_balls': state.total_balls,
                'total_tracks': len(state.tracks),
                'selected_target': state.selected_target_id,
                'robot': robot_service.get_status(),
                'robot_telemetry': dataclasses.asdict(robot_service.get_telemetry()),
                'dataset': dataset_service.get_stats(),
                'detection_enabled': state.is_detecting,
                'following_enabled': state.is_following,
                'esp32_connected': len(state.esp32_clients) > 0,
                'detection_mode': detection_service.current_mode
            }
            await websocket.send_text(json.dumps(metrics_data))
    except WebSocketDisconnect:
        pass
    finally:
        state.metrics_clients.discard(websocket)

@app.websocket("/ws/esp32")
async def websocket_esp32(websocket: WebSocket):
    await websocket.accept()
    state.esp32_clients.add(websocket)
    logger.info(f"[ESP32] ESP32 WebSocket connected. Clients: {len(state.esp32_clients)}")
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                if message['type'] == 'websocket.receive':
                    if 'text' in message:
                        try:
                            data = json.loads(message['text'])
                            if data.get('type') == 'ping':
                                await websocket.send_text(json.dumps({'type': 'pong'}))
                        except:
                            pass
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except Exception as e:
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.esp32_clients.discard(websocket)
        logger.info(f"[ESP32] ESP32 WebSocket disconnected. Clients: {len(state.esp32_clients)}")

# ============= MAIN PROCESSING LOOP =============
def fast_resize_with_aspect_ratio(image, target_width, target_height):
    if image is None:
        return None
    h, w = image.shape[:2]
    if w == target_width and h == target_height:
        return image
    target_ratio = target_width / target_height
    image_ratio = w / h
    if target_ratio > image_ratio:
        new_width = int(target_height * image_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / image_ratio)
    if new_width > 0 and new_height > 0:
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if new_width == target_width and new_height == target_height:
            return resized
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        return canvas
    return image

async def process_and_broadcast():
    """Main processing loop with PID control and telemetry"""
    frame_counter = 0
    scale_width = display_width
    scale_height = display_height
    jpeg_quality = config.get('performance', {}).get('jpeg_quality', 85)
    
    # Điều chỉnh lại mục tiêu FPS xuống 24 để tránh quá tải CPU
    state.target_fps = 24
    
    logger.info(f"[DISPLAY] Scaling from {camera_service.default_width}x{camera_service.default_height} to {scale_width}x{scale_height}")
    
    while True:
        try:
            start_loop = time.perf_counter()
            
            if not camera_service.is_connected:
                await asyncio.sleep(0.01)
                continue
            
            frame = camera_service.read()
            
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            frame_counter += 1
            state.current_frame = frame
            state.frame_count += 1
            
            # FPS calculation
            current_time = time.time()
            if current_time - state.last_frame_time >= 1.0:
                state.fps = state.frame_count
                state.frame_count = 0
                state.last_frame_time = current_time
            
            annotated_frame = frame.copy()
            
            # ==============================================================
            # TỐI ƯU CPU: Chiến lược Dynamic Adaptive Interval
            # Khi ít người, interval = 2 (YOLO chạy 15fps) -> FPS hiển thị 24-26
            # Khi đông người, interval = 3 (YOLO chạy 10fps) -> FPS hiển thị 17-19
            # ==============================================================
            current_track_count = len(state.tracks)
            target_interval = 2  # Mặc định cách 1 frame (1/2 tốc độ)
            
            if current_track_count > 4:
                target_interval = 3  # Quá đông, chỉ detect 1/3 số frame
                
            # Chỉ gán nếu khác biệt để đỡ tốn tài nguyên
            if detection_service.process_every_n_frames != target_interval:
                detection_service.set_process_interval(target_interval)
            
            process_interval = detection_service.process_every_n_frames
            # ================================================================

            if state.is_detecting and detection_service.is_model_loaded:
                should_detect = (frame_counter % process_interval == 0)
                
                if should_detect:
                    try:
                        start_time = time.perf_counter()
                        detections = detection_service.detect(frame)
                        state.inference_time = (time.perf_counter() - start_time) * 1000
                        
                        if detections:
                            state.detections = detections
                            state.detection_updated = True
                            tracks = tracking_service.update(detections, frame)
                            state.tracks = tracks
                            state.last_tracks = tracks
                        else:
                            if state.tracks:
                                try:
                                    state.tracks = tracking_service.predict(frame)
                                except Exception as e:
                                    logger.error(f"Predict error: {e}")
                            state.detection_updated = False
                        
                        state.latest_frame = annotated_frame
                        state.latest_tracks = state.tracks.copy()
                    except Exception as e:
                        logger.error(f"Detection error: {e}")
                else:
                    # Frame không detect: Kalman dự đoán (giữ mượt video)
                    if state.tracks:
                        try:
                            state.tracks = tracking_service.predict(frame)
                        except Exception as e:
                            logger.error(f"Predict error: {e}")
            else:
                state.tracks = []
            
            # Update statistics
            if state.detections:
                state.total_persons = sum(1 for d in state.detections if d.get('class') == 'person')
                state.total_balls = sum(1 for d in state.detections if d.get('class') == 'ball')
            else:
                state.total_persons = 0
                state.total_balls = 0
            
            # ============= ROBOT PID CONTROL =============
            if state.selected_target_id is not None:
                locked_id = tracking_service.get_locked_target_id()
                
                if locked_id is not None:
                    target_track = tracking_service.get_track_by_id(locked_id)
                    
                    if target_track and target_track.get('bbox'):
                        state.current_target = target_track
                        state.selected_target_id = locked_id
                        
                        if state.is_following:
                            try:
                                robot_service.update_target(
                                    target_track,
                                    camera_service.default_width,
                                    camera_service.default_height,
                                    state.fps
                                )
                                log_telemetry(robot_service.get_telemetry())
                            except Exception as e:
                                logger.error(f"Robot update error: {e}")
                        
                        state.latest_target = target_track
                    else:
                        state.current_target = None
                        state.latest_target = None
                        
                        if state.is_following:
                            try:
                                robot_service.handle_target_lost()
                            except Exception as e:
                                logger.error(f"Target lost error: {e}")
                else:
                    target_track = tracking_service.get_track_by_id(state.selected_target_id)
                    
                    if target_track and target_track.get('bbox'):
                        state.current_target = target_track
                        
                        if state.is_following:
                            try:
                                robot_service.update_target(
                                    target_track,
                                    camera_service.default_width,
                                    camera_service.default_height,
                                    state.fps
                                )
                                log_telemetry(robot_service.get_telemetry())
                            except Exception as e:
                                logger.error(f"Robot update error: {e}")
                        
                        state.latest_target = target_track
                    else:
                        state.current_target = None
                        state.latest_target = None
                        
                        if state.is_following:
                            try:
                                robot_service.handle_target_lost()
                            except Exception as e:
                                logger.error(f"Target lost error: {e}")
            else:
                state.current_target = None
                state.latest_target = None
            
            # ============= VẼ ANNOTATIONS + OVERLAY =============
            try:
                annotated_frame = detection_service.draw_tracks(annotated_frame, state.tracks)
                annotated_frame = robot_service.draw_overlay(annotated_frame)
            except Exception as e:
                logger.error(f"Draw error: {e}")
            
            # Scale for display
            if display_scale_up and annotated_frame is not None:
                try:
                    if preserve_aspect_ratio:
                        display_frame = fast_resize_with_aspect_ratio(annotated_frame, scale_width, scale_height)
                    else:
                        display_frame = cv2.resize(annotated_frame, (scale_width, scale_height),
                                                 interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    logger.error(f"Resize error: {e}")
                    display_frame = annotated_frame
            else:
                display_frame = annotated_frame
            
            state.display_frame = display_frame
            
            # Recording
            if state.is_recording and annotated_frame is not None:
                state.recording_frames.append(annotated_frame.copy())
            
            # ============= BROADCAST =============
            if state.frame_clients and display_frame is not None:
                try:
                    encode_param = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    _, buffer = cv2.imencode('.jpg', display_frame, encode_param)
                    
                    target_info = None
                    if state.selected_target_id is not None and state.current_target:
                        track = state.current_target
                        target_info = {
                            'id': track.get('track_id'),
                            'class': track.get('class', 'person'),
                            'confidence': track.get('confidence', 0),
                            'status': 'ACTIVE' if state.is_following else 'SELECTED',
                            'bbox': track.get('bbox')
                        }
                    
                    tracks_data = []
                    for t in state.tracks[:20]:
                        tracks_data.append({
                            'track_id': t.get('track_id'),
                            'bbox': t.get('bbox'),
                            'class': t.get('class', 'person'),
                            'confidence': t.get('confidence', 0),
                            'age': t.get('frame_count', 0),
                            'lost': t.get('lost', 0)
                        })
                    
                    robot_telemetry = dataclasses.asdict(robot_service.get_telemetry())
                    
                    frame_data = {
                        'type': 'frame',
                        'fps': state.fps,
                        'inference_time': state.inference_time,
                        'display_width': scale_width,
                        'display_height': scale_height,
                        'source_width': camera_service.default_width,
                        'source_height': camera_service.default_height,
                        'preserve_aspect_ratio': preserve_aspect_ratio,
                        'tracks': tracks_data,
                        'selected_target': state.selected_target_id,
                        'target': target_info,
                        'statistics': {
                            'total_persons': state.total_persons,
                            'total_balls': state.total_balls,
                            'selected_target': state.selected_target_id
                        },
                        'robot': robot_service.get_status(),
                        'robot_telemetry': robot_telemetry,
                        'esp32_connected': len(state.esp32_clients) > 0,
                        'timestamp': time.time(),
                        'detection_enabled': state.is_detecting,
                        'following_enabled': state.is_following,
                        'detection_mode': detection_service.current_mode
                    }
                    
                    json_str = json.dumps(frame_data)
                    disconnected = set()
                    for client in state.frame_clients:
                        try:
                            await client.send_text(json_str)
                            await client.send_bytes(buffer.tobytes())
                        except Exception:
                            disconnected.add(client)
                    
                    for client in disconnected:
                        state.frame_clients.discard(client)
                        
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")
            
            # Dynamic sleep (Mục tiêu 24 FPS để CPU không bị ép quá mức)
            loop_time = time.perf_counter() - start_loop
            state.frame_times.append(loop_time)
            target_frame_time = 1.0 / state.target_fps
            sleep_time = max(0, target_frame_time - loop_time - 0.0005)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.0005)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            await asyncio.sleep(0.01)

async def esp32_broadcast_loop():
    """Broadcast control commands to ESP32 - TỐI ƯU HÓA"""
    send_interval = esp32_send_interval
    
    while True:
        try:
            if state.esp32_clients:
                # Lấy telemetry từ robot service
                telemetry = robot_service.get_telemetry()
                robot_status = robot_service.get_status()
                
                # Lấy thông tin target
                target_info = None
                if state.selected_target_id is not None:
                    target_track = tracking_service.get_track_by_id(state.selected_target_id)
                    if target_track and target_track.get('bbox'):
                        target_info = {
                            'id': target_track.get('track_id'),
                            'bbox': target_track.get('bbox'),
                            'class': target_track.get('class', 'person'),
                            'confidence': target_track.get('confidence', 0)
                        }
                
                # ===== TẠO JSON TỐI ƯU CHO ESP32 =====
                command = {
                    'type': 'control',
                    'timestamp': int(time.time() * 1000),
                    'command_id': int(time.time() * 1000),
                    'is_following': state.is_following,
                    'target_detected': target_info is not None,
                    'target_id': state.selected_target_id,
                    'targets_count': len(state.tracks),
                    'control': {
                        'motor_speed': round(telemetry.motor_speed, 1),
                        'linear_velocity': round(robot_status.get('linear_velocity', 0), 3),
                        'angular_velocity': round(robot_status.get('angular_velocity', 0), 3),
                        'servo_angle': round(telemetry.servo_angle, 1),
                        'direction': robot_status.get('direction', 'STOP'),
                        'state': robot_status.get('state', 'IDLE'),
                    },
                    'error': {
                        'steering_error': round(telemetry.error_x, 2),
                        'distance_error': round(telemetry.distance_error, 2),
                        'pid_steering': round(telemetry.pid_steering_output, 3),
                        'pid_distance': round(telemetry.pid_distance_output, 3),
                    },
                    'target': {
                        'id': target_info.get('id') if target_info else None,
                        'class': target_info.get('class') if target_info else None,
                        'confidence': round(target_info.get('confidence', 0), 2) if target_info else 0,
                        'bbox_center': [
                            round((target_info.get('bbox', [0,0,0,0])[0] + target_info.get('bbox', [0,0,0,0])[2]) / 2, 1) if target_info else 0,
                            round((target_info.get('bbox', [0,0,0,0])[1] + target_info.get('bbox', [0,0,0,0])[3]) / 2, 1) if target_info else 0
                        ],
                        'bbox_height': round(telemetry.bbox_height, 1) if target_info else 0,
                        'bbox_width': round(telemetry.bbox_width, 1) if target_info else 0,
                    },
                    'distance': round(robot_status.get('distance', 0), 2),
                    'system': {
                        'fps': round(state.fps, 1),
                        'inference_time': round(state.inference_time, 1),
                        'esp32_connected': len(state.esp32_clients) > 0,
                        'detection_mode': detection_service.current_mode
                    }
                }
                
                state.last_esp32_command = command
                state.esp32_command_time = time.time()
                
                message = json.dumps(command)
                
                disconnected = set()
                for client in list(state.esp32_clients):
                    try:
                        await client.send_text(message)
                    except Exception as e:
                        logger.error(f"[ESP32] Send error: {e}")
                        disconnected.add(client)
                
                for client in disconnected:
                    state.esp32_clients.discard(client)
            
            await asyncio.sleep(send_interval)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[ESP32] Broadcast error: {e}")
            await asyncio.sleep(0.1)

async def update_metrics():
    """Update system metrics"""
    while True:
        try:
            state.system_metrics = {
                'cpu': psutil.cpu_percent(),
                'ram': psutil.virtual_memory().percent,
                'storage': psutil.disk_usage('/').percent if os.name == 'posix' else 0
            }
            await asyncio.sleep(2)
        except:
            await asyncio.sleep(5)

if __name__ == "__main__":
    os.chdir(BASE_DIR)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
        workers=1
    )