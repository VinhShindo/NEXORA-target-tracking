/**
 * NEXORA - AI Target Following Robot Dashboard
 * FIXED: Complete data display for all panels
 */

class NexoraDashboard {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.videoElement = document.getElementById('videoFeed');
        this.overlayElement = document.getElementById('videoOverlay');
        this.startTime = Date.now();
        this.currentTracks = [];
        this.selectedTargetId = null;
        this.frameCount = 0;
        this.currentFps = 0;
        this.fpsUpdateInterval = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;

        // UI Elements
        this.elements = {
            // Stats
            statPersons: document.getElementById('statPersons'),
            statBalls: document.getElementById('statBalls'),
            statFps: document.getElementById('statFps'),
            statInference: document.getElementById('statInference'),
            statTracks: document.getElementById('statTracks'),
            statActive: document.getElementById('statActive'),
            statLost: document.getElementById('statLost'),
            statSelected: document.getElementById('statSelected'),
            statCpu: document.getElementById('statCpu'),
            statRam: document.getElementById('statRam'),
            statStorage: document.getElementById('statStorage'),
            statUptime: document.getElementById('statUptime'),
            statImages: document.getElementById('statImages'),
            statVideos: document.getElementById('statVideos'),
            statStorageUsed: document.getElementById('statStorageUsed'),
            statRecording: document.getElementById('statRecording'),
            
            // Target
            targetId: document.getElementById('targetId'),
            targetClass: document.getElementById('targetClass'),
            targetConfidence: document.getElementById('targetConfidence'),
            targetTime: document.getElementById('targetTime'),
            targetStatusBadge: document.getElementById('targetStatusBadge'),
            
            // Robot Control
            robotState: document.getElementById('robotState'),
            robotDirection: document.getElementById('robotDirection'),
            robotServo: document.getElementById('robotServo'),
            robotMotor: document.getElementById('robotMotor'),
            robotErrorX: document.getElementById('robotErrorX'),
            robotDistError: document.getElementById('robotDistError'),
            
            // Header
            fpsDisplay: document.getElementById('fpsDisplay'),
            connectionStatus: document.getElementById('connectionStatus'),
            cameraStatus: document.getElementById('cameraStatus'),
            detectionStatus: document.getElementById('detectionStatus'),
            recordingStatus: document.getElementById('recordingStatus'),
            
            // HUD
            hudFps: document.getElementById('hudFps'),
            hudInference: document.getElementById('hudInference'),
            hudTargets: document.getElementById('hudTargets'),
            
            // Telemetry
            telemTargetX: document.getElementById('telemTargetX'),
            telemFrameX: document.getElementById('telemFrameX'),
            telemErrorX: document.getElementById('telemErrorX'),
            telemSteerOut: document.getElementById('telemSteerOut'),
            telemServo: document.getElementById('telemServo'),
            telemBboxW: document.getElementById('telemBboxW'),
            telemBboxH: document.getElementById('telemBboxH'),
            telemBboxA: document.getElementById('telemBboxA'),
            telemTargetH: document.getElementById('telemTargetH'),
            telemDistErr: document.getElementById('telemDistErr'),
            telemDistOut: document.getElementById('telemDistOut'),
            telemMotor: document.getElementById('telemMotor'),
            telemFPS: document.getElementById('telemFPS'),
            telemLastUp: document.getElementById('telemLastUp'),
            
            // PID Tuning
            steerKp: document.getElementById('steerKp'),
            steerKi: document.getElementById('steerKi'),
            steerKd: document.getElementById('steerKd'),
            distKp: document.getElementById('distKp'),
            distKi: document.getElementById('distKi'),
            distKd: document.getElementById('distKd'),
            targetHeight: document.getElementById('targetHeight')
        };

        this.initializeEventListeners();
        this.connectWebSocket();
        this.updateTime();
        this.startUptime();
        this.initializeVideoClick();
        this.startFpsCounter();
        
        console.log('[DASHBOARD] Initialized');
    }

    initializeEventListeners() {
        document.getElementById('btnStartCamera')?.addEventListener('click', () => this.startCamera());
        document.getElementById('btnStopCamera')?.addEventListener('click', () => this.stopCamera());
        document.getElementById('btnStartDetection')?.addEventListener('click', () => this.startDetection());
        document.getElementById('btnStopDetection')?.addEventListener('click', () => this.stopDetection());
        document.getElementById('btnSelectTarget')?.addEventListener('click', () => this.selectTarget());
        document.getElementById('btnReleaseTarget')?.addEventListener('click', () => this.releaseTarget());
        document.getElementById('btnStartFollow')?.addEventListener('click', () => this.startFollow());
        document.getElementById('btnStopFollow')?.addEventListener('click', () => this.stopFollow());
        document.getElementById('btnCapture')?.addEventListener('click', () => this.captureFrame());
        document.getElementById('btnRecordStart')?.addEventListener('click', () => this.startRecording());
        document.getElementById('btnRecordStop')?.addEventListener('click', () => this.stopRecording());
        document.getElementById('btnExport')?.addEventListener('click', () => this.exportDataset());
        document.getElementById('btnApplyPID')?.addEventListener('click', () => this.applyPID());
    }

    initializeVideoClick() {
        this.videoElement?.addEventListener('click', (e) => {
            if (this.currentTracks.length === 0) {
                this.showNotification('No targets available', 'warning');
                return;
            }

            const rect = this.videoElement.getBoundingClientRect();
            const clickX = (e.clientX - rect.left) / rect.width;
            const clickY = (e.clientY - rect.top) / rect.height;

            let closestTrack = null;
            let closestDistance = Infinity;

            for (const track of this.currentTracks) {
                if (!track.bbox || track.bbox.length < 4) continue;
                
                const [x1, y1, x2, y2] = track.bbox;
                const imgW = this.videoElement.naturalWidth || 640;
                const imgH = this.videoElement.naturalHeight || 480;
                const centerX = ((x1 + x2) / 2) / imgW;
                const centerY = ((y1 + y2) / 2) / imgH;
                
                const distance = Math.sqrt(
                    Math.pow(clickX - centerX, 2) + 
                    Math.pow(clickY - centerY, 2)
                );
                
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestTrack = track;
                }
            }

            if (closestTrack && closestDistance < 0.15) {
                this.selectTargetById(closestTrack.track_id);
            } else {
                this.showNotification('No target near click', 'warning');
            }
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/frame`;

        console.log(`[WS] Connecting to ${wsUrl}`);
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus(true);
            console.log('[WS] Connected');
        };

        this.ws.onmessage = (event) => {
            try {
                if (typeof event.data === 'string') {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } else if (event.data instanceof ArrayBuffer) {
                    this.handleBinaryFrame(event.data);
                }
            } catch (e) {
                console.error('[WS] Parse error:', e);
            }
        };

        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateConnectionStatus(false);
            console.log('[WS] Disconnected');
            
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                setTimeout(() => this.connectWebSocket(), 2000);
            }
        };

        this.ws.onerror = (error) => {
            console.error('[WS] Error:', error);
        };
    }

    handleWebSocketMessage(data) {
        if (data.type === 'frame' || data.type === 'frame_data') {
            this.currentTracks = data.tracks || [];
            
            this.updateStatistics(data);
            this.updateTargetPanel(data);
            this.updateRobotPanel(data);
            this.updateRobotTelemetry(data.robot_telemetry);
            this.updateHUD(data);
            
            if (data.system) this.updateSystemMetrics({system: data.system});
            if (data.dataset) this.updateDatasetStats({dataset: data.dataset});
        } else if (data.type === 'metrics') {
            this.updateSystemMetrics(data);
            if (data.dataset) this.updateDatasetStats(data);
            if (data.robot_telemetry) this.updateRobotTelemetry(data.robot_telemetry);
            if (data.robot) this.updateRobotPanel(data);
        }
    }

    handleBinaryFrame(buffer) {
        try {
            const blob = new Blob([buffer], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            
            this.videoElement.src = url;
            this.videoElement.onload = () => {
                URL.revokeObjectURL(url);
                this.videoElement.style.objectFit = 'contain';
            };
            
            if (this.overlayElement) {
                this.overlayElement.style.display = 'none';
            }
            
            this.frameCount++;
        } catch (error) {
            console.error('[FRAME] Error:', error);
        }
    }

    startFpsCounter() {
        this.fpsUpdateInterval = setInterval(() => {
            this.currentFps = this.frameCount;
            this.frameCount = 0;
            
            const fpsText = `FPS: ${this.currentFps}`;
            if (this.elements.hudFps) this.elements.hudFps.textContent = fpsText;
            if (this.elements.statFps) this.elements.statFps.textContent = this.currentFps;
            if (this.elements.fpsDisplay) this.elements.fpsDisplay.textContent = fpsText;
        }, 1000);
    }

    updateStatistics(data) {
        const stats = data.statistics || {};
        const tracks = data.tracks || [];

        if (this.elements.statPersons) this.elements.statPersons.textContent = stats.total_persons || 0;
        if (this.elements.statBalls) this.elements.statBalls.textContent = stats.total_balls || 0;
        if (this.elements.statInference) this.elements.statInference.textContent = data.inference_time ? `${Math.round(data.inference_time)}ms` : '0ms';
        if (this.elements.statTracks) this.elements.statTracks.textContent = tracks.length;
        
        const activeCount = tracks.filter(t => !t.lost || t.lost < 3).length;
        const lostCount = tracks.filter(t => t.lost && t.lost >= 3).length;
        if (this.elements.statActive) this.elements.statActive.textContent = activeCount;
        if (this.elements.statLost) this.elements.statLost.textContent = lostCount;
        
        if (stats.selected_target !== null && stats.selected_target !== undefined) {
            if (this.elements.statSelected) this.elements.statSelected.textContent = `ID: ${stats.selected_target}`;
            this.selectedTargetId = stats.selected_target;
        } else {
            if (this.elements.statSelected) this.elements.statSelected.textContent = '---';
            this.selectedTargetId = null;
        }
    }

    updateTargetPanel(data) {
        const target = data.target || {};
        const tracks = data.tracks || [];

        if (target && target.id !== null && target.id !== undefined) {
            if (this.elements.targetId) this.elements.targetId.textContent = target.id;
            if (this.elements.targetClass) this.elements.targetClass.textContent = target.class || 'Person';
            if (this.elements.targetConfidence) this.elements.targetConfidence.textContent = target.confidence ? `${(target.confidence * 100).toFixed(1)}%` : '0%';

            const track = tracks.find(t => t.track_id === target.id);
            if (this.elements.targetTime) this.elements.targetTime.textContent = track ? `${track.age || 0}s` : '0s';

            if (this.elements.targetStatusBadge) {
                const status = target.status || 'ACTIVE';
                this.elements.targetStatusBadge.textContent = status;
                this.elements.targetStatusBadge.className = `target-status ${status.toLowerCase()}`;
            }
        } else {
            if (this.elements.targetId) this.elements.targetId.textContent = '---';
            if (this.elements.targetClass) this.elements.targetClass.textContent = '---';
            if (this.elements.targetConfidence) this.elements.targetConfidence.textContent = '0%';
            if (this.elements.targetTime) this.elements.targetTime.textContent = '0s';
            if (this.elements.targetStatusBadge) {
                this.elements.targetStatusBadge.textContent = 'IDLE';
                this.elements.targetStatusBadge.className = 'target-status idle';
            }
        }
    }

    updateRobotPanel(data) {
        const robot = data.robot || {};
        
        const stateMap = {
            'IDLE': 'IDLE', 'FOLLOWING': 'FOLLOWING',
            'TURN LEFT': 'TURN LEFT', 'TURN RIGHT': 'TURN RIGHT',
            'STOP': 'STOP', 'SEARCHING': 'SEARCHING',
            'TARGET LOST': 'TARGET LOST'
        };

        const state = stateMap[robot.state] || robot.state || 'IDLE';
        if (this.elements.robotState) {
            this.elements.robotState.textContent = state;
            this.elements.robotState.className = `state-${state.toLowerCase().replace(' ', '-')}`;
        }
        if (this.elements.robotDirection) this.elements.robotDirection.textContent = robot.direction || 'STOP';
        if (this.elements.robotServo) this.elements.robotServo.textContent = robot.servo_angle != null ? `${robot.servo_angle}°` : '90.0°';
        if (this.elements.robotMotor) this.elements.robotMotor.textContent = robot.motor_speed != null ? robot.motor_speed.toFixed(1) : '0.0';
        if (this.elements.robotErrorX) this.elements.robotErrorX.textContent = robot.offset_x != null ? robot.offset_x.toFixed(2) : '0.00';
        
        const telem = data.robot_telemetry || {};
        if (this.elements.robotDistError) this.elements.robotDistError.textContent = telem.distance_error != null ? telem.distance_error.toFixed(2) : '0.00';
    }

    updateRobotTelemetry(telem) {
        if (!telem) return;
        
        // Steering
        if (this.elements.telemTargetX) this.elements.telemTargetX.textContent = telem.target_center_x?.toFixed(1) ?? '-';
        if (this.elements.telemFrameX) this.elements.telemFrameX.textContent = telem.frame_center_x?.toFixed(1) ?? '-';
        if (this.elements.telemErrorX) this.elements.telemErrorX.textContent = telem.error_x?.toFixed(2) ?? '-';
        if (this.elements.telemSteerOut) this.elements.telemSteerOut.textContent = telem.pid_steering_output?.toFixed(3) ?? '-';
        if (this.elements.telemServo) this.elements.telemServo.textContent = telem.servo_angle != null ? telem.servo_angle.toFixed(1) + '°' : '-';
        
        // Distance
        if (this.elements.telemBboxW) this.elements.telemBboxW.textContent = telem.bbox_width?.toFixed(1) ?? '-';
        if (this.elements.telemBboxH) this.elements.telemBboxH.textContent = telem.bbox_height?.toFixed(1) ?? '-';
        if (this.elements.telemBboxA) this.elements.telemBboxA.textContent = telem.bbox_area?.toFixed(0) ?? '-';
        if (this.elements.telemTargetH) this.elements.telemTargetH.textContent = telem.target_height ?? '-';
        if (this.elements.telemDistErr) this.elements.telemDistErr.textContent = telem.distance_error?.toFixed(2) ?? '-';
        if (this.elements.telemDistOut) this.elements.telemDistOut.textContent = telem.pid_distance_output?.toFixed(3) ?? '-';
        if (this.elements.telemMotor) this.elements.telemMotor.textContent = telem.motor_speed?.toFixed(1) ?? '-';
        
        // System
        if (this.elements.telemFPS) this.elements.telemFPS.textContent = telem.fps?.toFixed(1) ?? '-';
        if (this.elements.telemLastUp && telem.last_update) {
            this.elements.telemLastUp.textContent = new Date(telem.last_update * 1000).toLocaleTimeString();
        }
    }

    updateHUD(data) {
        if (this.elements.hudFps) this.elements.hudFps.textContent = `FPS: ${this.currentFps || data.fps || 0}`;
        if (this.elements.hudInference) this.elements.hudInference.textContent = `INF: ${data.inference_time ? Math.round(data.inference_time) : 0}ms`;
        if (this.elements.hudTargets) this.elements.hudTargets.textContent = `TARGETS: ${data.tracks ? data.tracks.length : 0}`;
    }

    updateSystemMetrics(data) {
        const system = data.system || {};
        if (this.elements.statCpu) this.elements.statCpu.textContent = `${Math.round(system.cpu || 0)}%`;
        if (this.elements.statRam) this.elements.statRam.textContent = `${Math.round(system.ram || 0)}%`;
        if (this.elements.statStorage) this.elements.statStorage.textContent = `${Math.round(system.storage || 0)}%`;
    }

    updateDatasetStats(data) {
        const dataset = data.dataset || {};
        if (this.elements.statImages) this.elements.statImages.textContent = dataset.images || 0;
        if (this.elements.statVideos) this.elements.statVideos.textContent = dataset.videos || 0;
        if (this.elements.statStorageUsed) this.elements.statStorageUsed.textContent = dataset.storage_used ? `${dataset.storage_used}MB` : '0MB';
        if (this.elements.statRecording) {
            this.elements.statRecording.textContent = dataset.recording ? 'ON' : 'OFF';
            this.elements.statRecording.style.color = dataset.recording ? '#ff0044' : '#4a5a7a';
        }
    }

    updateConnectionStatus(connected) {
        const el = this.elements.connectionStatus;
        if (!el) return;
        if (connected) {
            el.innerHTML = '<i class="fas fa-circle" style="color: #00ff88;"></i> CONNECTED';
            el.style.color = '#00ff88';
        } else {
            el.innerHTML = '<i class="fas fa-circle" style="color: #ff0044;"></i> DISCONNECTED';
            el.style.color = '#ff0044';
        }
    }

    updateTime() {
        const now = new Date();
        const timeStr = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
        document.getElementById('timeDisplay').textContent = timeStr;
        setTimeout(() => this.updateTime(), 1000);
    }

    startUptime() {
        let seconds = 0;
        setInterval(() => {
            seconds++;
            const mins = String(Math.floor(seconds / 60)).padStart(2, '0');
            const secs = String(seconds % 60).padStart(2, '0');
            if (this.elements.statUptime) this.elements.statUptime.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    showNotification(message, type = 'info') {
        const oldNotifications = document.querySelectorAll('.notification');
        oldNotifications.forEach(n => n.remove());

        const icons = {'success': '✅', 'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'};
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `${icons[type]} ${message}`;
        notification.style.cssText = `
            position: fixed; top: 80px; right: 20px; padding: 14px 24px;
            border-radius: 8px; font-weight: 600; font-size: 14px; z-index: 9999;
            animation: slideIn 0.3s ease; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            ${type === 'success' ? 'background: #00ff88; color: #000;' : ''}
            ${type === 'error' ? 'background: #ff0044; color: #fff;' : ''}
            ${type === 'warning' ? 'background: #ffcc00; color: #000;' : ''}
            ${type === 'info' ? 'background: #00d4ff; color: #000;' : ''}
        `;
        
        document.body.appendChild(notification);
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transition = 'opacity 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    async apiCall(endpoint, method = 'POST', data = null) {
        try {
            const options = {
                method: method,
                headers: { 'Content-Type': 'application/json' }
            };
            if (data) options.body = JSON.stringify(data);
            const response = await fetch(`/api${endpoint}`, options);
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'API call failed');
            return result;
        } catch (error) {
            console.error(`API error: ${endpoint}`, error);
            throw error;
        }
    }

    async startCamera() {
        try {
            const result = await this.apiCall('/camera/start', 'POST', {camera_id:1, width:640, height:480, fps:30});
            if (result.success) {
                if (this.elements.cameraStatus) { this.elements.cameraStatus.textContent = 'CAMERA: ACTIVE'; this.elements.cameraStatus.className = 'badge badge-success'; }
                this.showNotification('Camera started', 'success');
            }
        } catch (error) { this.showNotification('Failed to start camera', 'error'); }
    }

    async stopCamera() {
        try {
            await this.apiCall('/camera/stop', 'POST');
            if (this.elements.cameraStatus) { this.elements.cameraStatus.textContent = 'CAMERA: OFF'; this.elements.cameraStatus.className = 'badge badge-danger'; }
            this.videoElement.src = '';
            if (this.overlayElement) this.overlayElement.style.display = 'flex';
        } catch (error) { this.showNotification('Failed to stop camera', 'error'); }
    }

    async startDetection() {
        try {
            if (this.elements.detectionStatus) { this.elements.detectionStatus.textContent = 'DETECTION: LOADING...'; this.elements.detectionStatus.className = 'badge badge-warning'; }
            const result = await this.apiCall('/detection/start', 'POST', {
                model_path: 'weights/yolov8n_person.pt', confidence_threshold: 0.5, iou_threshold: 0.45, device: 'cpu'
            });
            if (result.success) {
                if (this.elements.detectionStatus) { this.elements.detectionStatus.textContent = 'DETECTION: ACTIVE'; this.elements.detectionStatus.className = 'badge badge-success'; }
                this.showNotification('Detection started', 'success');
            }
        } catch (error) { this.showNotification('Failed to start detection', 'error'); }
    }

    async stopDetection() {
        try {
            await this.apiCall('/detection/stop', 'POST');
            if (this.elements.detectionStatus) { this.elements.detectionStatus.textContent = 'DETECTION: OFF'; this.elements.detectionStatus.className = 'badge badge-warning'; }
        } catch (error) { this.showNotification('Failed to stop detection', 'error'); }
    }

    async selectTargetById(trackId) {
        try {
            const result = await this.apiCall('/target/select', 'POST', { track_id: trackId });
            if (result.success) {
                this.selectedTargetId = trackId;
                if (this.elements.statSelected) this.elements.statSelected.textContent = `ID: ${trackId}`;
                this.showNotification(`Target ID ${trackId} selected!`, 'success');
            }
        } catch (error) { this.showNotification('Failed to select target', 'error'); }
    }

    async selectTarget() {
        const trackId = prompt('Enter Target Track ID:');
        if (!trackId) return;
        const id = parseInt(trackId);
        if (isNaN(id) || id < 1) { this.showNotification('Invalid ID', 'warning'); return; }
        await this.selectTargetById(id);
    }

    async releaseTarget() {
        try {
            await this.apiCall('/target/release', 'POST');
            this.selectedTargetId = null;
            if (this.elements.statSelected) this.elements.statSelected.textContent = '---';
            this.showNotification('Target released', 'success');
        } catch (error) { this.showNotification('Failed to release target', 'error'); }
    }

    async startFollow() {
        try {
            if (!this.selectedTargetId) { this.showNotification('Select target first!', 'warning'); return; }
            await this.apiCall('/follow/start', 'POST');
            this.showNotification('Following started', 'success');
        } catch (error) { this.showNotification('Failed to start follow', 'error'); }
    }

    async stopFollow() {
        try {
            await this.apiCall('/follow/stop', 'POST');
            this.showNotification('Following stopped', 'info');
        } catch (error) { this.showNotification('Failed to stop follow', 'error'); }
    }

    async captureFrame() {
        try {
            await this.apiCall('/dataset/capture', 'POST');
            this.showNotification('Frame captured', 'success');
        } catch (error) { this.showNotification('Failed to capture', 'error'); }
    }

    async startRecording() {
        try {
            await this.apiCall('/dataset/record/start', 'POST');
            document.getElementById('btnRecordStart').disabled = true;
            document.getElementById('btnRecordStop').disabled = false;
        } catch (error) { this.showNotification('Failed to start recording', 'error'); }
    }

    async stopRecording() {
        try {
            await this.apiCall('/dataset/record/stop', 'POST');
            document.getElementById('btnRecordStart').disabled = false;
            document.getElementById('btnRecordStop').disabled = true;
        } catch (error) { this.showNotification('Failed to stop recording', 'error'); }
    }

    async exportDataset() {
        try {
            await this.apiCall('/dataset/export', 'POST');
            this.showNotification('Dataset exported', 'success');
        } catch (error) { this.showNotification('Failed to export', 'error'); }
    }

    async applyPID() {
        try {
            const payload = {
                steering: {
                    kp: parseFloat(this.elements.steerKp?.value) || 0.05,
                    ki: parseFloat(this.elements.steerKi?.value) || 0.001,
                    kd: parseFloat(this.elements.steerKd?.value) || 0.01
                },
                distance: {
                    kp: parseFloat(this.elements.distKp?.value) || 0.1,
                    ki: parseFloat(this.elements.distKi?.value) || 0.002,
                    kd: parseFloat(this.elements.distKd?.value) || 0.02
                },
                target_height: parseFloat(this.elements.targetHeight?.value) || 150
            };
            await this.apiCall('/robot/pid', 'POST', payload);
            this.showNotification('PID applied!', 'success');
        } catch (error) { this.showNotification('Failed to apply PID', 'error'); }
    }

    destroy() {
        if (this.fpsUpdateInterval) clearInterval(this.fpsUpdateInterval);
        if (this.ws) this.ws.close();
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new NexoraDashboard();
    window.addEventListener('beforeunload', () => {
        window.dashboard?.destroy();
    });
});