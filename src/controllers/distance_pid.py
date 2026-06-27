import time

class DistancePID:
    def __init__(self, kp: float = 0.1, ki: float = 0.002, kd: float = 0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.perf_counter()

    def update(self, error: float) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        if dt <= 0.0:
            dt = 1e-6

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        self.last_time = now
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.perf_counter()