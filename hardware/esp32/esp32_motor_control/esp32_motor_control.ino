/**
 * NEXORA - ESP32 Motor Control Firmware
 * 
 * This firmware receives motor commands via serial and controls DC motors
 * using PWM signals.
 */

#include <WiFi.h>
#include <ESP32PWM.h>
#include <ESP32Servo.h>

// ===================== CONFIGURATION =====================
// Motor pins
#define MOTOR_LEFT_PWM  12
#define MOTOR_LEFT_IN1  14
#define MOTOR_LEFT_IN2  27
#define MOTOR_RIGHT_PWM 13
#define MOTOR_RIGHT_IN1 26
#define MOTOR_RIGHT_IN2 25

// PWM configuration
#define PWM_FREQUENCY    5000
#define PWM_RESOLUTION   8
#define MAX_PWM         255

// Serial configuration
#define SERIAL_BAUD     115200

// Motor control constants
#define DEAD_ZONE       10  // Dead zone for motor speeds
#define MIN_SPEED       50  // Minimum speed to overcome inertia

// ===================== GLOBAL VARIABLES =====================
int left_speed = 0;
int right_speed = 0;

// ===================== SETUP =====================
void setup() {
  // Initialize serial communication
  Serial.begin(SERIAL_BAUD);
  
  // Initialize motor pins
  pinMode(MOTOR_LEFT_PWM, OUTPUT);
  pinMode(MOTOR_LEFT_IN1, OUTPUT);
  pinMode(MOTOR_LEFT_IN2, OUTPUT);
  pinMode(MOTOR_RIGHT_PWM, OUTPUT);
  pinMode(MOTOR_RIGHT_IN1, OUTPUT);
  pinMode(MOTOR_RIGHT_IN2, OUTPUT);
  
  // Configure PWM
  ledcSetup(0, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcSetup(1, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_LEFT_PWM, 0);
  ledcAttachPin(MOTOR_RIGHT_PWM, 1);
  
  // Initial state - stop motors
  stopMotors();
  
  Serial.println("NEXORA ESP32 Motor Controller Ready");
  Serial.println("Commands: [speed_left] [speed_right]");
  Serial.println("Speed range: -255 to 255");
}

// ===================== MAIN LOOP =====================
void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    processCommand(command);
  }
  
  // Update motors with current speeds
  updateMotors();
  
  // Small delay to prevent overwhelming the serial buffer
  delay(10);
}

// ===================== COMMAND PROCESSING =====================
void processCommand(String command) {
  command.trim();
  
  // Skip empty commands
  if (command.length() == 0) return;
  
  // Parse command
  int delimiterIndex = command.indexOf(' ');
  if (delimiterIndex == -1) {
    // Single command
    if (command == "stop") {
      stopMotors();
      Serial.println("Motors stopped");
    }
    else if (command == "status") {
      sendStatus();
    }
    else if (command == "reset") {
      resetMotors();
      Serial.println("Motors reset");
    }
    else {
      Serial.println("Unknown command: " + command);
    }
  }
  else {
    // Two values: left_speed right_speed
    String leftStr = command.substring(0, delimiterIndex);
    String rightStr = command.substring(delimiterIndex + 1);
    
    left_speed = leftStr.toInt();
    right_speed = rightStr.toInt();
    
    // Constrain speeds
    left_speed = constrain(left_speed, -255, 255);
    right_speed = constrain(right_speed, -255, 255);
    
    Serial.print("Speed set: L=");
    Serial.print(left_speed);
    Serial.print(", R=");
    Serial.println(right_speed);
  }
}

// ===================== MOTOR CONTROL =====================
void updateMotors() {
  // Apply dead zone
  int left_pwm = applyDeadZone(left_speed);
  int right_pwm = applyDeadZone(right_speed);
  
  // Update left motor
  setMotor(MOTOR_LEFT_IN1, MOTOR_LEFT_IN2, 0, left_pwm);
  
  // Update right motor
  setMotor(MOTOR_RIGHT_IN1, MOTOR_RIGHT_IN2, 1, right_pwm);
}

void setMotor(int in1, int in2, int pwm_channel, int speed) {
  if (speed > 0) {
    // Forward
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    ledcWrite(pwm_channel, speed);
  } 
  else if (speed < 0) {
    // Reverse
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    ledcWrite(pwm_channel, -speed);
  } 
  else {
    // Stop
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    ledcWrite(pwm_channel, 0);
  }
}

int applyDeadZone(int speed) {
  if (abs(speed) < DEAD_ZONE) {
    return 0;
  }
  
  // Scale speed to overcome dead zone
  int scaled_speed = speed;
  if (speed > 0) {
    scaled_speed = map(speed, DEAD_ZONE, 255, MIN_SPEED, 255);
  } else {
    scaled_speed = map(speed, -DEAD_ZONE, -255, -MIN_SPEED, -255);
  }
  
  return constrain(scaled_speed, -255, 255);
}

void stopMotors() {
  left_speed = 0;
  right_speed = 0;
  updateMotors();
}

void resetMotors() {
  // Reset to default state
  stopMotors();
  left_speed = 0;
  right_speed = 0;
}

// ===================== STATUS REPORTING =====================
void sendStatus() {
  Serial.print("STATUS: L=");
  Serial.print(left_speed);
  Serial.print(", R=");
  Serial.print(right_speed);
  Serial.print(", PWM_L=");
  Serial.print(applyDeadZone(left_speed));
  Serial.print(", PWM_R=");
  Serial.println(applyDeadZone(right_speed));
}

// ===================== ERROR HANDLING =====================
void handleError(String error) {
  Serial.print("ERROR: ");
  Serial.println(error);
  stopMotors();
}