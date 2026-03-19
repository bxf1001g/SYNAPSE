"""SYNAPSE Hardware Abstraction Layer for Jetson / Raspberry Pi.

Provides camera capture, servo/motor control, and sensor reading.
Falls back gracefully when hardware libraries aren't available.

Usage:
    from hardware import HardwareController
    hw = HardwareController()
    hw.capture_image("photo.jpg")
    hw.move_servo(channel=0, angle=90)
    hw.read_sensor("distance")
"""

import base64
import os
import time

# ── Optional Hardware Imports (graceful fallback) ───────────────

_cv2_available = False
try:
    import cv2
    _cv2_available = True
except ImportError:
    cv2 = None

_gpio_available = False
_gpio = None
try:
    import Jetson.GPIO as GPIO
    _gpio = GPIO
    _gpio_available = True
except ImportError:
    try:
        import RPi.GPIO as GPIO
        _gpio = GPIO
        _gpio_available = True
    except ImportError:
        pass

_servo_kit_available = False
try:
    from adafruit_servokit import ServoKit
    _servo_kit_available = True
except ImportError:
    ServoKit = None

_i2c_available = False
try:
    import board
    import busio
    _i2c_available = True
except ImportError:
    board = None
    busio = None


class HardwareController:
    """Unified hardware interface for SYNAPSE AI agents."""

    def __init__(self, workspace="./workspace"):
        self.workspace = workspace
        self._camera = None
        self._camera_index = 0
        self._servo_kit = None
        self._gpio_pins = {}
        self._gpio_initialized = False
        self._i2c = None

    # ── Status / Discovery ──────────────────────────────────────

    def status(self):
        """Return hardware availability status."""
        cam_ok = False
        if _cv2_available:
            try:
                cap = cv2.VideoCapture(self._camera_index)
                cam_ok = cap.isOpened()
                cap.release()
            except Exception:
                pass

        return {
            "camera": {
                "available": _cv2_available,
                "connected": cam_ok,
                "backend": "OpenCV" if _cv2_available else "not installed",
            },
            "gpio": {
                "available": _gpio_available,
                "platform": (
                    "Jetson" if "Jetson" in str(type(_gpio))
                    else "RPi" if _gpio_available
                    else "not available"
                ),
            },
            "servo": {
                "available": _servo_kit_available,
                "backend": "PCA9685 (adafruit-servokit)" if _servo_kit_available else "not installed",
            },
            "i2c": {
                "available": _i2c_available,
            },
            "install_hints": self._install_hints(),
        }

    def _install_hints(self):
        """Return pip install hints for missing hardware libraries."""
        hints = []
        if not _cv2_available:
            hints.append("pip install opencv-python")
        if not _gpio_available:
            hints.append("pip install Jetson.GPIO  # or RPi.GPIO for Raspberry Pi")
        if not _servo_kit_available:
            hints.append("pip install adafruit-servokit")
        if not _i2c_available:
            hints.append("pip install adafruit-blinka")
        return hints

    # ── Camera ──────────────────────────────────────────────────

    def capture_image(self, filename="capture.jpg", camera_index=None):
        """Capture a single image from camera. Returns filepath and base64 data."""
        if not _cv2_available:
            return {
                "success": False,
                "error": "OpenCV not installed. Run: pip install opencv-python",
            }

        idx = camera_index if camera_index is not None else self._camera_index
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            # Try CSI camera on Jetson via GStreamer pipeline
            gst_pipeline = (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink"
            )
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            return {"success": False, "error": f"Cannot open camera {idx}"}

        try:
            # Warm up camera (first frames can be dark)
            for _ in range(5):
                cap.read()

            ret, frame = cap.read()
            if not ret:
                return {"success": False, "error": "Failed to capture frame"}

            filepath = os.path.join(self.workspace, filename)
            os.makedirs(os.path.dirname(filepath) or self.workspace, exist_ok=True)
            cv2.imwrite(filepath, frame)

            # Base64 encode for AI vision
            _, buffer = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buffer).decode("utf-8")

            h, w = frame.shape[:2]
            return {
                "success": True,
                "filepath": filepath,
                "base64": b64,
                "width": w,
                "height": h,
            }
        finally:
            cap.release()

    def capture_video(self, filename="capture.mp4", duration=5, fps=30):
        """Capture a short video clip. Returns filepath."""
        if not _cv2_available:
            return {"success": False, "error": "OpenCV not installed"}

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            return {"success": False, "error": "Cannot open camera"}

        filepath = os.path.join(self.workspace, filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

        try:
            frames = int(duration * fps)
            for _ in range(frames):
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
            return {"success": True, "filepath": filepath, "duration": duration}
        finally:
            cap.release()
            out.release()

    # ── Servo / Motor Control ───────────────────────────────────

    def _init_servo_kit(self, channels=16):
        """Initialize PCA9685 servo controller."""
        if self._servo_kit:
            return self._servo_kit
        if not _servo_kit_available:
            return None
        try:
            self._servo_kit = ServoKit(channels=channels)
            return self._servo_kit
        except Exception as e:
            print(f"[HARDWARE] ServoKit init error: {e}")
            return None

    def move_servo(self, channel=0, angle=90):
        """Move a servo to a specific angle (0-180 degrees)."""
        angle = max(0, min(180, int(angle)))
        kit = self._init_servo_kit()
        if kit:
            try:
                kit.servo[channel].angle = angle
                return {
                    "success": True,
                    "channel": channel,
                    "angle": angle,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Fallback: try direct GPIO PWM
        if _gpio_available:
            return self._servo_gpio_fallback(channel, angle)

        return {
            "success": False,
            "error": "No servo library. Install: pip install adafruit-servokit",
        }

    def _servo_gpio_fallback(self, pin, angle):
        """Direct GPIO PWM servo control (fallback without PCA9685)."""
        try:
            if not self._gpio_initialized:
                _gpio.setmode(_gpio.BOARD)
                self._gpio_initialized = True

            if pin not in self._gpio_pins:
                _gpio.setup(pin, _gpio.OUT)
                self._gpio_pins[pin] = _gpio.PWM(pin, 50)  # 50Hz for servos
                self._gpio_pins[pin].start(0)

            # Convert angle (0-180) to duty cycle (2.5-12.5%)
            duty = 2.5 + (angle / 180.0) * 10.0
            self._gpio_pins[pin].ChangeDutyCycle(duty)
            time.sleep(0.5)
            self._gpio_pins[pin].ChangeDutyCycle(0)  # Stop jitter

            return {"success": True, "channel": pin, "angle": angle, "method": "gpio_pwm"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_motor_speed(self, channel=0, speed=0.0):
        """Set continuous rotation servo / motor speed (-1.0 to 1.0)."""
        kit = self._init_servo_kit()
        if kit:
            try:
                speed = max(-1.0, min(1.0, float(speed)))
                kit.continuous_servo[channel].throttle = speed
                return {"success": True, "channel": channel, "speed": speed}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "ServoKit not available"}

    def stop_all_motors(self):
        """Emergency stop — all servos/motors to neutral."""
        results = []
        kit = self._init_servo_kit()
        if kit:
            for i in range(16):
                try:
                    kit.continuous_servo[i].throttle = 0
                except Exception:
                    pass
            results.append("ServoKit: all channels stopped")

        for pin, pwm in self._gpio_pins.items():
            try:
                pwm.ChangeDutyCycle(0)
            except Exception:
                pass
            results.append(f"GPIO pin {pin}: stopped")

        return {"success": True, "stopped": results}

    # ── GPIO Digital I/O ────────────────────────────────────────

    def gpio_write(self, pin, value):
        """Write HIGH (1) or LOW (0) to a GPIO pin."""
        if not _gpio_available:
            return {"success": False, "error": "GPIO not available"}
        try:
            if not self._gpio_initialized:
                _gpio.setmode(_gpio.BOARD)
                self._gpio_initialized = True
            _gpio.setup(pin, _gpio.OUT)
            _gpio.output(pin, _gpio.HIGH if value else _gpio.LOW)
            return {"success": True, "pin": pin, "value": value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def gpio_read(self, pin):
        """Read digital value from a GPIO pin."""
        if not _gpio_available:
            return {"success": False, "error": "GPIO not available"}
        try:
            if not self._gpio_initialized:
                _gpio.setmode(_gpio.BOARD)
                self._gpio_initialized = True
            _gpio.setup(pin, _gpio.IN)
            value = _gpio.input(pin)
            return {"success": True, "pin": pin, "value": value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Sensors ─────────────────────────────────────────────────

    def read_sensor(self, sensor_type="distance", pin=None, **kwargs):
        """Read from various sensors. Returns sensor data dict."""
        handlers = {
            "distance": self._read_ultrasonic,
            "temperature": self._read_temperature,
            "imu": self._read_imu,
            "light": self._read_light,
        }
        handler = handlers.get(sensor_type)
        if handler:
            return handler(pin=pin, **kwargs)
        return {"success": False, "error": f"Unknown sensor type: {sensor_type}"}

    def _read_ultrasonic(self, pin=None, trigger_pin=None, echo_pin=None, **kwargs):
        """Read HC-SR04 ultrasonic distance sensor."""
        trig = trigger_pin or pin or 18
        echo = echo_pin or (pin + 1 if pin else 24)

        if not _gpio_available:
            return {"success": False, "error": "GPIO not available"}

        try:
            if not self._gpio_initialized:
                _gpio.setmode(_gpio.BOARD)
                self._gpio_initialized = True

            _gpio.setup(trig, _gpio.OUT)
            _gpio.setup(echo, _gpio.IN)

            # Send trigger pulse
            _gpio.output(trig, False)
            time.sleep(0.01)
            _gpio.output(trig, True)
            time.sleep(0.00001)
            _gpio.output(trig, False)

            # Measure echo
            start = time.time()
            timeout = start + 0.1
            while _gpio.input(echo) == 0 and time.time() < timeout:
                start = time.time()

            end = time.time()
            while _gpio.input(echo) == 1 and time.time() < timeout:
                end = time.time()

            distance_cm = (end - start) * 34300 / 2

            return {
                "success": True,
                "sensor": "ultrasonic",
                "distance_cm": round(distance_cm, 2),
                "trigger_pin": trig,
                "echo_pin": echo,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _read_temperature(self, pin=None, **kwargs):
        """Read temperature — tries I2C sensor, falls back to system thermal."""
        # Try system thermal zone (works on Jetson without extra hardware)
        try:
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal_path):
                with open(thermal_path) as f:
                    temp_mc = int(f.read().strip())
                return {
                    "success": True,
                    "sensor": "system_thermal",
                    "temperature_c": round(temp_mc / 1000.0, 1),
                    "source": thermal_path,
                }
        except Exception:
            pass

        return {"success": False, "error": "No temperature sensor found"}

    def _read_imu(self, pin=None, **kwargs):
        """Read IMU (accelerometer/gyro) via I2C — MPU6050 or similar."""
        if not _i2c_available:
            return {"success": False, "error": "I2C not available. Install: pip install adafruit-blinka"}

        try:
            import adafruit_mpu6050
            i2c = busio.I2C(board.SCL, board.SDA)
            mpu = adafruit_mpu6050.MPU6050(i2c)
            accel = mpu.acceleration
            gyro = mpu.gyro
            temp = mpu.temperature
            return {
                "success": True,
                "sensor": "mpu6050",
                "acceleration": {"x": round(accel[0], 3), "y": round(accel[1], 3), "z": round(accel[2], 3)},
                "gyro": {"x": round(gyro[0], 3), "y": round(gyro[1], 3), "z": round(gyro[2], 3)},
                "temperature_c": round(temp, 1),
            }
        except ImportError:
            return {"success": False, "error": "Install: pip install adafruit-mpu6050"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _read_light(self, pin=None, **kwargs):
        """Read ambient light sensor via I2C — BH1750 or similar."""
        if not _i2c_available:
            return {"success": False, "error": "I2C not available"}

        try:
            import adafruit_bh1750
            i2c = busio.I2C(board.SCL, board.SDA)
            sensor = adafruit_bh1750.BH1750(i2c)
            return {
                "success": True,
                "sensor": "bh1750",
                "light_lux": round(sensor.lux, 1),
            }
        except ImportError:
            return {"success": False, "error": "Install: pip install adafruit-bh1750"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Autonomy Loop ───────────────────────────────────────────

    def observe(self, filename="observation.jpg"):
        """Capture image + all available sensor data. Returns observation dict."""
        observation = {"timestamp": time.time(), "sensors": {}}

        # Camera
        img = self.capture_image(filename)
        observation["image"] = img

        # Temperature (always available on Jetson)
        observation["sensors"]["temperature"] = self._read_temperature()

        # Distance (if ultrasonic connected)
        dist = self._read_ultrasonic()
        if dist.get("success"):
            observation["sensors"]["distance"] = dist

        # IMU (if connected)
        imu = self._read_imu()
        if imu.get("success"):
            observation["sensors"]["imu"] = imu

        return observation

    def describe_observation(self, observation):
        """Convert observation dict to natural language for AI prompt."""
        parts = []

        img = observation.get("image", {})
        if img.get("success"):
            parts.append(f"Camera captured {img['width']}x{img['height']} image saved to {img['filepath']}")
        else:
            parts.append(f"Camera: {img.get('error', 'unavailable')}")

        sensors = observation.get("sensors", {})
        for name, data in sensors.items():
            if data.get("success"):
                if name == "distance":
                    parts.append(f"Distance sensor: {data['distance_cm']} cm")
                elif name == "temperature":
                    parts.append(f"Temperature: {data['temperature_c']}°C")
                elif name == "imu":
                    a = data["acceleration"]
                    parts.append(f"IMU accel: x={a['x']}, y={a['y']}, z={a['z']} m/s²")

        return "\n".join(parts)

    # ── Cleanup ─────────────────────────────────────────────────

    def cleanup(self):
        """Release all hardware resources."""
        if self._camera:
            self._camera.release()
            self._camera = None

        self.stop_all_motors()

        if _gpio_available and self._gpio_initialized:
            try:
                _gpio.cleanup()
            except Exception:
                pass
            self._gpio_initialized = False

    def __del__(self):
        self.cleanup()
