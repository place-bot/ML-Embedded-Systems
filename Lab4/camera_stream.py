"""One camera owner; retain the newest complete JPEG instead of queuing frames."""
from collections import deque
import io
import shutil
import subprocess
import threading
import time
from PIL import Image
from common import CONFIG


class Camera:
    def __init__(self, camera, log_path, image_path=None):
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.process = None
        self.log = None
        self.image_path = image_path
        if image_path is None:
            executable = shutil.which('rpicam-vid')
            if executable is None:
                raise RuntimeError('rpicam-vid was not found. Use the course Raspberry Pi OS.')
            self.log = open(log_path, 'wb')
            self.process = subprocess.Popen(
                [executable, '--camera', str(camera), '--nopreview', '--timeout', '0',
                 '--codec', 'mjpeg', '--quality', '80', '--framerate', str(CONFIG['camera_fps']),
                 '--width', str(CONFIG['camera_width']), '--height', str(CONFIG['camera_height']),
                 '--output', '-'], stdout=subprocess.PIPE, stderr=self.log)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _publish(self, jpeg, sequence):
        with Image.open(io.BytesIO(jpeg)) as source:
            image = source.convert('RGB')
        with self.condition:
            self.latest = (sequence, time.monotonic(), time.time(), image, jpeg)
            self.condition.notify_all()

    def _read(self):
        try:
            sequence = 0
            if self.image_path is not None:
                with Image.open(self.image_path) as source:
                    image = source.convert('RGB')
                    image.thumbnail((CONFIG['camera_width'], CONFIG['camera_height']))
                    buffer = io.BytesIO()
                    image.save(buffer, 'JPEG', quality=95)
                while not self.stop.is_set():
                    sequence += 1
                    self._publish(buffer.getvalue(), sequence)
                    self.stop.wait(1/CONFIG['camera_fps'])
                return
            buffer = bytearray()
            while not self.stop.is_set():
                chunk = self.process.stdout.read1(65536)
                if not chunk:
                    raise RuntimeError('Camera stream ended. See camera.log; stop other camera programs.')
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9', start+2) if start >= 0 else -1
                    if end < 0:
                        break
                    jpeg = bytes(buffer[start:end+2])
                    del buffer[:end+2]
                    sequence += 1
                    self._publish(jpeg, sequence)
                if len(buffer) > 4*1024*1024:
                    raise RuntimeError('Invalid MJPEG camera data. See camera.log.')
        except Exception as exc:
            if not self.stop.is_set():
                with self.condition:
                    self.error = str(exc)
                    self.condition.notify_all()

    def next(self, after):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error or self.stop.is_set() or (self.latest and self.latest[0] != after), 8)
            if self.error:
                raise RuntimeError(self.error)
            if self.stop.is_set():
                return None
            if not ready:
                raise RuntimeError('No camera frame for 8 seconds. Check the camera connection.')
            return self.latest

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            self.process.stdout.close()
        self.thread.join(timeout=3)
        if self.log:
            self.log.close()
