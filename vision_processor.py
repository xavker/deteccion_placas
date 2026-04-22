import cv2
import threading
from PIL import Image, ImageTk

class VisionProcessor:
    def __init__(self):
        self.cap = None
        self.running = False
        self.frame = None
        self.thread = None
        self.lock = threading.Lock()

    def start(self, camera_index=0):
        if not self.running:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print("Error: No se pudo abrir la cámara.")
                return False
            
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            return True
        return False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
            self.cap = None

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                self.running = False
                break

    def get_latest_frame(self):
        with self.lock:
            if self.frame is not None:
                # Convertir de BGR a RGB
                rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                return rgb_frame
        return None
