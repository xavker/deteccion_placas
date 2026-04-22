import cv2
import threading
import time
from datetime import datetime

class VisionProcessor:
    """
    Procesador de visión que captura video y ejecuta el pipeline de detección.
    La captura y el procesamiento se ejecutan en un hilo separado.
    """

    def __init__(self, plate_detector=None, ocr_reader=None):
        self.cap = None
        self.running = False
        self.frame = None          # Frame original con anotaciones
        self.raw_frame = None      # Frame sin anotaciones
        self.thread = None
        self.lock = threading.Lock()
        
        # Módulos de detección
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader
        
        # Estado de detección
        self.auto_mode = False
        self.frame_count = 0
        self.detect_every_n = 5    # Procesar detección cada N frames (optimización CPU)
        
        # Resultados de detección
        self.latest_detections = []       # Detecciones actuales
        self.latest_plate_text = None     # Texto de la última placa
        self.latest_plate_conf = 0.0      # Confianza de la última lectura
        self.latest_plate_image = None    # Imagen recortada de la última placa
        
        # Historial de placas
        self.plate_history = []    # Lista de tuplas (texto, confianza, timestamp)
        self.max_history = 50      # Máximo de entradas en el historial
        
        # Lock para resultados de detección
        self.detection_lock = threading.Lock()

    def start(self, camera_index=0):
        if not self.running:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print("Error: No se pudo abrir la cámara.")
                return False
            
            # Configurar resolución de cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            return True
        return False

    def stop(self):
        self.running = False
        self.auto_mode = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None

    def set_auto_mode(self, enabled):
        """Activa/desactiva el modo de detección automática."""
        self.auto_mode = enabled
        if enabled:
            print("[VisionProcessor] Modo AUTO activado - detección continua")
        else:
            print("[VisionProcessor] Modo AUTO desactivado")

    def detect_single(self):
        """Ejecuta una detección única en el frame actual."""
        with self.lock:
            if self.raw_frame is not None:
                self._run_detection(self.raw_frame.copy())

    def _update(self):
        """Loop principal de captura y procesamiento (ejecuta en hilo separado)."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            self.frame_count += 1
            annotated_frame = frame.copy()

            # Ejecutar detección si el modo auto está activo
            if self.auto_mode and self.plate_detector:
                if self.frame_count % self.detect_every_n == 0:
                    self._run_detection(frame)

            # Dibujar las detecciones más recientes sobre el frame
            with self.detection_lock:
                for det in self.latest_detections:
                    self._draw_detection(annotated_frame, det)

            with self.lock:
                self.raw_frame = frame
                self.frame = annotated_frame

    def _run_detection(self, frame):
        """Ejecuta el pipeline completo de detección + OCR."""
        if self.plate_detector is None:
            return

        detections = self.plate_detector.detect(frame)

        for det in detections:
            # Ejecutar OCR en cada placa detectada
            if self.ocr_reader and det['cropped'] is not None and det['cropped'].size > 0:
                text, conf = self.ocr_reader.read_plate(det['cropped'])
                det['text'] = text
                det['text_conf'] = conf
            else:
                det['text'] = None
                det['text_conf'] = 0.0

        with self.detection_lock:
            self.latest_detections = detections

            # Actualizar última placa leída
            for det in detections:
                if det['text'] and det['text_conf'] > 0.3:
                    self.latest_plate_text = det['text']
                    self.latest_plate_conf = det['text_conf']
                    self.latest_plate_image = det['cropped']

                    # Agregar al historial (evitar duplicados consecutivos)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if not self.plate_history or self.plate_history[-1][0] != det['text']:
                        self.plate_history.append((det['text'], det['text_conf'], timestamp))
                        if len(self.plate_history) > self.max_history:
                            self.plate_history.pop(0)
                        print(f"[Placa Detectada] {det['text']} (conf: {det['text_conf']:.2f}) a las {timestamp}")

    def _draw_detection(self, frame, detection):
        """Dibuja bounding box y texto de una detección sobre el frame."""
        x1, y1, x2, y2 = detection['bbox']
        text = detection.get('text', None)
        conf = detection.get('confidence', 0)

        # Color del bounding box
        if text:
            color = (0, 255, 0)   # Verde si se leyó la placa
            thickness = 3
        else:
            color = (0, 255, 255) # Amarillo si solo se detectó
            thickness = 2

        # Dibujar rectángulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Dibujar texto
        label = text if text else f"Placa ({conf:.0%})"
        
        # Fondo del texto
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    def get_latest_frame(self):
        """Retorna el último frame procesado en formato RGB."""
        with self.lock:
            if self.frame is not None:
                rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                return rgb_frame
        return None

    def get_detection_results(self):
        """Retorna los resultados de la última detección."""
        with self.detection_lock:
            return {
                'plate_text': self.latest_plate_text,
                'plate_conf': self.latest_plate_conf,
                'plate_image': self.latest_plate_image,
                'detections_count': len(self.latest_detections),
                'history': list(self.plate_history)
            }

    def set_confidence(self, value):
        """Actualiza el umbral de confianza del detector."""
        if self.plate_detector:
            self.plate_detector.set_confidence(value)
