from ultralytics import YOLO
import cv2
import os

class PlateDetector:
    """
    Detector de placas vehiculares usando YOLOv8.
    Localiza placas en un frame y retorna las regiones detectadas.
    """

    def __init__(self, model_path="models/plate_detector.pt", confidence=0.4):
        """
        Inicializa el detector de placas.
        
        Args:
            model_path: Ruta al modelo YOLOv8 entrenado para placas.
                        Si no existe, usa el modelo base yolov8n.pt.
            confidence: Umbral mínimo de confianza para aceptar detecciones.
        """
        self.confidence = confidence
        
        if os.path.exists(model_path):
            print(f"[PlateDetector] Cargando modelo personalizado: {model_path}")
            self.model = YOLO(model_path)
            self.use_custom_model = True
        else:
            print("[PlateDetector] Modelo personalizado no encontrado.")
            print("[PlateDetector] Cargando modelo base YOLOv8n (detecta vehículos)...")
            self.model = YOLO("yolov8n.pt")
            self.use_custom_model = False
            # Clases COCO para vehículos: 2=car, 3=motorcycle, 5=bus, 7=truck
            self.vehicle_classes = [2, 3, 5, 7]

        print("[PlateDetector] Modelo cargado exitosamente.")

    def detect(self, frame):
        """
        Detecta placas en un frame.
        
        Args:
            frame: Frame BGR de OpenCV.
        
        Returns:
            Lista de diccionarios con:
                - 'bbox': (x1, y1, x2, y2) coordenadas del bounding box
                - 'confidence': confianza de la detección
                - 'cropped': imagen recortada de la placa/vehículo
        """
        detections = []

        if self.use_custom_model:
            detections = self._detect_plates_custom(frame)
        else:
            detections = self._detect_plates_fallback(frame)

        return detections

    def _detect_plates_custom(self, frame):
        """Detección directa de placas con modelo entrenado."""
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    # Recortar la placa del frame
                    cropped = frame[y1:y2, x1:x2].copy()

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'cropped': cropped
                    })

        return detections

    def _detect_plates_fallback(self, frame):
        """
        Fallback: Detecta vehículos con COCO model, 
        luego busca regiones rectangulares que parezcan placas usando OpenCV.
        """
        results = self.model(frame, conf=self.confidence, classes=self.vehicle_classes, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())

                    # Buscar placas dentro de la región del vehículo
                    vehicle_roi = frame[vy1:vy2, vx1:vx2]
                    plate_regions = self._find_plate_contours(vehicle_roi)

                    for (px1, py1, px2, py2) in plate_regions:
                        # Convertir coordenadas relativas al frame completo
                        abs_x1 = vx1 + px1
                        abs_y1 = vy1 + py1
                        abs_x2 = vx1 + px2
                        abs_y2 = vy1 + py2

                        cropped = frame[abs_y1:abs_y2, abs_x1:abs_x2].copy()

                        detections.append({
                            'bbox': (abs_x1, abs_y1, abs_x2, abs_y2),
                            'confidence': float(box.conf[0]),
                            'cropped': cropped
                        })

        return detections

    def _find_plate_contours(self, vehicle_roi):
        """
        Busca contornos rectangulares que parezcan placas dentro de la ROI del vehículo.
        Las placas ecuatorianas tienen una relación de aspecto aprox. 2:1 a 4:1.
        """
        if vehicle_roi.size == 0:
            return []

        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        
        # Filtro bilateral para reducir ruido preservando bordes
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Detección de bordes
        edges = cv2.Canny(blur, 30, 200)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_regions = []
        h_roi, w_roi = vehicle_roi.shape[:2]

        # Ordenar contornos por área (más grande primero)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        for contour in contours:
            # Aproximar el contorno a un polígono
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Una placa tiene 4 esquinas
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                min_area = h_roi * w_roi * 0.005  # Al menos 0.5% del área del vehículo
                max_area = h_roi * w_roi * 0.25   # Máximo 25% del área del vehículo

                # Filtrar por relación de aspecto y tamaño
                if 1.5 <= aspect_ratio <= 5.0 and min_area < area < max_area:
                    plate_regions.append((x, y, x + w, y + h))

        return plate_regions

    def set_confidence(self, confidence):
        """Actualiza el umbral de confianza."""
        self.confidence = max(0.1, min(1.0, confidence))
