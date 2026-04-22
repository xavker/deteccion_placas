import easyocr
import cv2
import re
import numpy as np

class OCRReader:
    """
    Lector OCR para placas vehiculares ecuatorianas usando EasyOCR.
    Formato de placa ecuatoriana: ABC-1234 (3 letras + guión + 4 dígitos)
    """

    # Caracteres permitidos en placas
    ALLOWED_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'

    # Patrón regex para placas ecuatorianas
    # Formatos válidos: ABC-1234, ABC1234, AB-1234, AB1234
    PLATE_PATTERN = re.compile(r'^[A-Z]{2,3}-?\d{3,4}$')

    def __init__(self, gpu=False):
        """
        Inicializa el lector OCR.
        
        Args:
            gpu: Usar GPU para OCR. False para Intel Iris Xe (sin CUDA).
        """
        print("[OCRReader] Inicializando EasyOCR (puede tardar la primera vez)...")
        self.reader = easyocr.Reader(
            ['en'],  # Idioma inglés para caracteres alfanuméricos
            gpu=gpu
        )
        print("[OCRReader] EasyOCR listo.")

    def read_plate(self, plate_image):
        """
        Lee el texto de una imagen recortada de placa.
        
        Args:
            plate_image: Imagen BGR de OpenCV con la placa recortada.
        
        Returns:
            Tupla (texto, confianza) o (None, 0.0) si no se pudo leer.
        """
        if plate_image is None or plate_image.size == 0:
            return None, 0.0

        # Preprocesar la imagen para mejorar la lectura
        processed = self._preprocess(plate_image)

        # Ejecutar OCR
        try:
            results = self.reader.readtext(
                processed,
                allowlist=self.ALLOWED_CHARS,
                detail=1,
                paragraph=False
            )
        except Exception as e:
            print(f"[OCRReader] Error en OCR: {e}")
            return None, 0.0

        if not results:
            return None, 0.0

        # Combinar todos los textos detectados
        full_text = ''
        total_conf = 0.0
        count = 0

        for (bbox, text, conf) in results:
            if conf > 0.2:  # Filtrar resultados de baja confianza
                full_text += text
                total_conf += conf
                count += 1

        if count == 0:
            return None, 0.0

        avg_conf = total_conf / count

        # Limpiar el texto
        cleaned = self._clean_text(full_text)

        if cleaned and len(cleaned) >= 4:
            return cleaned, avg_conf

        return None, 0.0

    def _preprocess(self, image):
        """
        Preprocesa la imagen de la placa para mejorar la precisión del OCR.
        """
        # Redimensionar si es muy pequeña
        h, w = image.shape[:2]
        if w < 200:
            scale = 200 / w
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Filtro bilateral: reduce ruido manteniendo bordes nítidos
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)

        # Mejora de contraste con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Binarización adaptativa
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return binary

    def _clean_text(self, text):
        """
        Limpia y normaliza el texto leído de la placa.
        """
        # Convertir a mayúsculas
        text = text.upper().strip()

        # Remover caracteres no permitidos
        cleaned = ''.join(c for c in text if c in self.ALLOWED_CHARS)

        # Intentar formatear como placa ecuatoriana
        cleaned_no_dash = cleaned.replace('-', '')

        # Si tiene formato válido sin guión, agregar guión
        if len(cleaned_no_dash) >= 6:
            # Intentar separar letras de números
            letters = ''
            numbers = ''
            for c in cleaned_no_dash:
                if c.isalpha() and not numbers:
                    letters += c
                elif c.isdigit():
                    numbers += c

            if 2 <= len(letters) <= 3 and 3 <= len(numbers) <= 4:
                return f"{letters}-{numbers}"

        return cleaned

    def validate_plate(self, text):
        """
        Valida si el texto tiene formato de placa ecuatoriana.
        
        Returns:
            True si el formato es válido.
        """
        if text is None:
            return False
        return bool(self.PLATE_PATTERN.match(text))
