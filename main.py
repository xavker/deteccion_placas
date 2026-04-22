import sys
import os

def main():
    print("=" * 50)
    print("  DETECTOR DE PLACAS VEHICULARES - XAVKER")
    print("  YOLOv8 + EasyOCR")
    print("=" * 50)
    print()

    # Inicializar el detector de placas (YOLOv8)
    print("[1/3] Cargando detector de placas (YOLOv8)...")
    from plate_detector import PlateDetector
    plate_detector = PlateDetector(
        model_path="models/plate_detector.pt",
        confidence=0.4
    )

    # Inicializar el lector OCR (EasyOCR)
    print("[2/3] Cargando lector OCR (EasyOCR)...")
    from ocr_reader import OCRReader
    ocr_reader = OCRReader(gpu=False)  # Intel Iris Xe → CPU mode

    # Inicializar el procesador de visión con los módulos
    print("[3/3] Inicializando procesador de visión...")
    from vision_processor import VisionProcessor
    vision_processor = VisionProcessor(
        plate_detector=plate_detector,
        ocr_reader=ocr_reader
    )

    # Crear directorio de modelos si no existe
    os.makedirs("models", exist_ok=True)

    # Inicializar la interfaz gráfica
    print()
    print("[OK] Sistema listo. Iniciando interfaz gráfica...")
    print()

    from gui_manager import App
    app = App(vision_processor)
    
    # Manejar el cierre de la ventana correctamente
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Iniciar el bucle principal de la interfaz
    app.mainloop()

if __name__ == "__main__":
    main()
