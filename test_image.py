"""
Script de prueba: Detectar placas en una imagen estática.
Uso:
    python test_image.py ruta/a/imagen.jpg
    python test_image.py  (sin argumentos → usa imagen de prueba de internet)
"""
import sys
import os
import cv2
import urllib.request

def download_test_image():
    """Descarga una imagen de prueba con un auto ecuatoriano."""
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, "test_car.jpg")
    
    if not os.path.exists(test_path):
        print("[*] Descargando imagen de prueba...")
        # Imagen de ejemplo de un vehículo (Unsplash - libre de uso)
        url = "https://images.unsplash.com/photo-1549317661-bd32c8ce0afa?w=800"
        try:
            urllib.request.urlretrieve(url, test_path)
            print(f"[OK] Imagen guardada en: {test_path}")
        except Exception as e:
            print(f"[ERROR] No se pudo descargar: {e}")
            print("[TIP] Coloca manualmente una imagen en test_images/test_car.jpg")
            return None
    return test_path


def main():
    # Determinar la imagen a usar
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("=" * 50)
        print("  TEST DE DETECCIÓN DE PLACAS EN IMAGEN")
        print("=" * 50)
        print()
        print("Uso: python test_image.py <ruta_imagen>")
        print()
        image_path = download_test_image()
        if image_path is None:
            return

    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        print(f"[ERROR] Imagen no encontrada: {image_path}")
        return

    # Cargar imagen
    print(f"\n[1/4] Cargando imagen: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("[ERROR] No se pudo leer la imagen. Verifica el formato.")
        return
    print(f"       Tamaño: {frame.shape[1]}x{frame.shape[0]} px")

    # Inicializar detector
    print("[2/4] Cargando YOLOv8...")
    from plate_detector import PlateDetector
    detector = PlateDetector(model_path="models/plate_detector.pt", confidence=0.3)

    # Inicializar OCR
    print("[3/4] Cargando EasyOCR...")
    from ocr_reader import OCRReader
    ocr = OCRReader(gpu=False)

    # Ejecutar detección
    print("[4/4] Ejecutando detección...")
    print()
    
    detections = detector.detect(frame)
    annotated = frame.copy()

    if not detections:
        print("⚠️  No se detectaron placas en la imagen.")
        print("    Posibles razones:")
        print("    - No hay vehículos visibles")
        print("    - La placa está muy lejos o borrosa")
        print("    - El umbral de confianza es muy alto (actual: 30%)")
        print("    - Se necesita un modelo entrenado para placas (models/plate_detector.pt)")
    else:
        print(f"✅ Se detectaron {len(detections)} región(es) de placa(s):")
        print("-" * 50)

        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']

            # Ejecutar OCR
            text, text_conf = ocr.read_plate(det['cropped'])

            print(f"\n  Detección #{i}:")
            print(f"    Posición: ({x1}, {y1}) → ({x2}, {y2})")
            print(f"    Confianza detección: {conf:.1%}")

            if text:
                print(f"    📋 Texto leído: {text}")
                print(f"    Confianza OCR: {text_conf:.1%}")
                valid = ocr.validate_plate(text)
                print(f"    Formato EC válido: {'✅ Sí' if valid else '❌ No'}")
                color = (0, 255, 0)
                label = text
            else:
                print(f"    📋 Texto: (no se pudo leer)")
                color = (0, 255, 255)
                label = f"Placa ({conf:.0%})"

            # Dibujar en la imagen
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Guardar recorte de la placa
            crop_path = f"test_images/placa_{i}.jpg"
            cv2.imwrite(crop_path, det['cropped'])
            print(f"    Recorte guardado: {crop_path}")

    # Guardar imagen anotada
    output_path = "test_images/resultado.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\n{'=' * 50}")
    print(f"Imagen con resultados guardada en: {output_path}")

    # Mostrar imagen
    print("\nPresiona cualquier tecla en la ventana para cerrar...")
    
    # Redimensionar si es muy grande
    h, w = annotated.shape[:2]
    if w > 1200:
        scale = 1200 / w
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))
    
    cv2.imshow("Resultado - Deteccion de Placas", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
