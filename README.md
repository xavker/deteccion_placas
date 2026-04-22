# Detector de Placas Vehiculares Ecuatorianas (XAVKER)

Sistema de detección y lectura de placas vehiculares en tiempo real, desarrollado en Python. Utiliza **YOLOv8** para la localización de placas y **EasyOCR** para el reconocimiento de caracteres, con una interfaz gráfica moderna construida con CustomTkinter.

## Características

*   **Detección en Tiempo Real**: Stream de video con detección automática de placas vehiculares.
*   **Pipeline de 2 Etapas**: YOLOv8 (localización) + EasyOCR (lectura de caracteres).
*   **Placas Ecuatorianas**: Validación del formato estándar (ABC-1234).
*   **Interfaz Moderna**: Desarrollada con `CustomTkinter`, modo oscuro nativo.
*   **Procesamiento Asíncrono**: Captura de video en hilo separado, sin lag en la UI.
*   **Panel de Resultados**:
    *   Última placa detectada en texto grande.
    *   Imagen recortada de la placa.
    *   Indicador de confianza.
    *   Slider para ajustar umbral de detección.
*   **Historial de Placas**: Registro scrollable con timestamp de cada placa leída.
*   **Indicadores LED**: LEDs visuales que cambian de color según el estado del sistema.
*   **Modo AUTO**: Detección continua activable/desactivable con un botón.
*   **Test con Imagen**: Script para probar la detección en imágenes estáticas.

## Requisitos

*   Python 3.10+ (Probado en Python 3.13)
*   Webcam o entrada de video compatible con OpenCV.
*   **Opcional**: Modelo YOLOv8 entrenado para placas (`models/plate_detector.pt`).

## Instalación

1.  Clona el repositorio:
    ```powershell
    git clone https://github.com/xavker/deteccion_placas.git
    cd deteccion_placas
    ```

2.  Crea un ambiente virtual:
    ```powershell
    python -m venv .venv
    ```

3.  Activa el ambiente virtual:
    *   **PowerShell**: `.\.venv\Scripts\Activate.ps1`
    *   **CMD**: `.\.venv\Scripts\activate`

4.  Instala las dependencias:
    ```powershell
    pip install -r requirements.txt
    ```

## Uso

### Modo Interfaz Gráfica (tiempo real)
```powershell
python main.py
```

1.  **START**: Inicia la cámara.
2.  **AUTO**: Activa la detección automática de placas.
3.  **STOP**: Detiene la cámara y la detección.
4.  Ajusta el **slider de confianza** según la calidad de la imagen.

### Modo Test con Imagen
```powershell
# Con tu propia imagen
python test_image.py "ruta/a/foto_auto.jpg"

# Sin argumentos (descarga imagen de prueba)
python test_image.py
```

## Estructura del Proyecto

*   `main.py`: Punto de entrada y orquestación de módulos.
*   `gui_manager.py`: Interfaz gráfica con panel de video, controles y resultados.
*   `vision_processor.py`: Captura de video y pipeline de detección en hilo separado.
*   `plate_detector.py`: Localización de placas con YOLOv8.
*   `ocr_reader.py`: Lectura de caracteres con EasyOCR (optimizado para placas EC).
*   `test_image.py`: Script de prueba con imágenes estáticas.
*   `models/`: Directorio para modelos YOLOv8 personalizados (`.pt`).
*   `requirements.txt`: Dependencias del proyecto.

## Modelo de Detección

El sistema funciona en **dos modos**:

| Modo | Archivo | Descripción |
|---|---|---|
| **Sin modelo propio** | `yolov8n.pt` (auto-descarga) | Detecta vehículos con COCO, luego busca placas por contornos con OpenCV. |
| **Con modelo propio** | `models/plate_detector.pt` | Detecta placas directamente. Mayor precisión. |

Para mejores resultados, se recomienda entrenar un modelo específico para placas o descargar uno pre-entrenado de [Roboflow Universe](https://universe.roboflow.com/) y colocarlo como `models/plate_detector.pt`.

## Tecnologías

| Componente | Tecnología |
|---|---|
| Detección de objetos | YOLOv8 (Ultralytics) |
| OCR | EasyOCR |
| Interfaz gráfica | CustomTkinter |
| Procesamiento de imagen | OpenCV, Pillow |
| Lenguaje | Python 3.13 |

---
**Desarrollado por**: [XAVKER](https://github.com/xavker)  
**Fecha de Creación**: Abril 2026
