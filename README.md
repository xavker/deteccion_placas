# Plantilla de Interfaz para Visión de Máquina (XAVKER)

Esta es una plantilla profesional desarrollada en Python para servir como base en proyectos de visión artificial y control industrial. Utiliza una interfaz moderna y un motor de procesamiento de video optimizado en hilos.

## Características

*   **Interfaz Moderna**: Desarrollada con `CustomTkinter` para un look profesional y modo oscuro nativo.
*   **Procesamiento Asíncrono**: La captura de video se realiza en un hilo separado para mantener la fluidez de la UI (sin lag).
*   **Panel de Control Dual**: 
    *   Lado izquierdo dedicado al stream de video.
    *   Lado derecho con botones de **Start**, **Stop** y **Auto**.
*   **Indicadores LED**: LEDs visuales que cambian de color según el estado del sistema.
*   **Pie de Página Personalizado**: Información de versión, fecha y desarrollador.

## Requisitos

*   Python 3.10+ (Probado en Python 3.13)
*   Webcam o entrada de video compatible con OpenCV.

## Instalación

1.  Clona el repositorio:
    ```powershell
    git clone https://github.com/tu-usuario/plantilla-vision-interfaz-xavker.git
    cd "Interfraz grafica vision de maquina"
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

Para iniciar la aplicación, simplemente ejecuta:
```powershell
python main.py
```

## Estructura del Proyecto

*   `main.py`: Punto de entrada y orquestación.
*   `gui_manager.py`: Gestión integral de la interfaz gráfica y widgets.
*   `vision_processor.py`: Lógica de captura y procesamiento de frames.
*   `requirements.txt`: Dependencias del proyecto con versiones fijas.

---
**Desarrollado por**: [XAVKER](https://github.com/tu-usuario)  
**Fecha de Creación**: Abril 2026
