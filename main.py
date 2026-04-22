from vision_processor import VisionProcessor
from gui_manager import App

def main():
    # Inicializar el procesador de visión
    vision_processor = VisionProcessor()

    # Inicializar la aplicación
    app = App(vision_processor)
    
    # Manejar el cierre de la ventana correctamente
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Iniciar el bucle principal de la interfaz
    app.mainloop()

if __name__ == "__main__":
    main()

