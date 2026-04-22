import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from datetime import datetime

class App(ctk.CTk):
    def __init__(self, vision_processor):
        super().__init__()

        self.vision_processor = vision_processor
        
        # Configuración de la ventana
        self.title("Sistema de Visión de Máquina - XAVKER")
        self.geometry("1100x700")
        
        # Tema profesional
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Configuración de grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # El cuerpo principal se expande

        self.create_header()
        self.create_main_layout()
        self.create_footer()

        # Variable para controlar la actualización de la interfaz
        self.is_updating = True
        self.update_gui()

    def create_header(self):
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, height=80)
        self.header_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="CONTROL DE VISIÓN ARTIFICIAL", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(10, 0))
        
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame, 
            text="Interfaz de Monitoreo y Procesamiento en Tiempo Real", 
            font=ctk.CTkFont(size=14)
        )
        self.subtitle_label.pack(pady=(0, 10))

    def create_main_layout(self):
        # Contenedor principal con dos columnas
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_container.grid_columnconfigure(0, weight=3) # Video más ancho
        self.main_container.grid_columnconfigure(1, weight=1) # Controles

        # --- LADO IZQUIERDO: VISTA DE CÁMARA ---
        self.video_frame = ctk.CTkFrame(self.main_container)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Cargando Cámara...", fg_color="black")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # --- LADO DERECHO: PANEL DE CONTROL ---
        self.control_panel = ctk.CTkFrame(self.main_container)
        self.control_panel.grid(row=0, column=1, sticky="nsew")
        
        self.panel_title = ctk.CTkLabel(
            self.control_panel, 
            text="PANEL DE CONTROL", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.panel_title.pack(pady=20)

        # Botones y LEDs
        self.btn_start = self.create_control_group("START", "green", self.on_start)
        self.btn_stop = self.create_control_group("STOP", "red", self.on_stop)
        self.btn_auto = self.create_control_group("AUTO", "orange", self.on_auto)

    def create_control_group(self, name, color, command):
        frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=10)
        
        # Botón
        btn = ctk.CTkButton(
            frame, 
            text=name, 
            command=command,
            width=100,
            font=ctk.CTkFont(weight="bold")
        )
        btn.pack(side="left", padx=(0, 10))
        
        # LED (Implementado como un CTkFrame pequeño y redondo)
        led = ctk.CTkFrame(
            frame, 
            width=20, 
            height=20, 
            corner_radius=10, 
            fg_color="gray30" # Inicialmente apagado
        )
        led.pack(side="left")
        
        # Guardar referencia al LED para cambiar su color
        setattr(self, f"led_{name.lower()}", led)
        return btn

    def create_footer(self):
        self.footer_frame = ctk.CTkFrame(self, corner_radius=0, height=40)
        self.footer_frame.grid(row=2, column=0, sticky="nsew")
        
        today = datetime.now().strftime("%Y-%m-%d")
        info_text = f"Versión: 1.0.0  |  Fecha: {today}  |  Desarrollador: XAVKER"
        
        self.footer_label = ctk.CTkLabel(
            self.footer_frame, 
            text=info_text, 
            font=ctk.CTkFont(size=10)
        )
        self.footer_label.pack(side="right", padx=20, pady=10)

    def update_gui(self):
        if self.is_updating:
            frame = self.vision_processor.get_latest_frame()
            if frame is not None:
                # Redimensionar frame para ajustar al label
                # Obtenemos el tamaño del video_label
                w = self.video_label.winfo_width()
                h = self.video_label.winfo_height()
                
                if w > 1 and h > 1:
                    img = Image.fromarray(frame)
                    # Mantener relación de aspecto
                    img.thumbnail((w, h))
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                    self.video_label.configure(image=ctk_img, text="")
            
            self.after(10, self.update_gui)

    # Handlers para los botones
    def on_start(self):
        print("Botón Start presionado")
        self.vision_processor.start()
        self.led_start.configure(fg_color="green")
        self.led_stop.configure(fg_color="gray30")
        self.led_auto.configure(fg_color="gray30")

    def on_stop(self):
        print("Botón Stop presionado")
        self.vision_processor.stop()
        self.led_start.configure(fg_color="gray30")
        self.led_stop.configure(fg_color="red")
        self.led_auto.configure(fg_color="gray30")

    def on_auto(self):
        print("Botón Auto presionado")
        self.led_auto.configure(fg_color="orange")
        # Aquí iría la lógica del modo automático

    def on_closing(self):
        self.is_updating = False
        self.vision_processor.stop()
        self.destroy()
