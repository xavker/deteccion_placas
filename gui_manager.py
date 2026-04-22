import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from datetime import datetime

class App(ctk.CTk):
    def __init__(self, vision_processor):
        super().__init__()

        self.vision_processor = vision_processor
        
        # Configuración de la ventana
        self.title("Detector de Placas Vehiculares — XAVKER")
        self.geometry("1300x800")
        self.minsize(1100, 700)
        
        # Tema profesional
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Configuración de grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.create_header()
        self.create_main_layout()
        self.create_footer()

        # Variable para controlar la actualización de la interfaz
        self.is_updating = True
        self.auto_active = False
        self.update_gui()

    def create_header(self):
        self.header_frame = ctk.CTkFrame(self, corner_radius=10, height=80)
        self.header_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="🚗  DETECTOR DE PLACAS VEHICULARES", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(10, 0))
        
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame, 
            text="YOLOv8 + EasyOCR  |  Placas Ecuatorianas  |  Tiempo Real", 
            font=ctk.CTkFont(size=13),
            text_color="gray60"
        )
        self.subtitle_label.pack(pady=(0, 10))

    def create_main_layout(self):
        # Contenedor principal con dos columnas
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.main_container.grid_columnconfigure(0, weight=3)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # --- LADO IZQUIERDO: VISTA DE CÁMARA ---
        self.video_frame = ctk.CTkFrame(self.main_container, corner_radius=10)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.video_label = ctk.CTkLabel(
            self.video_frame, 
            text="📷  Presiona START para iniciar la cámara", 
            fg_color="gray10",
            corner_radius=8,
            font=ctk.CTkFont(size=14)
        )
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # --- LADO DERECHO: PANEL DE CONTROL ---
        self.right_panel = ctk.CTkScrollableFrame(self.main_container, corner_radius=10)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        
        # ---- SECCIÓN: Controles ----
        self.panel_title = ctk.CTkLabel(
            self.right_panel, 
            text="PANEL DE CONTROL", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.panel_title.pack(pady=(15, 10))

        # Botones y LEDs
        self.btn_start = self.create_control_group("START", "green", self.on_start)
        self.btn_stop = self.create_control_group("STOP", "red", self.on_stop)
        self.btn_auto = self.create_control_group("AUTO", "orange", self.on_auto)

        # Separador
        self.sep1 = ctk.CTkFrame(self.right_panel, height=2, fg_color="gray30")
        self.sep1.pack(fill="x", padx=15, pady=15)

        # ---- SECCIÓN: Confianza ----
        self.conf_label = ctk.CTkLabel(
            self.right_panel,
            text="Umbral de Confianza",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.conf_label.pack(pady=(0, 5))

        self.conf_value_label = ctk.CTkLabel(
            self.right_panel,
            text="40%",
            font=ctk.CTkFont(size=12),
            text_color="gray60"
        )
        self.conf_value_label.pack()

        self.conf_slider = ctk.CTkSlider(
            self.right_panel,
            from_=10,
            to=95,
            number_of_steps=17,
            command=self.on_confidence_change
        )
        self.conf_slider.set(40)
        self.conf_slider.pack(padx=15, pady=(0, 10))

        # Separador
        self.sep2 = ctk.CTkFrame(self.right_panel, height=2, fg_color="gray30")
        self.sep2.pack(fill="x", padx=15, pady=10)

        # ---- SECCIÓN: Última Placa Detectada ----
        self.result_title = ctk.CTkLabel(
            self.right_panel,
            text="ÚLTIMA PLACA",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.result_title.pack(pady=(5, 5))

        self.plate_text_label = ctk.CTkLabel(
            self.right_panel,
            text="---",
            font=ctk.CTkFont(family="Consolas", size=28, weight="bold"),
            text_color="#00FF88",
            fg_color="gray10",
            corner_radius=8,
            height=50
        )
        self.plate_text_label.pack(fill="x", padx=15, pady=5)

        self.plate_conf_label = ctk.CTkLabel(
            self.right_panel,
            text="Confianza: —",
            font=ctk.CTkFont(size=11),
            text_color="gray50"
        )
        self.plate_conf_label.pack(pady=(0, 5))

        # Imagen de la placa recortada
        self.plate_image_label = ctk.CTkLabel(
            self.right_panel,
            text="Sin imagen",
            fg_color="gray10",
            corner_radius=8,
            height=60,
            font=ctk.CTkFont(size=10)
        )
        self.plate_image_label.pack(fill="x", padx=15, pady=5)

        # Separador
        self.sep3 = ctk.CTkFrame(self.right_panel, height=2, fg_color="gray30")
        self.sep3.pack(fill="x", padx=15, pady=10)

        # ---- SECCIÓN: Historial ----
        self.history_title = ctk.CTkLabel(
            self.right_panel,
            text="📋  HISTORIAL",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.history_title.pack(pady=(5, 5))

        self.btn_clear_history = ctk.CTkButton(
            self.right_panel,
            text="Limpiar",
            width=80,
            height=28,
            fg_color="gray30",
            hover_color="gray40",
            font=ctk.CTkFont(size=11),
            command=self.on_clear_history
        )
        self.btn_clear_history.pack(pady=(0, 5))

        self.history_frame = ctk.CTkFrame(self.right_panel, fg_color="gray10", corner_radius=8)
        self.history_frame.pack(fill="both", padx=15, pady=(0, 15), expand=True)

        self.history_labels = []

    def create_control_group(self, name, color, command):
        frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=8)
        
        # Botón
        btn = ctk.CTkButton(
            frame, 
            text=name, 
            command=command,
            width=120,
            height=36,
            font=ctk.CTkFont(weight="bold")
        )
        btn.pack(side="left", padx=(0, 10))
        
        # LED
        led = ctk.CTkFrame(
            frame, 
            width=20, 
            height=20, 
            corner_radius=10, 
            fg_color="gray30"
        )
        led.pack(side="left")
        
        # Guardar referencia al LED
        setattr(self, f"led_{name.lower()}", led)
        return btn

    def create_footer(self):
        self.footer_frame = ctk.CTkFrame(self, corner_radius=0, height=40)
        self.footer_frame.grid(row=2, column=0, sticky="nsew")
        
        today = datetime.now().strftime("%Y-%m-%d")
        info_text = f"Versión: 2.0.0  |  Fecha: {today}  |  Desarrollador: XAVKER  |  Motor: YOLOv8 + EasyOCR"
        
        self.footer_label = ctk.CTkLabel(
            self.footer_frame, 
            text=info_text, 
            font=ctk.CTkFont(size=10),
            text_color="gray50"
        )
        self.footer_label.pack(side="right", padx=20, pady=10)

    def update_gui(self):
        """Actualiza el video feed y los resultados de detección."""
        if self.is_updating:
            # Actualizar video
            frame = self.vision_processor.get_latest_frame()
            if frame is not None:
                w = self.video_label.winfo_width()
                h = self.video_label.winfo_height()
                
                if w > 1 and h > 1:
                    img = Image.fromarray(frame)
                    img.thumbnail((w, h))
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                    self.video_label.configure(image=ctk_img, text="")

            # Actualizar resultados de detección
            if self.auto_active:
                self._update_detection_display()

            self.after(30, self.update_gui)

    def _update_detection_display(self):
        """Actualiza los widgets de resultados de detección."""
        results = self.vision_processor.get_detection_results()

        # Actualizar texto de placa
        if results['plate_text']:
            self.plate_text_label.configure(text=results['plate_text'])
            conf_pct = results['plate_conf'] * 100
            self.plate_conf_label.configure(text=f"Confianza: {conf_pct:.1f}%")

            # Actualizar imagen recortada de la placa
            if results['plate_image'] is not None and results['plate_image'].size > 0:
                try:
                    plate_rgb = cv2.cvtColor(results['plate_image'], cv2.COLOR_BGR2RGB)
                    plate_pil = Image.fromarray(plate_rgb)
                    plate_pil = plate_pil.resize((200, 60), Image.LANCZOS)
                    plate_ctk = ctk.CTkImage(light_image=plate_pil, dark_image=plate_pil, size=(200, 60))
                    self.plate_image_label.configure(image=plate_ctk, text="")
                except Exception:
                    pass

        # Actualizar historial
        history = results['history']
        if history:
            self._refresh_history(history)

    def _refresh_history(self, history):
        """Refresca el panel de historial de placas."""
        # Limpiar labels existentes
        for label in self.history_labels:
            label.destroy()
        self.history_labels.clear()

        # Mostrar las últimas entradas (más reciente arriba)
        for text, conf, timestamp in reversed(history[-20:]):
            conf_pct = conf * 100
            entry_text = f"  {timestamp}  |  {text}  |  {conf_pct:.0f}%"
            
            label = ctk.CTkLabel(
                self.history_frame,
                text=entry_text,
                font=ctk.CTkFont(family="Consolas", size=11),
                text_color="#E0E0E0",
                anchor="w"
            )
            label.pack(fill="x", padx=8, pady=2)
            self.history_labels.append(label)

    # ---- Handlers ----
    def on_start(self):
        print("[GUI] Botón Start presionado")
        self.vision_processor.start()
        self.led_start.configure(fg_color="green")
        self.led_stop.configure(fg_color="gray30")
        self.led_auto.configure(fg_color="gray30")

    def on_stop(self):
        print("[GUI] Botón Stop presionado")
        self.vision_processor.stop()
        self.auto_active = False
        self.led_start.configure(fg_color="gray30")
        self.led_stop.configure(fg_color="red")
        self.led_auto.configure(fg_color="gray30")

    def on_auto(self):
        print("[GUI] Botón Auto presionado")
        self.auto_active = not self.auto_active
        
        if self.auto_active:
            # Si la cámara no está corriendo, iniciarla
            if not self.vision_processor.running:
                self.vision_processor.start()
                self.led_start.configure(fg_color="green")
                self.led_stop.configure(fg_color="gray30")
            
            self.vision_processor.set_auto_mode(True)
            self.led_auto.configure(fg_color="orange")
            print("[GUI] Detección automática ACTIVADA")
        else:
            self.vision_processor.set_auto_mode(False)
            self.led_auto.configure(fg_color="gray30")
            print("[GUI] Detección automática DESACTIVADA")

    def on_confidence_change(self, value):
        conf = int(value)
        self.conf_value_label.configure(text=f"{conf}%")
        self.vision_processor.set_confidence(conf / 100.0)

    def on_clear_history(self):
        """Limpia el historial de placas."""
        self.vision_processor.plate_history.clear()
        self.plate_text_label.configure(text="---")
        self.plate_conf_label.configure(text="Confianza: —")
        self.plate_image_label.configure(image=None, text="Sin imagen")
        for label in self.history_labels:
            label.destroy()
        self.history_labels.clear()
        print("[GUI] Historial limpiado")

    def on_closing(self):
        self.is_updating = False
        self.vision_processor.stop()
        self.destroy()
