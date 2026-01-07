import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
import requests
import platform
import subprocess
import os
import json
import re
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from threading import Lock
from collections import deque


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pothole_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PotholeDetection:
    """Pothole Detection Application using YOLOv10 and Tkinter GUI."""
    
    # Constants
    POTHOLE_CLASS_ID = 0
    MIN_SAVE_INTERVAL_SECONDS = 2.0  # Minimum time between saves to prevent duplicates
    MAX_DISPLAY_SIZE = (800, 600)  # Increased display size
    DEFAULT_CONFIDENCE = 0.5
    
    # Color scheme
    COLOR_PRIMARY = "#2C3E50"
    COLOR_SECONDARY = "#3498DB"
    COLOR_SUCCESS = "#27AE60"
    COLOR_DANGER = "#E74C3C"
    COLOR_WARNING = "#F39C12"
    COLOR_INFO = "#3498DB"
    COLOR_BG = "#ECF0F1"
    
    def __init__(self, root, config_path: Optional[str] = None):
        """
        Initialize the Pothole Detection application.
        
        Args:
            root: Tkinter root window
            config_path: Optional path to configuration JSON file
        """
        self.root = root
        self.root.title("Pothole Detection System - YOLOv10")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.configure(bg=self.COLOR_BG)
        
        # Set window size and center it
        window_width = 1200
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize paths
        self.model_path = self.config.get("model_path", "runs/segment/train/weights/best.pt")
        self.class_list_path = self.config.get("class_list_path", "coco.txt")
        self.save_dir = Path(self.config.get("save_dir", "ResultPrediction"))
        self.coordinates_dir = Path(self.config.get("coordinates_dir", "ResultLocation"))
        self.google_api_key = self.config.get("google_api_key", "")
        
        # Create directories if they don't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.coordinates_dir.mkdir(parents=True, exist_ok=True)

        # Initialize YOLO model
        self.model = None
        self._load_model()

        # Load class list
        try:
            with open(self.class_list_path, "r", encoding="utf-8") as f:
                self.class_list = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(self.class_list)} classes from {self.class_list_path}")
        except FileNotFoundError:
            self.logger.warning(f"Class list file not found at {self.class_list_path}, using default")
            self.class_list = ["pothole"]  # Default fallback

        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'total_saved': 0,
            'session_start_time': datetime.now(),
            'frames_processed': 0,
            'fps': 0.0
        }
        
        # Thread safety
        self.save_lock = Lock()
        self.frame_lock = Lock()
        
        # Detection tracking for debouncing
        self.last_save_time = None
        self.last_detection_boxes = []
        self.fps_history = deque(maxlen=30)  # Store last 30 FPS values
        self.last_frame_time = time.time()

        # Video source
        self.video_source_type = "camera"  # "camera" or "file"
        self.video_file_path = None

        # Create GUI elements
        self._create_gui()

        # Set up video capture
        self.cap = None
        self.video_on = False
        self.update_id = None  # Track scheduled update ID

        # Initialize sequence number for naming cropped pothole images
        self.sequence_number = self._get_next_sequence_number()
        
        # Bind keyboard shortcuts
        self.root.bind('<F1>', lambda e: self._show_help())
        self.root.bind('<F2>', lambda e: self._show_settings())
        self.root.bind('<space>', lambda e: self.toggle_video())
        self.root.bind('<Escape>', lambda e: self.stop_video())

    def _load_model(self):
        """Load YOLO model with error handling."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                # Try to find best.pt in default location
                default_path = Path("runs/segment/train/weights/best.pt")
                if default_path.exists():
                    self.model_path = str(default_path)
                    self.logger.info(f"Using default model: {self.model_path}")
                else:
                    # Show dialog to select model
                    self._show_model_selection_dialog()
                    return
            
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            messagebox.showerror(
                "Model Loading Error", 
                f"Failed to load model:\n{self.model_path}\n\nError: {e}\n\nPlease select a valid model file."
            )
            self._show_model_selection_dialog()

    def _show_model_selection_dialog(self):
        """Show dialog to select model file."""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("PyTorch Model", "*.pt"),
                ("All Files", "*.*")
            ],
            initialdir=os.getcwd()
        )
        if file_path:
            self.model_path = file_path
            try:
                self.model = YOLO(self.model_path)
                self.logger.info(f"Model loaded from: {self.model_path}")
                messagebox.showinfo("Success", f"Model loaded successfully:\n{self.model_path}")
            except Exception as e:
                self.logger.error(f"Error loading selected model: {e}")
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from JSON file or use defaults."""
        default_config = {
            "model_path": "runs/segment/train/weights/best.pt",
            "class_list_path": "coco.txt",
            "save_dir": "ResultPrediction",
            "coordinates_dir": "ResultLocation",
            "google_api_key": "",
            "confidence_threshold": 0.5,
            "camera_index": 0,
            "update_interval_ms": 33,  # ~30 FPS
            "enable_frame_skip": False,
            "frame_skip_factor": 1  # Process every Nth frame
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Could not load config file: {e}")
                messagebox.showwarning("Config Warning", f"Could not load config file:\n{e}\nUsing defaults.")
        else:
            self.logger.info("No config file specified, using defaults")
        
        return default_config
    
    def _get_next_sequence_number(self) -> int:
        """Get the next sequence number by checking existing files."""
        try:
            existing_files = list(self.save_dir.glob("pothole_*.jpg"))
            if not existing_files:
                return 1
            
            numbers = []
            for file in existing_files:
                match = re.search(r'pothole_(\d+)\.jpg', file.name)
                if match:
                    numbers.append(int(match.group(1)))
            
            return max(numbers) + 1 if numbers else 1
        except Exception as e:
            self.logger.warning(f"Error determining sequence number: {e}")
            return 1

    def _create_gui(self):
        """Create the GUI elements with improved layout."""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Select Model...", command=self._show_model_selection_dialog)
        file_menu.add_command(label="Select Video File...", command=self._select_video_file)
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self._show_settings, accelerator="F2")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Help", command=self._show_help, accelerator="F1")
        help_menu.add_command(label="About", command=self._show_about)
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a title label with better styling
        title_frame = tk.Frame(main_container, bg=self.COLOR_BG)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = tk.Label(
            title_frame, 
            text="🕳️ Pothole Detection System", 
            font=("Helvetica", 20, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_PRIMARY
        )
        self.title_label.pack(side=tk.LEFT)
        
        # Model status indicator
        self.model_status_label = tk.Label(
            title_frame,
            text="✓ Model Ready",
            font=("Helvetica", 10),
            bg=self.COLOR_BG,
            fg=self.COLOR_SUCCESS
        )
        self.model_status_label.pack(side=tk.RIGHT, padx=10)

        # Create a frame to contain video feed and control panel
        content_frame = tk.Frame(main_container, bg=self.COLOR_BG)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Video display frame with border
        video_frame = tk.LabelFrame(
            content_frame, 
            text="Video Feed", 
            font=("Helvetica", 12, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_PRIMARY,
            relief=tk.RAISED,
            bd=2
        )
        video_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        # Create a label to display the video feed
        self.video_label = tk.Label(
            video_frame, 
            bg="black", 
            relief=tk.SUNKEN, 
            bd=2,
            text="No video feed\nClick 'Start Video' to begin",
            fg="white",
            font=("Helvetica", 12)
        )
        self.video_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Video info frame
        video_info_frame = tk.Frame(video_frame, bg=self.COLOR_BG)
        video_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # FPS label
        self.fps_label = tk.Label(
            video_info_frame,
            text="FPS: 0.0",
            font=("Helvetica", 10, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_SUCCESS
        )
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        # Video source label
        self.video_source_label = tk.Label(
            video_info_frame,
            text="Source: Camera",
            font=("Helvetica", 9),
            bg=self.COLOR_BG,
            fg=self.COLOR_INFO
        )
        self.video_source_label.pack(side=tk.RIGHT, padx=5)

        # Control panel frame
        control_frame = tk.LabelFrame(
            content_frame, 
            text="Controls", 
            font=("Helvetica", 12, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_PRIMARY,
            relief=tk.RAISED,
            bd=2
        )
        control_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

        # Video source selection
        source_frame = tk.Frame(control_frame, bg=self.COLOR_BG)
        source_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(
            source_frame, 
            text="Video Source:", 
            font=("Helvetica", 10, "bold"),
            bg=self.COLOR_BG
        ).pack(anchor=tk.W)
        
        source_btn_frame = tk.Frame(source_frame, bg=self.COLOR_BG)
        source_btn_frame.pack(fill=tk.X, pady=5)
        
        self.camera_btn = tk.Button(
            source_btn_frame,
            text="📷 Camera",
            command=lambda: self._set_video_source("camera"),
            width=12,
            font=("Helvetica", 9),
            bg=self.COLOR_SECONDARY,
            fg="white",
            activebackground="#2980B9"
        )
        self.camera_btn.pack(side=tk.LEFT, padx=2)
        
        self.file_btn = tk.Button(
            source_btn_frame,
            text="📁 File",
            command=self._select_video_file,
            width=12,
            font=("Helvetica", 9),
            bg=self.COLOR_SECONDARY,
            fg="white",
            activebackground="#2980B9"
        )
        self.file_btn.pack(side=tk.LEFT, padx=2)

        # Start/Stop button with better styling
        self.start_stop_button = tk.Button(
            control_frame, 
            text="▶ Start Video", 
            command=self.toggle_video,
            width=25,
            height=3,
            font=("Helvetica", 12, "bold"),
            bg=self.COLOR_SUCCESS,
            fg="white",
            activebackground="#229954",
            relief=tk.RAISED,
            bd=3
        )
        self.start_stop_button.pack(pady=15, padx=10)

        # Status label with better styling
        status_frame = tk.Frame(control_frame, bg=self.COLOR_BG)
        status_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(
            status_frame, 
            text="Status:", 
            font=("Helvetica", 10, "bold"),
            bg=self.COLOR_BG
        ).pack(anchor=tk.W)
        
        self.status_label = tk.Label(
            status_frame,
            text="● Stopped",
            font=("Helvetica", 11, "bold"),
            fg=self.COLOR_DANGER,
            bg=self.COLOR_BG
        )
        self.status_label.pack(anchor=tk.W, pady=(5, 0))

        # Statistics panel with better styling
        stats_frame = tk.LabelFrame(
            control_frame,
            text="Statistics",
            font=("Helvetica", 11, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_PRIMARY
        )
        stats_frame.pack(pady=10, padx=10, fill=tk.X)

        stats_inner = tk.Frame(stats_frame, bg=self.COLOR_BG)
        stats_inner.pack(pady=10, padx=10, fill=tk.X)

        # Detection count
        self.detection_label = tk.Label(
            stats_inner,
            text="🔍 Detections: 0",
            font=("Helvetica", 10),
            anchor=tk.W,
            bg=self.COLOR_BG
        )
        self.detection_label.pack(fill=tk.X, pady=5)

        # Saved count
        self.saved_label = tk.Label(
            stats_inner,
            text="💾 Saved: 0",
            font=("Helvetica", 10),
            anchor=tk.W,
            bg=self.COLOR_BG
        )
        self.saved_label.pack(fill=tk.X, pady=5)

        # Session time
        self.session_label = tk.Label(
            stats_inner,
            text="⏱️ Session: 0:00:00",
            font=("Helvetica", 10),
            anchor=tk.W,
            bg=self.COLOR_BG
        )
        self.session_label.pack(fill=tk.X, pady=5)

        # Confidence threshold control with better styling
        conf_frame = tk.LabelFrame(
            control_frame,
            text="Confidence Threshold",
            font=("Helvetica", 11, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_PRIMARY
        )
        conf_frame.pack(pady=10, padx=10, fill=tk.X)
        
        conf_inner = tk.Frame(conf_frame, bg=self.COLOR_BG)
        conf_inner.pack(pady=10, padx=10, fill=tk.X)
        
        self.confidence_var = tk.DoubleVar(value=self.config.get("confidence_threshold", 0.5))
        self.confidence_scale = tk.Scale(
            conf_inner,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var,
            command=self._on_confidence_change,
            bg=self.COLOR_BG,
            fg=self.COLOR_PRIMARY,
            highlightbackground=self.COLOR_BG,
            length=200
        )
        self.confidence_scale.pack(fill=tk.X, pady=5)
        
        self.confidence_value_label = tk.Label(
            conf_inner,
            text=f"{self.confidence_var.get():.2f}",
            font=("Helvetica", 12, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_PRIMARY
        )
        self.confidence_value_label.pack()

        # Quick actions frame
        actions_frame = tk.Frame(control_frame, bg=self.COLOR_BG)
        actions_frame.pack(pady=10, padx=10, fill=tk.X)
        
        open_folder_btn = tk.Button(
            actions_frame,
            text="📂 Open Results Folder",
            command=self._open_results_folder,
            width=25,
            font=("Helvetica", 9),
            bg=self.COLOR_INFO,
            fg="white",
            activebackground="#2980B9"
        )
        open_folder_btn.pack(pady=2)

    def _set_video_source(self, source: str):
        """Set video source type."""
        self.video_source_type = source
        if source == "camera":
            self.video_source_label.config(text="Source: Camera")
            self.video_file_path = None
        else:
            self.video_source_label.config(text=f"Source: {Path(self.video_file_path).name if self.video_file_path else 'File'}")

    def _select_video_file(self):
        """Select video file for processing."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.video_file_path = file_path
            self.video_source_type = "file"
            self.video_source_label.config(text=f"Source: {Path(file_path).name}")
            messagebox.showinfo("Video Selected", f"Video file selected:\n{Path(file_path).name}")

    def _open_results_folder(self):
        """Open results folder in file explorer."""
        try:
            if platform.system() == "Windows":
                os.startfile(self.save_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(self.save_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(self.save_dir)])
        except Exception as e:
            self.logger.error(f"Error opening folder: {e}")
            messagebox.showerror("Error", f"Could not open folder:\n{e}")

    def _on_confidence_change(self, value):
        """Update confidence threshold label."""
        self.confidence_value_label.config(text=f"{float(value):.2f}")

    def _show_settings(self):
        """Show settings window."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        settings_window.configure(bg=self.COLOR_BG)
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Center window
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (settings_window.winfo_screenheight() // 2) - (400 // 2)
        settings_window.geometry(f"500x400+{x}+{y}")
        
        # Settings content
        tk.Label(
            settings_window,
            text="Settings",
            font=("Helvetica", 16, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_PRIMARY
        ).pack(pady=20)
        
        # Camera index
        cam_frame = tk.Frame(settings_window, bg=self.COLOR_BG)
        cam_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(cam_frame, text="Camera Index:", bg=self.COLOR_BG).pack(side=tk.LEFT)
        camera_var = tk.IntVar(value=self.config.get("camera_index", 0))
        camera_spin = tk.Spinbox(cam_frame, from_=0, to=10, textvariable=camera_var, width=10)
        camera_spin.pack(side=tk.RIGHT)
        
        # Save interval
        interval_frame = tk.Frame(settings_window, bg=self.COLOR_BG)
        interval_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(interval_frame, text="Save Interval (seconds):", bg=self.COLOR_BG).pack(side=tk.LEFT)
        interval_var = tk.DoubleVar(value=self.MIN_SAVE_INTERVAL_SECONDS)
        interval_spin = tk.Spinbox(interval_frame, from_=0.5, to=10.0, increment=0.5, textvariable=interval_var, width=10)
        interval_spin.pack(side=tk.RIGHT)
        
        # Buttons
        btn_frame = tk.Frame(settings_window, bg=self.COLOR_BG)
        btn_frame.pack(pady=20)
        
        def save_settings():
            self.config["camera_index"] = camera_var.get()
            self.MIN_SAVE_INTERVAL_SECONDS = interval_var.get()
            messagebox.showinfo("Settings Saved", "Settings have been saved!")
            settings_window.destroy()
        
        tk.Button(
            btn_frame,
            text="Save",
            command=save_settings,
            bg=self.COLOR_SUCCESS,
            fg="white",
            width=15,
            font=("Helvetica", 10)
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="Cancel",
            command=settings_window.destroy,
            bg=self.COLOR_DANGER,
            fg="white",
            width=15,
            font=("Helvetica", 10)
        ).pack(side=tk.LEFT, padx=5)

    def _show_help(self):
        """Show help dialog."""
        help_text = """
Pothole Detection System - Help

Keyboard Shortcuts:
• Space: Start/Stop video
• Escape: Stop video
• F1: Show this help
• F2: Open settings

Controls:
• Start Video: Begin detection from camera or video file
• Confidence Threshold: Adjust detection sensitivity (0.1 - 1.0)
• Select Model: Choose a different YOLO model file
• Select Video File: Process a video file instead of camera

Tips:
• Higher confidence = fewer false positives
• Lower confidence = more detections (may include false positives)
• Detected images are saved automatically
• Check the Results folder for saved images

For more information, see README.md
        """
        messagebox.showinfo("Help", help_text)

    def _show_about(self):
        """Show about dialog."""
        about_text = """
Pothole Detection System
Version 1.0

A real-time pothole detection application using YOLOv10.

Features:
• Real-time detection from camera or video files
• Automatic image saving
• GPS coordinate recording (optional)
• Configurable confidence threshold
• Comprehensive statistics

Developed with:
• YOLOv10 (Ultralytics)
• OpenCV
• Tkinter

© 2024
        """
        messagebox.showinfo("About", about_text)

    def toggle_video(self):
        """Toggle video feed on/off."""
        if self.video_on:
            self.stop_video()
        else:
            self.start_video()

    def start_video(self):
        """Start the video feed."""
        if self.model is None:
            messagebox.showerror("Error", "No model loaded. Please select a model file.")
            return
        
        if self.video_source_type == "file" and not self.video_file_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return
        
        if self.cap is None:
            if self.video_source_type == "camera":
                camera_index = self.config.get("camera_index", 0)
                self.logger.info(f"Opening camera {camera_index}")
                self.cap = cv2.VideoCapture(camera_index)
                
                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                self.logger.info(f"Opening video file: {self.video_file_path}")
                self.cap = cv2.VideoCapture(self.video_file_path)
            
            if not self.cap.isOpened():
                error_msg = f"Failed to open {'camera' if self.video_source_type == 'camera' else 'video file'}"
                self.logger.error(error_msg)
                self.status_label.config(text="● Error", fg=self.COLOR_DANGER)
                messagebox.showerror("Error", error_msg)
                return
        
        self.video_on = True
        self.start_stop_button.configure(
            text="⏸ Stop Video", 
            bg=self.COLOR_DANGER, 
            activebackground="#C0392B"
        )
        self.status_label.config(text="● Running", fg=self.COLOR_SUCCESS)
        self.stats['session_start_time'] = datetime.now()
        self.last_frame_time = time.time()
        self.update_video()
        self.logger.info("Video feed started")

    def stop_video(self):
        """Stop the video feed."""
        self.video_on = False
        if self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None
        self.start_stop_button.configure(
            text="▶ Start Video", 
            bg=self.COLOR_SUCCESS, 
            activebackground="#229954"
        )
        self.status_label.config(text="● Stopped", fg=self.COLOR_DANGER)
        self.logger.info("Video feed stopped")
        
        # Update statistics
        self._update_statistics_display()

    def get_wifi_location(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get location using WiFi access points and Google Geolocation API.
        
        Returns:
            Tuple of (latitude, longitude) or (None, None) if failed
        """
        if not self.google_api_key or self.google_api_key == "":
            self.logger.debug("Google API key not configured, skipping location")
            return None, None

        def get_wifi_mac_addresses() -> list:
            """Get MAC addresses of nearby WiFi networks."""
            try:
                system = platform.system().lower()
                if 'darwin' in system:
                    # macOS
                    cmd = ["/System/Library/PrivateFrameworks/Apple80211.framework/Resources/airport", "-s"]
                    results = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
                elif 'win' in system:
                    # Windows
                    cmd = ["netsh", "wlan", "show", "network", "mode=Bssid"]
                    results = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
                elif 'linux' in system:
                    # Linux
                    cmd = ["iwlist", "scanning"]
                    results = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
                else:
                    return []
                
                mac_addresses = []
                lines = results.decode("utf-8", errors="ignore").strip().splitlines()
                
                # Parse MAC addresses (simplified - may need OS-specific parsing)
                mac_pattern = r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})'
                for line in lines:
                    # Look for MAC address pattern (XX:XX:XX:XX:XX:XX)
                    matches = re.findall(mac_pattern, line)
                    if matches:
                        mac = ''.join(matches[0][:-1]) + matches[0][-1]
                        if mac not in mac_addresses:
                            mac_addresses.append(mac)
                
                return mac_addresses[:10]  # Limit to 10 for API
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
                self.logger.warning(f"Error getting WiFi MAC addresses: {e}")
                return []

        wifi_mac_addresses = get_wifi_mac_addresses()
        if not wifi_mac_addresses:
            return None, None

        wifi_data = [{"macAddress": mac} for mac in wifi_mac_addresses]
        url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={self.google_api_key}"

        try:
            response = requests.post(url, json={"wifiAccessPoints": wifi_data}, timeout=5)
            response.raise_for_status()
            response_data = response.json()

            if 'location' in response_data:
                latitude = response_data['location']['lat']
                longitude = response_data['location']['lng']
                return latitude, longitude
        except requests.RequestException as e:
            self.logger.warning(f"Error getting location: {e}")

        return None, None

    def update_video(self):
        """Update the video frame with object detection."""
        if not self.video_on or self.cap is None:
            return
        
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if frame_time > 0:
            fps = 1.0 / frame_time
            self.fps_history.append(fps)
            self.stats['fps'] = sum(self.fps_history) / len(self.fps_history)
        
        # Capture a frame from the camera or video
        ret, frame = self.cap.read()
        if not ret:
            if self.video_source_type == "file":
                # Video file ended
                self.logger.info("Video file ended")
                self.stop_video()
                messagebox.showinfo("Video Complete", "Video processing completed!")
                return
            else:
                self.logger.warning("Failed to read frame from camera")
                self.status_label.config(text="● Camera Error", fg=self.COLOR_DANGER)
                self._schedule_next_update()
                return
        
        self.stats['frames_processed'] += 1
        
        try:
            # Perform object detection on the frame
            confidence = self.confidence_var.get()
            results = self.model.predict(
                source=frame, 
                conf=confidence, 
                save=False, 
                verbose=False,
                imgsz=320  # Match training size for better performance
            )
            
            pothole_detected = False
            current_detections = []
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                # Check if any potholes are detected
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        if cls_id == self.POTHOLE_CLASS_ID:
                            pothole_detected = True
                            bb = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf.item())
                            current_detections.append({
                                'bbox': bb,
                                'confidence': conf,
                                'class_id': cls_id
                            })
                    
                    # Update statistics
                    if pothole_detected:
                        self.stats['total_detections'] += 1
                        # Save if meets criteria (debouncing)
                        self._handle_detection_saving(frame, current_detections)
            
            # Draw detections using YOLO's built-in plotting
            annotated_frame = results[0].plot() if results and len(results) > 0 else frame
            
            # Add FPS to frame
            fps_text = f"FPS: {self.stats['fps']:.1f}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add detection count
            if pothole_detected:
                det_text = f"Potholes: {len(current_detections)}"
                cv2.putText(
                    annotated_frame,
                    det_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
        except Exception as e:
            self.logger.error(f"Error during detection: {e}", exc_info=True)
            annotated_frame = frame

        # Convert the OpenCV frame to a PIL Image
        image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        # Display the updated video frame
        self.show_video_frame(image)

        # Update statistics display periodically
        if self.stats['frames_processed'] % 30 == 0:  # Update every 30 frames
            self._update_statistics_display()

        # Schedule the next update
        self._schedule_next_update()
    
    def _schedule_next_update(self):
        """Schedule the next video frame update."""
        interval = self.config.get("update_interval_ms", 33)
        self.update_id = self.root.after(interval, self.update_video)
    
    def _handle_detection_saving(self, frame: np.ndarray, detections: List[Dict]):
        """Handle saving detections with debouncing to prevent duplicates."""
        current_time = datetime.now()
        
        # Check if we should save (debouncing)
        should_save = False
        if self.last_save_time is None:
            should_save = True
        else:
            time_since_last_save = (current_time - self.last_save_time).total_seconds()
            if time_since_last_save >= self.MIN_SAVE_INTERVAL_SECONDS:
                # Check if detection is significantly different
                if self._is_new_detection(detections):
                    should_save = True
        
        if should_save:
            with self.save_lock:
                self._save_detection(frame, detections)
                self.last_save_time = current_time
                self.last_detection_boxes = detections
    
    def _is_new_detection(self, current_detections: List[Dict]) -> bool:
        """Check if current detections are significantly different from last saved ones."""
        if not self.last_detection_boxes:
            return True
        
        if len(current_detections) != len(self.last_detection_boxes):
            return True
        
        # Simple overlap check (can be improved with IoU)
        threshold = 0.3  # 30% overlap threshold
        for curr in current_detections:
            curr_bbox = curr['bbox']
            found_match = False
            for prev in self.last_detection_boxes:
                prev_bbox = prev['bbox']
                # Calculate IoU-like overlap
                overlap = self._calculate_bbox_overlap(curr_bbox, prev_bbox)
                if overlap > threshold:
                    found_match = True
                    break
            if not found_match:
                return True
        
        return False
    
    def _calculate_bbox_overlap(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _update_statistics_display(self):
        """Update statistics labels in the GUI."""
        self.detection_label.config(text=f"🔍 Detections: {self.stats['total_detections']}")
        self.saved_label.config(text=f"💾 Saved: {self.stats['total_saved']}")
        
        # Calculate session duration
        session_duration = datetime.now() - self.stats['session_start_time']
        hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.session_label.config(text=f"⏱️ Session: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update FPS label
        self.fps_label.config(text=f"FPS: {self.stats['fps']:.1f}")

    def _save_detection(self, frame, detections: List[Dict]):
        """Save detected pothole image and coordinates."""
        try:
            # Save the annotated frame
            save_path = self.save_dir / f"pothole_{self.sequence_number}.jpg"
            success = cv2.imwrite(str(save_path), frame)
            
            if not success:
                self.logger.error(f"Failed to save image to {save_path}")
                return

            # Fetch location and save coordinates
            latitude, longitude = self.get_wifi_location()
            coordinates_file = self.coordinates_dir / f"coordinates_{self.sequence_number}.txt"
            
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%Y-%m-%d")
            
            with open(coordinates_file, "w", encoding="utf-8") as coord_file:
                coord_file.write(f"Date: {current_date} Time: {current_time}\n")
                coord_file.write(f"Location: {latitude},{longitude}\n")
                coord_file.write(f"Image: {save_path}\n")
                coord_file.write(f"Detection Count: {len(detections)}\n")
                
                for i, det in enumerate(detections, 1):
                    bbox = det['bbox']
                    conf = det['confidence']
                    coord_file.write(f"\nDetection {i}:\n")
                    coord_file.write(f"  Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]\n")
                    coord_file.write(f"  Confidence: {conf:.3f}\n")

            # Increment sequence number and update stats
            self.sequence_number += 1
            self.stats['total_saved'] += 1
            
            self.logger.info(f"Saved detection #{self.sequence_number - 1} to {save_path}")
            
            # Update display immediately
            self._update_statistics_display()
            
        except Exception as e:
            self.logger.error(f"Error saving detection: {e}", exc_info=True)

    def show_video_frame(self, image: Image.Image):
        """Display the video frame in the GUI."""
        # Resize image to fit the label size while maintaining aspect ratio
        display_width, display_height = self.MAX_DISPLAY_SIZE
        img_width, img_height = image.size
        
        # Calculate scaling to fit while maintaining aspect ratio
        scale_w = display_width / img_width
        scale_h = display_height / img_height
        scale = min(scale_w, scale_h)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert PIL Image to Tkinter PhotoImage
        imgtk = ImageTk.PhotoImage(image)

        # Update the video label with the new frame
        self.video_label.configure(image=imgtk, text="")
        self.video_label.image = imgtk  # Keep a reference to prevent garbage collection

    def on_closing(self):
        """Handle window closing event."""
        self.logger.info("Closing application...")
        self.stop_video()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.logger.info("Application closed")
        self.root.destroy()
        
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        except Exception:
            pass  # Ignore errors during cleanup


if __name__ == "__main__":
    root = tk.Tk()
    try:
        logger.info("Starting Pothole Detection Application")
        app = PotholeDetection(root, config_path="config.json")
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("Application interrupted by user")
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Application Error", f"Failed to start application:\n{e}")
