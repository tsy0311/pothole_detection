import tkinter as tk
from tkinter import messagebox
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
    MAX_DISPLAY_SIZE = (640, 480)
    DEFAULT_CONFIDENCE = 0.5
    
    def __init__(self, root, config_path: Optional[str] = None):
        """
        Initialize the Pothole Detection application.
        
        Args:
            root: Tkinter root window
            config_path: Optional path to configuration JSON file
        """
        self.root = root
        self.root.title("Pothole Detection System")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize paths
        self.model_path = self.config.get("model_path", "yolov10n-seg.pt")
        self.class_list_path = self.config.get("class_list_path", "coco.txt")
        self.save_dir = Path(self.config.get("save_dir", "ResultPrediction"))
        self.coordinates_dir = Path(self.config.get("coordinates_dir", "ResultLocation"))
        self.google_api_key = self.config.get("google_api_key", "")
        
        # Create directories if they don't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.coordinates_dir.mkdir(parents=True, exist_ok=True)

        # Initialize YOLO model
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n{e}")
            raise

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

        # Create GUI elements
        self._create_gui()

        # Set up video capture
        self.cap = None
        self.video_on = False
        self.update_id = None  # Track scheduled update ID

        # Initialize sequence number for naming cropped pothole images
        self.sequence_number = self._get_next_sequence_number()

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from JSON file or use defaults."""
        default_config = {
            "model_path": "yolov10n-seg.pt",
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
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a title label
        self.title_label = tk.Label(
            main_container, 
            text="Pothole Detection System", 
            font=("Helvetica", 18, "bold")
        )
        self.title_label.pack(pady=(0, 10))

        # Create a frame to contain video feed and control panel
        content_frame = tk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Video display frame
        video_frame = tk.Frame(content_frame)
        video_frame.pack(side=tk.LEFT, padx=10)

        # Create a label to display the video feed
        self.video_label = tk.Label(video_frame, bg="black", relief=tk.SUNKEN, bd=2)
        self.video_label.pack()

        # FPS label
        self.fps_label = tk.Label(
            video_frame,
            text="FPS: 0.0",
            font=("Helvetica", 9),
            fg="green"
        )
        self.fps_label.pack(pady=(5, 0))

        # Control panel frame
        control_frame = tk.Frame(content_frame, relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

        # Control panel title
        control_title = tk.Label(
            control_frame,
            text="Controls",
            font=("Helvetica", 12, "bold")
        )
        control_title.pack(pady=10)

        # Start/Stop button
        self.start_stop_button = tk.Button(
            control_frame, 
            text="Start Video", 
            command=self.toggle_video,
            width=20,
            height=2,
            font=("Helvetica", 10),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049"
        )
        self.start_stop_button.pack(pady=10, padx=10)

        # Status label
        status_frame = tk.Frame(control_frame)
        status_frame.pack(pady=5, padx=10)
        
        tk.Label(status_frame, text="Status:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.status_label = tk.Label(
            status_frame,
            text="Stopped",
            font=("Helvetica", 10),
            fg="red"
        )
        self.status_label.pack(anchor=tk.W, pady=(2, 0))

        # Statistics panel
        stats_title = tk.Label(
            control_frame,
            text="Statistics",
            font=("Helvetica", 12, "bold")
        )
        stats_title.pack(pady=(20, 5))

        stats_frame = tk.Frame(control_frame)
        stats_frame.pack(pady=5, padx=10, fill=tk.X)

        # Detection count
        self.detection_label = tk.Label(
            stats_frame,
            text="Detections: 0",
            font=("Helvetica", 9),
            anchor=tk.W
        )
        self.detection_label.pack(fill=tk.X, pady=2)

        # Saved count
        self.saved_label = tk.Label(
            stats_frame,
            text="Saved: 0",
            font=("Helvetica", 9),
            anchor=tk.W
        )
        self.saved_label.pack(fill=tk.X, pady=2)

        # Session time
        self.session_label = tk.Label(
            stats_frame,
            text="Session: 0:00:00",
            font=("Helvetica", 9),
            anchor=tk.W
        )
        self.session_label.pack(fill=tk.X, pady=2)

        # Confidence threshold control
        conf_frame = tk.Frame(control_frame)
        conf_frame.pack(pady=(15, 5), padx=10, fill=tk.X)
        
        tk.Label(conf_frame, text="Confidence:", font=("Helvetica", 9)).pack(anchor=tk.W)
        
        self.confidence_var = tk.DoubleVar(value=self.config.get("confidence_threshold", 0.5))
        self.confidence_scale = tk.Scale(
            conf_frame,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var,
            command=self._on_confidence_change
        )
        self.confidence_scale.pack(fill=tk.X, pady=2)
        
        self.confidence_value_label = tk.Label(
            conf_frame,
            text=f"{self.confidence_var.get():.2f}",
            font=("Helvetica", 8)
        )
        self.confidence_value_label.pack()

    def _on_confidence_change(self, value):
        """Update confidence threshold label."""
        self.confidence_value_label.config(text=f"{float(value):.2f}")

    def toggle_video(self):
        """Toggle video feed on/off."""
        if self.video_on:
            self.stop_video()
        else:
            self.start_video()

    def start_video(self):
        """Start the video feed."""
        if self.cap is None:
            camera_index = self.config.get("camera_index", 0)
            self.logger.info(f"Opening camera {camera_index}")
            self.cap = cv2.VideoCapture(camera_index)
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                error_msg = f"Failed to open camera {camera_index}"
                self.logger.error(error_msg)
                self.status_label.config(text="Camera Error", fg="red")
                messagebox.showerror("Camera Error", error_msg)
                return
        
        self.video_on = True
        self.start_stop_button.configure(text="Stop Video", bg="#f44336", activebackground="#da190b")
        self.status_label.config(text="Running", fg="green")
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
        self.start_stop_button.configure(text="Start Video", bg="#4CAF50", activebackground="#45a049")
        self.status_label.config(text="Stopped", fg="red")
        self.logger.info("Video feed stopped")
        
        # Update statistics
        self._update_statistics_display()

    def get_wifi_location(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get location using WiFi access points and Google Geolocation API.
        
        Returns:
            Tuple of (latitude, longitude) or (None, None) if failed
        """
        if not self.google_api_key or self.google_api_key == "Enter Your Google API":
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
        
        # Capture a frame from the camera
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to read frame from camera")
            self.status_label.config(text="Camera Error", fg="red")
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
        self.detection_label.config(text=f"Detections: {self.stats['total_detections']}")
        self.saved_label.config(text=f"Saved: {self.stats['total_saved']}")
        
        # Calculate session duration
        session_duration = datetime.now() - self.stats['session_start_time']
        hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.session_label.config(text=f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
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
        self.video_label.configure(image=imgtk)
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
