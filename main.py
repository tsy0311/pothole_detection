import tkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
import requests
import platform
import subprocess
import os
import json
import re
from pathlib import Path
from typing import Optional, Tuple


class PotholeDetection:
    """Pothole Detection Application using YOLOv8 and Tkinter GUI."""
    
    def __init__(self, root, config_path: Optional[str] = None):
        """
        Initialize the Pothole Detection application.
        
        Args:
            root: Tkinter root window
            config_path: Optional path to configuration JSON file
        """
        self.root = root
        self.root.title("Pothole Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize paths
        self.model_path = self.config.get("model_path", "yolov8n-seg.pt")
        self.class_list_path = self.config.get("class_list_path", "coco.txt")
        self.save_dir = Path(self.config.get("save_dir", "ResultPrediction"))
        self.coordinates_dir = Path(self.config.get("coordinates_dir", "ResultLocation"))
        self.google_api_key = self.config.get("google_api_key", "")
        
        # Create directories if they don't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.coordinates_dir.mkdir(parents=True, exist_ok=True)

        # Initialize YOLO model
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Load class list
        try:
            with open(self.class_list_path, "r", encoding="utf-8") as f:
                self.class_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Warning: Class list file not found at {self.class_list_path}")
            self.class_list = ["pothole"]  # Default fallback

        # Create GUI elements
        self._create_gui()

        # Set up video capture
        self.cap = None
        self.video_on = False
        self.update_id = None  # Track scheduled update ID

        # Initialize sequence number for naming cropped pothole images
        self.sequence_number = 1

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from JSON file or use defaults."""
        default_config = {
            "model_path": "yolov8n-seg.pt",
            "class_list_path": "coco.txt",
            "save_dir": "ResultPrediction",
            "coordinates_dir": "ResultLocation",
            "google_api_key": "",
            "confidence_threshold": 0.7,
            "camera_index": 0,
            "update_interval_ms": 33  # ~30 FPS
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        return default_config

    def _create_gui(self):
        """Create the GUI elements."""
        # Create a title label
        self.title_label = tk.Label(
            self.root, 
            text="Pothole Detection System", 
            font=("Helvetica", 16, "bold")
        )
        self.title_label.pack(pady=10)

        # Create a frame to contain video feed and buttons
        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=10)

        # Create a label to display the video feed
        self.video_label = tk.Label(self.frame, bg="black")
        self.video_label.pack(side=tk.LEFT, padx=10)

        # Create a button frame
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack(side=tk.RIGHT, padx=10)

        # Create a button to start/stop the video feed
        self.start_stop_button = tk.Button(
            self.button_frame, 
            text="Start Video", 
            command=self.toggle_video,
            width=15,
            height=2
        )
        self.start_stop_button.pack(pady=5)

        # Status label
        self.status_label = tk.Label(
            self.button_frame,
            text="Status: Stopped",
            font=("Helvetica", 10)
        )
        self.status_label.pack(pady=5)

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
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                self.status_label.config(text="Status: Camera Error", fg="red")
                return
        
        self.video_on = True
        self.start_stop_button.configure(text="Stop Video")
        self.status_label.config(text="Status: Running", fg="green")
        self.update_video()

    def stop_video(self):
        """Stop the video feed."""
        self.video_on = False
        if self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None
        self.start_stop_button.configure(text="Start Video")
        self.status_label.config(text="Status: Stopped", fg="black")

    def get_wifi_location(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get location using WiFi access points and Google Geolocation API.
        
        Returns:
            Tuple of (latitude, longitude) or (None, None) if failed
        """
        if not self.google_api_key or self.google_api_key == "Enter Your Google API":
            print("Warning: Google API key not configured")
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
                print(f"Error getting WiFi MAC addresses: {e}")
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
            print(f"Error getting location: {e}")

        return None, None

    def update_video(self):
        """Update the video frame with object detection."""
        if not self.video_on or self.cap is None:
            return
        
        # Capture a frame from the camera
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Status: Camera Error", fg="red")
            return
        
        try:
            # Perform object detection on the frame
            confidence = self.config.get("confidence_threshold", 0.7)
            results = self.model.predict(source=frame, conf=confidence, save=False, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                # Check if any potholes are detected (class 0)
                pothole_detected = False
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        if cls_id == 0:  # Assuming pothole is class 0
                            pothole_detected = True
                            break
                
                # Draw detections and save if pothole detected
                if pothole_detected and boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        if cls_id == 0:  # Pothole class
                            bb = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf.item())
                            
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                            
                            # Draw rectangle on the frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            # Add label and confidence score
                            label = self.class_list[cls_id] if cls_id < len(self.class_list) else "pothole"
                            label_text = f"{label} {conf * 100:.1f}%"
                            cv2.putText(
                                frame, 
                                label_text, 
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 255), 
                                2
                            )
                    
                    # Save frame and location when pothole is detected
                    self._save_detection(frame)
            
            # Draw detections using YOLO's built-in plotting
            annotated_frame = results[0].plot() if results else frame
            
        except Exception as e:
            print(f"Error during detection: {e}")
            annotated_frame = frame

        # Convert the OpenCV frame to a PIL Image
        image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        # Display the updated video frame
        self.show_video_frame(image)

        # Schedule the next update
        interval = self.config.get("update_interval_ms", 33)
        self.update_id = self.root.after(interval, self.update_video)

    def _save_detection(self, frame):
        """Save detected pothole image and coordinates."""
        try:
            # Save the annotated frame
            save_path = self.save_dir / f"pothole_{self.sequence_number}.jpg"
            cv2.imwrite(str(save_path), frame)

            # Fetch location and save coordinates
            latitude, longitude = self.get_wifi_location()
            coordinates_file = self.coordinates_dir / f"coordinates_{self.sequence_number}.txt"
            
            with open(coordinates_file, "w", encoding="utf-8") as coord_file:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                current_date = now.strftime("%Y-%m-%d")
                coord_file.write(f"Date: {current_date} Time: {current_time}\n")
                coord_file.write(f"Location: {latitude},{longitude}\n")
                coord_file.write(f"Image: {save_path}\n")

            # Increment sequence number
            self.sequence_number += 1
            print(f"Saved detection #{self.sequence_number - 1}")
        except Exception as e:
            print(f"Error saving detection: {e}")

    def show_video_frame(self, image):
        """Display the video frame in the GUI."""
        # Resize image to fit the label size
        width, height = 640, 480
        image = image.resize((width, height), Image.Resampling.LANCZOS)

        # Convert PIL Image to Tkinter PhotoImage
        imgtk = ImageTk.PhotoImage(image)

        # Update the video label with the new frame
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk  # Keep a reference to prevent garbage collection

    def on_closing(self):
        """Handle window closing event."""
        self.stop_video()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
        
    def __del__(self):
        """Cleanup resources."""
        if self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = PotholeDetection(root, config_path="config.json")
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
