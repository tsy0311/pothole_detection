import tkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
import requests
import platform
import subprocess

class detection:
    def __init__(self, root):
        self.root = root
        self.root.title("Pothole Detection")

        # Initialize YOLO model
        self.model = YOLO("path\\to\\your\\model\\weights\\v8n-80.pt")

        # Open the class list file using a relative path
        with open("path\\to\\your\\model\\coco.txt", "r") as f:
            self.class_list = f.read().split("\n")

        # Create a title label
        self.title_label = tk.Label(root, text="Pothole Detection System", font=("Helvetica", 16, "bold"))
        self.title_label.pack()

        # Create a frame to contain video feed and buttons
        self.frame = tk.Frame(root)
        self.frame.pack(pady=10)

        # Create a label to display the video feed
        self.video_label = tk.Label(self.frame)
        self.video_label.pack(side=tk.LEFT, padx=10)

        # Create a button to start/stop the video feed
        self.start_stop_button = tk.Button(self.frame, text="Start Video", command=self.toggle_video)
        self.start_stop_button.pack(side=tk.RIGHT, padx=10)

        # Set up video capture
        self.cap = cv2.VideoCapture(0)
        self.video_on = False  # Flag to keep track of whether video is running or stopped

        # Initialize sequence number for naming cropped pothole images
        self.sequence_number = 1

        # Set directory to save cropped pothole images
        self.save_dir = "path\\to\\your\\model\\ResultPrediction"  # Replace to your own directory

        # Set directory to save coordinates
        self.coordinates_dir = "path\\to\\your\\model\\ResultLocation"  # Replace to your own directoryzz

    def toggle_video(self):
        if self.video_on:
            self.stop_video()
            self.start_stop_button.configure(text="Start Video")
        else:
            self.start_video()
            self.start_stop_button.configure(text="Stop Video")

    def start_video(self):
        self.video_on = True
        self.update_video()

    def stop_video(self):
        self.video_on = False
        self.root.after_cancel(self.update_video)

    def get_wifi_location(self):
            # Function to get the MAC addresses of nearby WiFi networks
            def get_wifi_mac_addresses():
                try:
                    if 'darwin' in platform.system().lower():
                        results = subprocess.check_output(["/System/Library/PrivateFrameworks/Apple80211.framework/Resources/airport", "-s"])
                    elif 'win' in platform.system().lower():
                        results = subprocess.check_output(["netsh", "wlan", "show", "network"])
                    elif 'linux' in platform.system().lower():
                        results = subprocess.check_output(["iwlist", "scanning"])
                    else:
                        raise Exception("Unsupported operating system")
                    
                    mac_addresses = []
                    lines = results.decode("utf-8").strip().splitlines()[1:]  # Decode bytes to string and skip the header line
                    for line in lines:
                        fields = line.split()
                        mac_addresses.append(fields[0])  # Only store the MAC address
                    return mac_addresses
                except Exception as e:
                    return []

            wifi_mac_addresses = get_wifi_mac_addresses()
            wifi_data = [{"macAddress": mac} for mac in wifi_mac_addresses]

            google_maps_api_key = "Enter Your Google API"
            url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={google_maps_api_key}"

            response = requests.post(url, json={"wifiAccessPoints": wifi_data})
            response_data = response.json()

            if 'location' in response_data:
                latitude = response_data['location']['lat']
                longitude = response_data['location']['lng']
                return latitude, longitude
            else:
                return None, None

    def update_video(self):
        if not self.video_on:
            return
        
        # Capture a frame from the camera
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Perform object detection on the frame
        detect_params = self.model.predict(source=[frame], conf=0.7, save=False)

        # Check if any potholes are detected
        pothole_detected = False
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            if float(clsID) == 0.0:  
                pothole_detected = True
                break

        if pothole_detected:
            for i in range(len(detect_params[0])):
                box = detect_params[0].boxes[i]
                clsID = box.cls.numpy()[0]
                if float(clsID) == 0.0:  
                    bb = box.xyxy.numpy()[0]
                    conf = box.conf.numpy()[0]
                    label = self.class_list[int(clsID)]
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                    # Draw rectangle on the frame
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Add label and confidence score
                    frame = cv2.putText(frame, f"{label} {round(conf * 100, 2)}%", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Save the cropped pothole image
                    save_path = self.save_dir + "\\pothole_" + str(self.sequence_number) + ".jpg"
                    cv2.imwrite(save_path, frame)

                    # Fetch location and save coordinates to text file
                    latitude, longitude = self.get_wifi_location()
                    coordinates_file = self.coordinates_dir + "\\coordinates_" + str(self.sequence_number) + ".txt"
                    with open(coordinates_file, "w") as coord_file:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        current_date = now.strftime("%Y-%m-%d")
                        coord_file.write(f"Date: {current_date} Time: {current_time}\n")
                        coord_file.write(f"Location: {latitude},{longitude}")

                    # Increment sequence number for naming the next cropped pothole image
                    self.sequence_number += 1

        # Convert the OpenCV frame to a PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Display the updated video frame
        self.show_video_frame(image)

        # Schedule the next update after 1000 milliseconds
        self.root.after(1000, self.update_video)

    def show_video_frame(self, image):
        # Resize image to fit the label size
        width, height = 640, 480  # Specify desired width and height for the displayed frame
        image = image.resize((width, height),Image.Resampling.LANCZOS)

        # Convert PIL Image to Tkinter PhotoImage
        imgtk = ImageTk.PhotoImage(image)

        # Update the video label with the new frame
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk  # Keep a reference to prevent garbage collection

        # Necessary to force update the label
        self.video_label.update_idletasks()

        # Schedule the next update after 1000 milliseconds
        self.root.after(2000, self.update_video)
        
    def __del__(self):
        # Release any resources when the application is closed
        if self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = detection(root)
    root.mainloop()
