# Pothole Detection System

A real-time pothole detection application using YOLOv8 segmentation model with a Tkinter GUI interface. The system detects potholes from live camera feed, saves detected images, and records GPS coordinates using WiFi-based geolocation.

## Features

- Real-time pothole detection using YOLOv8 segmentation model
- Live video feed from webcam
- Automatic saving of detected pothole images
- GPS coordinate recording using Google Geolocation API
- Cross-platform support (Windows, macOS, Linux)
- Configurable detection parameters

## Requirements

- Python 3.8 or higher
- Webcam/camera
- Google Maps API key (optional, for location services)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd pothole_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the application:
   - Copy `config.json.example` to `config.json`
   - Edit `config.json` with your settings:
     - Set your Google Maps API key (optional)
     - Adjust model path if using a custom trained model
     - Configure save directories

4. Ensure you have the model file:
   - Place `yolov8n-seg.pt` in the project directory, or
   - Update `model_path` in `config.json` to point to your model file

## Usage

Run the application:
```bash
python main.py
```

### Controls

- **Start Video**: Begin live detection from webcam
- **Stop Video**: Stop the video feed
- Status indicator shows current application state

### Output

- Detected pothole images are saved to `ResultPrediction/` directory
- GPS coordinates and metadata are saved to `ResultLocation/` directory
- Files are named sequentially: `pothole_1.jpg`, `pothole_2.jpg`, etc.

## Configuration

Edit `config.json` to customize:

- `model_path`: Path to YOLOv8 model file
- `class_list_path`: Path to class names file (coco.txt)
- `save_dir`: Directory for saving detected images
- `coordinates_dir`: Directory for saving location data
- `google_api_key`: Your Google Maps Geolocation API key
- `confidence_threshold`: Detection confidence threshold (0.0-1.0)
- `camera_index`: Camera device index (usually 0)
- `update_interval_ms`: Frame update interval in milliseconds

## Model Training

See `model.ipynb` for model training and evaluation code. The notebook includes:
- Dataset preparation
- Model training
- Performance metrics analysis
- Validation and testing

## Project Structure

```
pothole_detection/
├── main.py                 # Main application
├── model.ipynb            # Training notebook
├── config.json.example     # Configuration template
├── coco.txt               # Class names
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── ResultPrediction/      # Output directory (created automatically)
└── ResultLocation/        # Coordinates directory (created automatically)
```

## Notes

- The application requires a trained YOLOv8 segmentation model
- WiFi-based geolocation requires nearby WiFi access points
- Google Maps API key is optional but recommended for accurate location
- Camera permissions may be required on some systems

## Troubleshooting

**Camera not working:**
- Check camera permissions
- Try different `camera_index` values in config.json
- Ensure no other application is using the camera

**Model not loading:**
- Verify model file path in config.json
- Ensure model file exists and is accessible

**Location not working:**
- Check Google API key in config.json
- Ensure WiFi is enabled
- Check internet connection

## License

[Add your license here]

## Author

[Add your name/contact here]
