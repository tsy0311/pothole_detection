# Pothole Detection System - Complete Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Dataset Information](#dataset-information)
4. [Installation & Setup](#installation--setup)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Real-Time Detection Application](#real-time-detection-application)
8. [Project Structure](#project-structure)
9. [Results & Performance Metrics](#results--performance-metrics)
10. [Technical Details](#technical-details)
11. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project implements a **Pothole Detection System** using **YOLOv10** (You Only Look Once version 10), a state-of-the-art object detection and segmentation model. The system can:

- **Train a custom YOLOv10 model** on a pothole dataset
- **Detect potholes in real-time** from live camera feed or video files
- **Segment pothole regions** with pixel-level accuracy
- **Save detected images** with GPS coordinates (optional)
- **Evaluate model performance** with comprehensive metrics

### Key Features
- ✅ Real-time pothole detection using YOLOv10 segmentation model
- ✅ Transfer learning from COCO pre-trained weights
- ✅ Comprehensive model evaluation (mAP, Precision, Recall, F1-Score)
- ✅ Advanced features: TTA, ensemble prediction, hyperparameter optimization
- ✅ GUI application for live detection
- ✅ Video processing capabilities with instant preview
- ✅ Automatic result visualization and analysis
- ✅ Well-documented training notebook with markdown titles
- ✅ Fixed YOLOv10 compatibility (removed unsupported augment parameter)

---

## 🏗️ System Architecture

### Model Architecture
- **Base Model**: YOLOv10n (nano variant) - lightweight and fast
- **Task Type**: Object Detection + Instance Segmentation
- **Input Size**: 320x320 pixels (configurable)
- **Number of Classes**: 1 (Pothole)
- **Pre-trained Weights**: COCO dataset (80 classes)

### Training Approach
1. **Transfer Learning**: Start with COCO pre-trained weights
2. **Fine-tuning**: Adapt to pothole detection task
3. **Data Augmentation**: Applied during training for robustness
4. **Validation**: Continuous validation during training

### Detection Pipeline
```
Input Image/Video → YOLOv10 Model → Bounding Boxes + Masks → Post-processing → Results
```

---

## 📊 Dataset Information

### Dataset Source
- **Source**: Roboflow Universe (Pothole Segmentation YOLOv8 dataset)
- **License**: CC BY 4.0
- **Total Images**: 780 images
- **Format**: YOLO format annotations

### Dataset Structure
```
dataset/
├── data.yaml              # Dataset configuration
├── train/
│   ├── images/           # 720 training images
│   └── labels/           # 720 training annotations (.txt)
├── valid/
│   ├── images/           # 60 validation images
│   └── labels/           # 60 validation annotations (.txt)
└── sample_video.mp4      # Test video file
```

### Dataset Configuration (`data.yaml`)
```yaml
names:
- Pothole
nc: 1
train: train/images
val: valid/images
```

### Pre-processing Applied
- Auto-orientation of pixel data (EXIF-orientation stripping)
- Resize to 640x640 (Stretch)

### Augmentation Applied (during training)
- 50% probability of horizontal flip
- Random crop (0-20%)
- Random rotation (-15° to +15°)
- Random shear (-5° to +5°)
- Brightness adjustment (-25% to +25%)
- Exposure adjustment (-25% to +25%)

---

## 🚀 Installation & Setup

### Prerequisites
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **GPU**: Optional but recommended for training (CUDA-compatible)
- **RAM**: Minimum 8GB (16GB+ recommended for training)
- **Storage**: ~5GB for dataset and models

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd pothole_detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `ultralytics>=8.3.233` - YOLOv10 framework
- `torch>=1.8.0` - PyTorch deep learning framework
- `opencv-python>=4.12.0` - Image/video processing
- `matplotlib>=3.10.7` - Visualization
- `pandas>=2.3.3` - Data analysis
- `seaborn>=0.13.2` - Statistical visualization

### Step 3: Download Pre-trained Model (Optional)
The training notebook will automatically download `yolov10n.pt` if not present. For segmentation, use `yolov10n-seg.pt`.

### Step 4: Configure Application (Optional - for GUI)
```bash
# Copy configuration template
cp config.json.example config.json

# Edit config.json with your settings
# - Google Maps API key (optional, for GPS)
# - Model path
# - Camera settings
```

---

## 🎓 Model Training

### Training Notebook: `model.ipynb`

The training process is fully documented in the Jupyter notebook. Here's what it includes:

#### 1. **Dataset Preparation**
- Loads dataset from `dataset/` folder
- Validates dataset structure
- Configures data augmentation

#### 2. **Model Initialization**
- Loads YOLOv10n pre-trained weights
- Configures for segmentation task
- Sets up training parameters

#### 3. **Training Configuration**
```python
Training Parameters:
- Epochs: 100
- Image Size: 320x320
- Batch Size: 16
- Learning Rate: 0.01 (with scheduler)
- Optimizer: SGD
- Momentum: 0.937
- Weight Decay: 0.0005
```

#### 4. **Training Process**
- Real-time loss monitoring
- Validation after each epoch
- Automatic best model saving
- Training history visualization

#### 5. **Advanced Features**
- **Hyperparameter Optimization**: Confidence threshold tuning
- **Test-Time Augmentation (TTA)**: Improved inference accuracy
- **Model Ensemble**: Multiple model predictions combined
- **Robustness Testing**: Performance under various conditions

### Running Training

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook model.ipynb
   ```

2. **Run Cells Sequentially**:
   - Each code cell has a markdown title above it describing its purpose
   - Cell 0: Install dependencies
   - Cell 1: Import libraries
   - Cells 2-10: Dataset setup and validation
   - Cells 11-12: Model initialization
   - Cell 13: Start training
   - Cells 14+: Evaluation and analysis

**Note**: The notebook has been updated with descriptive markdown titles above each code cell for better organization and understanding.

3. **Monitor Training**:
   - Loss curves update in real-time
   - Validation metrics displayed after each epoch
   - Best model saved to `runs/segment/train/weights/best.pt`

### Training Output
- **Model Weights**: `runs/segment/train/weights/best.pt` (best model)
- **Training Logs**: `runs/segment/train/results.csv`
- **Visualizations**: Loss curves, validation images, confusion matrices

---

## 📈 Model Evaluation

### Evaluation Metrics

The notebook provides comprehensive evaluation including:

#### 1. **Bounding Box Metrics**
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all potholes
- **F1-Score**: Harmonic mean of Precision and Recall
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95

#### 2. **Segmentation Metrics** (if using segmentation model)
- Mask Precision, Recall, F1-Score
- Mask mAP@0.5 and mAP@0.5:0.95

#### 3. **Visualization Outputs**
- **Precision-Recall Curves**: Performance across confidence thresholds
- **Confusion Matrix**: Classification accuracy
- **Learning Curves**: Training/validation loss over epochs
- **Validation Images**: Model predictions on validation set

### Final Validation Results

All finalized validation graphs are saved to:
```
runs/detect/final_validation/
├── BoxP_curve.png          # Box Precision-Confidence curve
├── BoxR_curve.png          # Box Recall-Confidence curve
├── BoxF1_curve.png         # Box F1-Confidence curve
├── BoxPR_curve.png         # Box Precision-Recall curve
├── confusion_matrix.png     # Confusion matrix
└── confusion_matrix_normalized.png  # Normalized confusion matrix
```

### Interpreting Results

- **mAP@0.5 > 0.7**: Excellent performance
- **mAP@0.5 > 0.5**: Good performance
- **mAP@0.5 < 0.5**: May need more training or data

---

## 🖥️ Real-Time Detection Application

### GUI Application: `main.py`

A Tkinter-based GUI application for real-time pothole detection.

#### Features
- Live camera feed processing
- Video file processing with preview
- Real-time detection display with bounding boxes
- Automatic image saving on detection (with bounding boxes)
- Video preview when selecting files
- GPS coordinate recording (optional)
- Configurable confidence threshold
- FPS monitoring
- Improved error handling and validation

#### Running the Application

```bash
python main.py
```

#### Configuration (`config.json`)
```json
{
  "model_path": "runs/segment/train/weights/best.pt",
  "class_list_path": "coco.txt",
  "save_dir": "ResultPrediction",
  "coordinates_dir": "ResultLocation",
  "google_api_key": "YOUR_API_KEY",
  "confidence_threshold": 0.7,
  "camera_index": 0,
  "update_interval_ms": 33
}
```

#### Output
- **Detected Images**: `ResultPrediction/pothole_1.jpg`, `pothole_2.jpg`, ...
- **Location Data**: `ResultLocation/` (JSON files with GPS coordinates)

---

## 📁 Project Structure

```
pothole_detection/
│
├── 📄 main.py                    # GUI application for real-time detection
├── 📓 model.ipynb                # Complete training and evaluation notebook
├── ⚙️ config.json.example        # Configuration template
├── 📋 requirements.txt           # Python dependencies
├── 📖 README.md                  # This documentation
├── 📝 coco.txt                   # Class names file (used by GUI)
│
├── 📂 dataset/      # Dataset folder
│   ├── data.yaml                 # Dataset configuration
│   ├── sample_video.mp4          # Test video
│   ├── train/                    # Training data (720 images)
│   │   ├── images/
│   │   └── labels/
│   └── valid/                    # Validation data (60 images)
│       ├── images/
│       └── labels/
│
├── 📂 runs/                      # Training and evaluation outputs
│   ├── segment/
│   │   └── train/                # Training results
│   │       ├── weights/
│   │       │   ├── best.pt       # Best model weights ⭐
│   │       │   └── last.pt       # Last epoch weights
│   │       ├── results.csv       # Training metrics
│   │       └── *.png             # Training visualizations
│   │
│   └── detect/
│       ├── final_validation/     # Final validation results ⭐
│       ├── advanced_validation/  # Advanced evaluation results
│       ├── video_prediction/     # Video prediction outputs
│       └── threshold_optimization/ # Hyperparameter optimization results
│
├── 📂 ResultPrediction/          # GUI output: detected images (auto-created)
└── 📂 ResultLocation/            # GUI output: GPS coordinates (auto-created)
```

**⭐ Key Files/Folders:**
- `runs/segment/train/weights/best.pt` - **Trained model to use**
- `runs/detect/final_validation/` - **Final evaluation graphs**

---

## 📊 Results & Performance Metrics

### Understanding the Metrics

#### Mean Average Precision (mAP)
- **mAP@0.5**: Average precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Average precision across IoU 0.5 to 0.95 (more strict)

#### Precision vs Recall
- **High Precision**: Few false positives (detected potholes are real)
- **High Recall**: Few false negatives (finds most potholes)

#### F1-Score
- Balanced metric: `2 * (Precision * Recall) / (Precision + Recall)`
- Best when both Precision and Recall are high

### Typical Performance Ranges

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| mAP@0.5 | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| Precision | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| Recall | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| F1-Score | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |

---

## 🔧 Technical Details

### Model Specifications

**YOLOv10n (Nano)**
- Parameters: ~2.3M
- GFLOPs: ~6.5
- Speed: ~15-20ms per image (CPU), ~5-10ms (GPU)
- Input: 320x320 (configurable: 320, 640, 1280)

### Training Details

**Hardware Recommendations:**
- **CPU Training**: Possible but slow (hours to days)
- **GPU Training**: Recommended (NVIDIA GPU with CUDA)
- **Memory**: 8GB+ RAM, 4GB+ VRAM (GPU)

**Training Time Estimates:**
- 100 epochs on 720 images:
  - CPU: ~8-12 hours
  - GPU (RTX 3060): ~1-2 hours
  - GPU (RTX 4090): ~30-60 minutes

### Inference Performance

**Real-time Detection:**
- CPU: ~15-30 FPS (320x320 input)
- GPU: ~30-60 FPS (320x320 input)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **Model Training Fails**
**Problem**: Out of memory error
**Solution**: 
- Reduce batch size in training config
- Reduce image size (320 instead of 640)
- Close other applications

#### 2. **Low Detection Accuracy**
**Problem**: Model not detecting potholes well
**Solution**:
- Train for more epochs
- Check dataset quality and annotations
- Adjust confidence threshold
- Use data augmentation

#### 3. **Camera Not Working (GUI)**
**Problem**: Camera feed not showing
**Solution**:
- Check camera permissions
- Try different `camera_index` (0, 1, 2...)
- Ensure no other app is using camera
- Test camera with other applications

#### 4. **Model File Not Found**
**Problem**: `best.pt` not found
**Solution**:
- Ensure training completed successfully
- Check `runs/segment/train/weights/` folder
- Update `model_path` in `config.json`

#### 5. **Import Errors**
**Problem**: Module not found
**Solution**:
```bash
pip install -r requirements.txt
# Or reinstall specific package
pip install --upgrade ultralytics
```

#### 6. **CUDA/GPU Issues**
**Problem**: GPU not being used
**Solution**:
- Install CUDA-compatible PyTorch
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify CUDA version compatibility

---

## 📚 Additional Resources

### YOLOv10 Documentation
- Official Ultralytics Docs: https://docs.ultralytics.com/
- YOLOv10 Paper: [Reference if available]

### Dataset Source
- Roboflow Dataset: https://universe.roboflow.com/farzad/pothole_segmentation_yolov8

### Related Tools
- **Roboflow**: Dataset management and annotation
- **LabelImg**: Image annotation tool
- **TensorBoard**: Training visualization (optional)

---

## 📝 Notes for Professors/Reviewers

### Key Files to Review

1. **`model.ipynb`**: Complete training pipeline with evaluation
   - All cells are documented
   - Results are saved and visualized
   - Advanced features are implemented

2. **`runs/detect/final_validation/`**: Final model evaluation
   - All validation graphs
   - Confusion matrices
   - Performance metrics

3. **`runs/segment/train/`**: Training history
   - Loss curves
   - Training metrics over epochs
   - Best model weights

### Model Performance Summary

Check the notebook's final validation cell (Cell 36) for:
- Box mAP50, mAP50-95
- Precision, Recall, F1-Score
- Segmentation metrics (if applicable)

### Reproducibility

- All random seeds are set (if applicable)
- Dataset is fixed and included
- Training configuration is documented
- Results are saved automatically

---

## 👤 Author & Contact

[Add your name and contact information here]

---

## 📄 License

[Add license information here]

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv10 framework
- **Roboflow** for dataset hosting
- Dataset contributors on Roboflow Universe

---

**Last Updated**: December 2024
**Version**: 1.1

---

## 🔄 Recent Updates (v1.1)

### Application Improvements (`main.py`)
- ✅ **Video File Processing**: Fixed issue where selected video files weren't processing
- ✅ **Video Preview**: Added instant preview of first frame when selecting video files
- ✅ **Bounding Box Display**: Fixed saved images to include bounding boxes (previously missing)
- ✅ **Better Error Handling**: Improved validation and error messages for video files
- ✅ **Video Metadata Display**: Shows video resolution and duration in source label

### Notebook Improvements (`model.ipynb`)
- ✅ **Markdown Titles**: Added descriptive markdown titles above each code cell
- ✅ **YOLOv10 Compatibility**: Removed unsupported `augment=True` parameter (eliminated warnings)
- ✅ **Better Organization**: Improved notebook structure with 26+ new markdown cells

### Documentation
- ✅ **README Updates**: Added recent improvements and features
- ✅ **Testing Report**: Added `testing.md` with comprehensive research documentation
