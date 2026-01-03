"""
Quick test to verify YOLOv8 setup and basic functionality
"""
from ultralytics import YOLO
from pathlib import Path
import yaml

print("=" * 60)
print("YOLOv8 Setup Verification")
print("=" * 60)

# Test 1: Load YOLOv8 model
print("\n1. Testing YOLOv8 model loading...")
try:
    model = YOLO('yolov8n-seg.pt')
    print(f"   ✓ YOLOv8 model loaded successfully")
    print(f"   Model: {model.model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters())}")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    exit(1)

# Test 2: Check dataset configuration
print("\n2. Testing dataset configuration...")
dataset_path = Path('Pothole_Segmentation_YOLOv8')
yaml_path = dataset_path / 'data.yaml'
if yaml_path.exists():
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ Dataset YAML found")
    print(f"   Classes: {config.get('names', [])}")
    print(f"   Number of classes: {config.get('nc', 0)}")
    if config.get('nc') == 1 and config.get('names') == ['Pothole']:
        print("   ✓ Configuration correct (single class: Pothole)")
    else:
        print("   ⚠ Configuration may need adjustment")
else:
    print(f"   ✗ Dataset YAML not found at {yaml_path}")

# Test 3: Verify dataset structure
print("\n3. Testing dataset structure...")
train_images = dataset_path / 'train' / 'images'
val_images = dataset_path / 'valid' / 'images'
if train_images.exists():
    train_count = len(list(train_images.glob('*.jpg')))
    print(f"   ✓ Training images found: {train_count}")
else:
    print(f"   ✗ Training images directory not found")
if val_images.exists():
    val_count = len(list(val_images.glob('*.jpg')))
    print(f"   ✓ Validation images found: {val_count}")
else:
    print(f"   ✗ Validation images directory not found")

print("\n" + "=" * 60)
print("All basic checks completed!")
print("=" * 60)




