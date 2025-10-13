# Quick Training Guide - Get Results Fast! 🚀

## Step-by-Step Training Process (Before Hardware Setup)

### 1. Environment Setup
```bash
# Install required packages
pip install torch torchvision opencv-python albumentations scikit-learn matplotlib tqdm requests

# Navigate to project directory
cd "c:\Users\ABHISHEK ARUN RAJA\Documents\Coding Projects\pothole-detection"
```

### 2. Dataset Options (Choose One)

#### Option A: Quick Start with Synthetic Data (Recommended for immediate results)
```bash
# Create 2000 synthetic images for immediate training
python scripts/prepare_datasets.py --synthetic --synthetic-samples 2000
```
**Pros:** Immediate training, no download needed
**Cons:** Not real-world data

#### Option B: Real Pothole Datasets (Better accuracy)

**Kaggle Annotated Potholes Dataset:**
1. Go to: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset
2. Download manually to `data/kaggle_pothole/`
3. Run: `python scripts/prepare_datasets.py --prepare-kaggle`

**Road Damage Dataset:**
1. Go to: https://github.com/sekilab/RoadDamageDetector
2. Download RDD2020 or RDD2022
3. Extract to `data/road_damage/`

#### Option C: Create Your Own Mini Dataset
```bash
# Create folders
mkdir data\custom\train\pothole
mkdir data\custom\train\no_pothole
mkdir data\custom\val\pothole
mkdir data\custom\val\no_pothole

# Add 20-50 images to each folder from:
# - Google Images (search "pothole road damage")
# - YouTube screenshots of pothole videos
# - Stock photo websites
```

### 3. Quick Training Commands

#### Train Classification Model (Fastest Results)
```bash
# Train lightweight model for immediate results
python src/train.py --config config/model_config.yaml --task classification --model mini --epochs 20 --batch-size 32 --lr 0.001
```

#### Train Full Detection Model (Better Performance)
```bash
# Train detection model (takes longer)
python src/train.py --config config/model_config.yaml --task detection --model pothole_net --epochs 50 --batch-size 16 --lr 0.0005
```

### 4. Monitor Training Progress

**Real-time monitoring:**
```bash
# Watch training logs
tail -f logs/training_*.log
```

**Check results:**
- Training plots saved to `logs/plots/`
- Model checkpoints in `models/checkpoints/`
- Best model saved as `models/best_model.pth`

### 5. Quick Testing

#### Test Trained Model
```bash
# Test on sample images
python src/inference.py --model models/best_model.pth --input data/test_images/ --output results/
```

#### Benchmark Performance
```bash
# Get model statistics
python -c "
from src.models.pothole_net import create_model, count_parameters
model = create_model('mini', num_classes=2, task='classification')
print(f'Parameters: {count_parameters(model):,}')
print(f'Model size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB')
"
```

## Expected Training Times & Results

### System Requirements
- **CPU Only:** Works but slow (30-60 min for synthetic dataset)
- **GPU (NVIDIA):** Recommended (5-15 min for synthetic dataset)
- **RAM:** 8GB+ recommended
- **Storage:** 2-5GB for datasets

### Expected Results

#### Synthetic Dataset (20 epochs)
- **Training Time:** 5-15 minutes
- **Accuracy:** 85-95% (synthetic data is easier)
- **Model Size:** 1-5 MB (depending on architecture)

#### Real Dataset (50 epochs)
- **Training Time:** 30-90 minutes
- **Accuracy:** 75-90% (real-world performance)
- **Model Size:** 5-20 MB

## Recommended Training Sequence

### Phase 1: Quick Proof of Concept (Today)
```bash
# 1. Create synthetic dataset
python scripts/prepare_datasets.py --synthetic --synthetic-samples 1000

# 2. Train mini model
python src/train.py --task classification --model mini --epochs 10 --quick-test

# 3. Test inference
python src/inference.py --model models/best_model.pth --test-mode
```

### Phase 2: Better Model (This Week)
```bash
# 1. Download real dataset
# Follow Kaggle instructions above

# 2. Train full model
python src/train.py --task detection --model pothole_net --epochs 30

# 3. Evaluate thoroughly
python src/evaluate.py --model models/best_model.pth --test-set data/test/
```

### Phase 3: Hardware Integration (Next Week)
```bash
# 1. Set up Raspberry Pi with camera
# Follow hardware setup guide

# 2. Test real-time inference
python src/inference.py --model models/best_model.pth --camera --realtime

# 3. Run ROS system
roslaunch pothole_detection full_system.launch
```

## Troubleshooting Training Issues

### Common Problems & Solutions

**Out of Memory Error:**
```bash
# Reduce batch size
python src/train.py --batch-size 8

# Use mini model
python src/train.py --model mini
```

**Training Too Slow:**
```bash
# Reduce image size
python src/train.py --image-size 128 128

# Use fewer epochs
python src/train.py --epochs 10
```

**Poor Accuracy:**
```bash
# Use real dataset instead of synthetic
# Increase training data
# Try different learning rate
python src/train.py --lr 0.0001

# Use data augmentation
python src/train.py --augment-heavy
```

### Performance Optimization
```bash
# Enable mixed precision (GPU only)
python src/train.py --mixed-precision

# Use multiple workers
python src/train.py --num-workers 4

# Save model for mobile deployment
python src/train.py --export-mobile
```

## Next Steps After Training

1. **Analyze Results:** Check accuracy, loss curves, confusion matrices
2. **Collect Real Data:** Take photos of local roads for better training
3. **Hardware Setup:** Install camera and GPS on Raspberry Pi
4. **Real-world Testing:** Test model on actual road conditions
5. **Iterative Improvement:** Retrain with better data

---

**Start with synthetic data for immediate results, then upgrade to real datasets for production use!**