#!/usr/bin/env python3
"""
Quick Training Launcher - Get training results fast!
Automatically sets up datasets and starts training
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json

def run_command(cmd, description="Running command"):
    """Run command with progress indication"""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ Error!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'numpy', 
        'matplotlib', 'tqdm', 'scikit-learn', 'albumentations'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        
        install_cmd = f"pip install {' '.join(missing_packages)}"
        print(f"\nInstall with: {install_cmd}")
        
        if input("\nInstall now? (y/n): ").lower() == 'y':
            return run_command(install_cmd, "Installing dependencies")
        else:
            return False
    
    print("✅ All dependencies installed")
    return True

def setup_synthetic_dataset(num_samples=1000):
    """Set up synthetic dataset for quick training"""
    print(f"🎨 Setting up synthetic dataset with {num_samples} samples...")
    
    cmd = f"python scripts/prepare_datasets.py --synthetic --synthetic-samples {num_samples}"
    return run_command(cmd, "Creating synthetic dataset")

def quick_train_classification(epochs=10, batch_size=32):
    """Quick classification training for immediate results"""
    print(f"🚀 Starting quick classification training ({epochs} epochs)...")
    
    # Create a temporary config for quick training
    quick_config = {
        "model_type": "mini",
        "task": "classification", 
        "num_classes": 2,
        "image_size": [224, 224],
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "epochs": epochs,
        "early_stopping_patience": 5,
        "save_checkpoints": True,
        "mixed_precision": True
    }
    
    # Save quick config
    config_path = "config/quick_train_config.yaml"
    os.makedirs("config", exist_ok=True)
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(quick_config, f)
    
    # Run training
    cmd = f"python src/train.py --data-dir data/synthetic --config {config_path} --quick-mode"
    return run_command(cmd, "Training classification model")

def test_model():
    """Test the trained model"""
    print("🧪 Testing trained model...")
    
    # Check if model exists
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print("❌ No trained model found")
        return False
    
    # Create test command
    cmd = f"python src/inference.py --model {model_path} --test-mode --data-dir data/synthetic"
    return run_command(cmd, "Testing model performance")

def main():
    parser = argparse.ArgumentParser(description='Quick Training Launcher')
    parser.add_argument('--mode', choices=['full', 'quick', 'setup-only'], default='quick',
                       help='Training mode')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset creation')
    
    args = parser.parse_args()
    
    print("🔥 POTHOLE DETECTION - QUICK TRAINING LAUNCHER 🔥")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            print("❌ Dependency check failed. Please install missing packages.")
            return 1
    
    # Step 2: Setup dataset
    if not args.skip_dataset:
        if not os.path.exists("data/synthetic") or len(os.listdir("data/synthetic")) < 3:
            if not setup_synthetic_dataset(args.samples):
                print("❌ Dataset setup failed")
                return 1
        else:
            print("✅ Synthetic dataset already exists")
    
    if args.mode == 'setup-only':
        print("✅ Setup complete! Ready for training.")
        return 0
    
    # Step 3: Training
    if args.mode in ['quick', 'full']:
        epochs = args.epochs if args.mode == 'quick' else 30
        
        if not quick_train_classification(epochs, args.batch_size):
            print("❌ Training failed")
            return 1
    
    # Step 4: Test model
    if not test_model():
        print("❌ Model testing failed")
        return 1
    
    # Success summary
    print("\n" + "🎉" * 20)
    print("SUCCESS! Training completed successfully!")
    print("🎉" * 20)
    print("\nNext steps:")
    print("1. Check results in logs/ directory")
    print("2. View training plots in logs/plots/")
    print("3. Test model: python src/inference.py --model models/best_model.pth")
    print("4. Collect real pothole images for better training")
    print("5. Set up hardware when ready")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())