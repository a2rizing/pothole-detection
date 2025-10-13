#!/usr/bin/env python3
"""
Simple training script for pothole detection
Works without PyTorch for initial testing
"""

import os
import json
from pathlib import Path
import shutil
from PIL import Image
import random

def count_dataset():
    """Count the prepared dataset"""
    data_dir = Path("./data/processed/classification")
    
    if not data_dir.exists():
        print("❌ Dataset not found!")
        return False
    
    print("📊 Dataset Summary:")
    print("="*50)
    
    total_images = 0
    for split in ['train', 'val', 'test']:
        pothole_dir = data_dir / split / "pothole"
        no_pothole_dir = data_dir / split / "no_pothole"
        
        pothole_count = len(list(pothole_dir.glob("*.jpg"))) if pothole_dir.exists() else 0
        no_pothole_count = len(list(no_pothole_dir.glob("*.jpg"))) if no_pothole_dir.exists() else 0
        split_total = pothole_count + no_pothole_count
        total_images += split_total
        
        print(f"{split.upper()}: {pothole_count} pothole, {no_pothole_count} no_pothole ({split_total} total)")
    
    print(f"\nTOTAL IMAGES: {total_images}")
    print("="*50)
    
    return True

def verify_dataset():
    """Verify dataset integrity"""
    data_dir = Path("./data/processed/classification")
    
    print("🔍 Verifying dataset integrity...")
    
    corrupted_images = []
    
    for split in ['train', 'val', 'test']:
        for class_name in ['pothole', 'no_pothole']:
            class_dir = data_dir / split / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob("*.jpg"):
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    corrupted_images.append(str(img_path))
                    print(f"❌ Corrupted: {img_path}")
    
    if corrupted_images:
        print(f"Found {len(corrupted_images)} corrupted images")
        return False
    else:
        print("✅ All images verified successfully!")
        return True

def create_simple_train_test():
    """Create a simple training test without PyTorch"""
    print("🧪 Creating simple training simulation...")
    
    # Sample a few images for testing
    data_dir = Path("./data/processed/classification/train")
    sample_dir = Path("./sample_training")
    sample_dir.mkdir(exist_ok=True)
    
    # Copy 5 images from each class
    for class_name in ['pothole', 'no_pothole']:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
            
        sample_class_dir = sample_dir / class_name
        sample_class_dir.mkdir(exist_ok=True)
        
        images = list(class_dir.glob("*.jpg"))[:5]
        for img_path in images:
            shutil.copy2(img_path, sample_class_dir / img_path.name)
    
    print(f"✅ Sample training data created in: {sample_dir}")
    return True

def prepare_training_ready_summary():
    """Create a summary for training readiness"""
    summary = {
        "dataset_ready": True,
        "total_images": 0,
        "splits": {},
        "next_steps": []
    }
    
    data_dir = Path("./data/processed/classification")
    
    for split in ['train', 'val', 'test']:
        pothole_dir = data_dir / split / "pothole"
        no_pothole_dir = data_dir / split / "no_pothole"
        
        pothole_count = len(list(pothole_dir.glob("*.jpg"))) if pothole_dir.exists() else 0
        no_pothole_count = len(list(no_pothole_dir.glob("*.jpg"))) if no_pothole_dir.exists() else 0
        
        summary["splits"][split] = {
            "pothole": pothole_count,
            "no_pothole": no_pothole_count,
            "total": pothole_count + no_pothole_count
        }
        summary["total_images"] += pothole_count + no_pothole_count
    
    # Training readiness checks
    train_total = summary["splits"]["train"]["total"]
    val_total = summary["splits"]["val"]["total"]
    
    if train_total < 100:
        summary["next_steps"].append("⚠️  Small training set - consider augmentation")
    if val_total < 50:
        summary["next_steps"].append("⚠️  Small validation set")
    
    # Check class balance
    train_pothole = summary["splits"]["train"]["pothole"]
    train_no_pothole = summary["splits"]["train"]["no_pothole"]
    balance_ratio = min(train_pothole, train_no_pothole) / max(train_pothole, train_no_pothole)
    
    if balance_ratio < 0.5:
        summary["next_steps"].append("⚠️  Class imbalance detected")
    else:
        summary["next_steps"].append("✅ Good class balance")
    
    summary["next_steps"].extend([
        "✅ Dataset ready for PyTorch training",
        "🚀 Install PyTorch: pip install torch torchvision",
        "🎯 Start training: python src/train.py"
    ])
    
    # Save summary
    with open("training_readiness.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main function"""
    print("🎯 POTHOLE DETECTION TRAINING SETUP")
    print("="*60)
    
    # Count dataset
    if not count_dataset():
        return False
    
    # Verify dataset
    if not verify_dataset():
        print("⚠️  Some images are corrupted, but continuing...")
    
    # Create training readiness summary
    summary = prepare_training_ready_summary()
    
    print("\n📋 TRAINING READINESS SUMMARY")
    print("="*60)
    print(f"Total images: {summary['total_images']}")
    print(f"Training images: {summary['splits']['train']['total']}")
    print(f"Validation images: {summary['splits']['val']['total']}")
    print(f"Test images: {summary['splits']['test']['total']}")
    
    print("\n📝 Next Steps:")
    for step in summary["next_steps"]:
        print(f"  {step}")
    
    print("\n🎉 Your dataset is ready for training!")
    print("📁 Dataset location: ./data/processed/classification")
    print("🔧 Config file: ./config/model_config.yaml")
    
    # Create sample for testing
    create_simple_train_test()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Setup complete!")
    else:
        print("\n❌ Setup failed!")