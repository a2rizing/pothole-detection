#!/usr/bin/env python3
"""
Download and organize Kaggle annotated potholes dataset for training
"""

import os
import shutil
import json
from pathlib import Path
import kagglehub

def download_kaggle_dataset():
    """Download the annotated potholes dataset from Kaggle"""
    print("Downloading annotated potholes dataset from Kaggle...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("chitholian/annotated-potholes-dataset")
        print(f"Path to dataset files: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have:")
        print("1. Kaggle account and API key configured")
        print("2. Internet connection")
        print("3. Accepted the dataset terms on Kaggle website")
        return None

def organize_dataset(kaggle_path, target_dir="./data"):
    """
    Organize the Kaggle dataset into our expected structure
    """
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    kaggle_path = Path(kaggle_path)
    
    print(f"Organizing dataset from {kaggle_path} to {target_path}")
    
    # Create directory structure
    (target_path / "images" / "pothole").mkdir(parents=True, exist_ok=True)
    (target_path / "images" / "no_pothole").mkdir(parents=True, exist_ok=True)
    (target_path / "annotations").mkdir(parents=True, exist_ok=True)
    
    # Explore the Kaggle dataset structure
    print("\nExploring Kaggle dataset structure:")
    for item in kaggle_path.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(kaggle_path)}")
    
    # Look for common annotation formats
    annotation_files = []
    image_files = []
    
    for item in kaggle_path.rglob("*"):
        if item.is_file():
            suffix = item.suffix.lower()
            if suffix in ['.json', '.xml', '.txt', '.csv']:
                annotation_files.append(item)
            elif suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.append(item)
    
    print(f"\nFound {len(image_files)} images and {len(annotation_files)} annotation files")
    
    # Copy annotation files
    for ann_file in annotation_files:
        target_ann = target_path / "annotations" / ann_file.name
        shutil.copy2(ann_file, target_ann)
        print(f"Copied annotation: {ann_file.name}")
    
    # Analyze and organize images
    pothole_count = 0
    no_pothole_count = 0
    
    for img_file in image_files:
        # Try to determine if image contains pothole based on filename or structure
        filename = img_file.name.lower()
        parent_dir = img_file.parent.name.lower()
        
        # Common naming patterns for potholes
        has_pothole = any(keyword in filename for keyword in ['pothole', 'damage', 'crack', 'hole']) or \
                     any(keyword in parent_dir for keyword in ['pothole', 'damage', 'positive'])
        
        if has_pothole:
            target_img = target_path / "images" / "pothole" / img_file.name
            shutil.copy2(img_file, target_img)
            pothole_count += 1
        else:
            target_img = target_path / "images" / "no_pothole" / img_file.name
            shutil.copy2(img_file, target_img)
            no_pothole_count += 1
    
    print(f"\nOrganized images:")
    print(f"  Pothole images: {pothole_count}")
    print(f"  No pothole images: {no_pothole_count}")
    
    # Create a dataset info file
    dataset_info = {
        "source": "Kaggle - chitholian/annotated-potholes-dataset",
        "total_images": len(image_files),
        "pothole_images": pothole_count,
        "no_pothole_images": no_pothole_count,
        "annotation_files": [f.name for f in annotation_files],
        "download_date": str(Path.cwd()),
        "kaggle_path": str(kaggle_path)
    }
    
    with open(target_path / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    return target_path

def main():
    """Main function to download and organize dataset"""
    print("=== Kaggle Dataset Download and Organization ===\n")
    
    # Download dataset
    kaggle_path = download_kaggle_dataset()
    if not kaggle_path:
        return False
    
    # Organize dataset
    target_dir = organize_dataset(kaggle_path)
    
    print(f"\n✅ Dataset successfully organized in: {target_dir}")
    print("\nNext steps:")
    print("1. Review the organized dataset structure")
    print("2. Run: python scripts/prepare_datasets.py")
    print("3. Start training: python src/train.py")
    
    return True

if __name__ == "__main__":
    main()