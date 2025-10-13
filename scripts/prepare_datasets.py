#!/usr/bin/env python3
"""
Dataset Download and Preparation Script for Pothole Detection
Downloads and prepares publicly available pothole datasets for training
"""

import os
import requests
import zipfile
import tarfile
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Download and prepare pothole datasets"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_file(self, url, filename, description="Downloading"):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
    
    def download_kaggle_pothole_dataset(self):
        """
        Download Kaggle pothole dataset
        Note: Requires Kaggle API setup
        """
        dataset_dir = self.data_dir / "kaggle_pothole"
        dataset_dir.mkdir(exist_ok=True)
        
        logger.info("Downloading Kaggle pothole dataset...")
        
        # Instructions for manual download if Kaggle API not available
        print("\n" + "="*60)
        print("KAGGLE DATASET DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print("1. Go to: https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset")
        print("2. Download the dataset manually")
        print(f"3. Extract to: {dataset_dir}")
        print("4. Run this script again with --prepare-kaggle")
        print("="*60)
        
        # Try automatic download if kaggle API is available
        try:
            os.system(f"kaggle datasets download -d chitholian/annotated-potholes-dataset -p {dataset_dir}")
            # Extract if downloaded
            zip_file = dataset_dir / "annotated-potholes-dataset.zip"
            if zip_file.exists():
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                zip_file.unlink()  # Remove zip file
                logger.info("Kaggle dataset downloaded and extracted successfully")
                return True
        except Exception as e:
            logger.warning(f"Automatic download failed: {e}")
            return False
        
        return False
    
    def download_road_damage_dataset(self):
        """Download Road Damage Dataset (subset with potholes)"""
        dataset_dir = self.data_dir / "road_damage"
        dataset_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("ROAD DAMAGE DATASET DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print("1. Go to: https://github.com/sekilab/RoadDamageDetector")
        print("2. Download RDD2020 or RDD2022 dataset")
        print(f"3. Extract to: {dataset_dir}")
        print("4. Run this script again with --prepare-road-damage")
        print("="*60)
        
        return False
    
    def create_synthetic_dataset(self, num_samples=1000):
        """Create a synthetic dataset for initial testing"""
        logger.info(f"Creating synthetic dataset with {num_samples} samples...")
        
        synthetic_dir = self.data_dir / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        # Create directories
        train_dir = synthetic_dir / "train"
        val_dir = synthetic_dir / "val"
        test_dir = synthetic_dir / "test"
        
        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / "pothole").mkdir(parents=True, exist_ok=True)
            (split_dir / "no_pothole").mkdir(parents=True, exist_ok=True)
        
        # Distribution: 70% train, 20% val, 10% test
        train_samples = int(num_samples * 0.7)
        val_samples = int(num_samples * 0.2)
        test_samples = num_samples - train_samples - val_samples
        
        splits = [
            (train_dir, train_samples, "train"),
            (val_dir, val_samples, "val"),
            (test_dir, test_samples, "test")
        ]
        
        for split_dir, n_samples, split_name in splits:
            logger.info(f"Generating {n_samples} samples for {split_name} split...")
            
            for i in tqdm(range(n_samples), desc=f"Creating {split_name} images"):
                # Create pothole image (50% chance)
                is_pothole = np.random.random() > 0.5
                
                # Generate synthetic road image
                img = self._generate_road_image(is_pothole)
                
                # Save image
                class_name = "pothole" if is_pothole else "no_pothole"
                filename = f"{split_name}_{class_name}_{i:04d}.jpg"
                cv2.imwrite(str(split_dir / class_name / filename), img)
        
        # Create annotation files
        self._create_synthetic_annotations(synthetic_dir)
        
        logger.info(f"Synthetic dataset created at: {synthetic_dir}")
        return True
    
    def _generate_road_image(self, has_pothole=False, size=(224, 224)):
        """Generate a synthetic road image"""
        img = np.random.randint(80, 120, (size[1], size[0], 3), dtype=np.uint8)
        
        # Add road texture
        noise = np.random.randint(-20, 20, (size[1], size[0], 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Add road markings occasionally
        if np.random.random() > 0.7:
            cv2.line(img, (0, size[1]//2), (size[0], size[1]//2), (255, 255, 255), 2)
        
        if has_pothole:
            # Add pothole (dark circular/oval region)
            center_x = np.random.randint(50, size[0]-50)
            center_y = np.random.randint(50, size[1]-50)
            radius_x = np.random.randint(15, 40)
            radius_y = np.random.randint(10, 30)
            
            # Create pothole mask
            y, x = np.ogrid[:size[1], :size[0]]
            mask = ((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2) <= 1
            
            # Darken the pothole area
            img[mask] = img[mask] * 0.3
            
            # Add some water reflection effect
            if np.random.random() > 0.5:
                water_mask = mask & (np.random.random((size[1], size[0])) > 0.7)
                img[water_mask] = [100, 150, 200]  # Bluish water color
        
        return img
    
    def _create_synthetic_annotations(self, dataset_dir):
        """Create simple train/val/test split files"""
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            annotation_file = dataset_dir / f"{split}_annotations.txt"
            
            with open(annotation_file, 'w') as f:
                f.write("image_path,label,class_name\n")
                
                for class_name in ['pothole', 'no_pothole']:
                    class_dir = split_dir / class_name
                    label = 1 if class_name == 'pothole' else 0
                    
                    for img_file in class_dir.glob("*.jpg"):
                        relative_path = str(img_file.relative_to(dataset_dir))
                        f.write(f"{relative_path},{label},{class_name}\n")
    
    def prepare_kaggle_dataset(self):
        """Prepare downloaded Kaggle dataset"""
        kaggle_dir = self.data_dir / "kaggle_pothole"
        
        if not kaggle_dir.exists():
            logger.error("Kaggle dataset not found. Please download first.")
            return False
        
        logger.info("Preparing Kaggle pothole dataset...")
        
        # Find images and create structure
        prepared_dir = self.data_dir / "kaggle_prepared"
        prepared_dir.mkdir(exist_ok=True)
        
        # Look for images in the downloaded dataset
        image_files = list(kaggle_dir.rglob("*.jpg")) + list(kaggle_dir.rglob("*.png"))
        
        if not image_files:
            logger.error("No images found in Kaggle dataset directory")
            return False
        
        # Create train/val/test splits
        np.random.shuffle(image_files)
        n_total = len(image_files)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.2)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        # Copy files and create structure
        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_dir = prepared_dir / split_name
            (split_dir / "pothole").mkdir(parents=True, exist_ok=True)
            
            for img_file in tqdm(files, desc=f"Preparing {split_name} images"):
                # Assume all images contain potholes for now
                dest_file = split_dir / "pothole" / img_file.name
                shutil.copy2(img_file, dest_file)
        
        logger.info(f"Kaggle dataset prepared at: {prepared_dir}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Download and prepare pothole datasets')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic dataset')
    parser.add_argument('--synthetic-samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--kaggle', action='store_true', help='Download Kaggle dataset')
    parser.add_argument('--prepare-kaggle', action='store_true', help='Prepare downloaded Kaggle dataset')
    parser.add_argument('--road-damage', action='store_true', help='Download Road Damage dataset')
    parser.add_argument('--all', action='store_true', help='Download all available datasets')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.synthetic or args.all:
        downloader.create_synthetic_dataset(args.synthetic_samples)
    
    if args.kaggle or args.all:
        downloader.download_kaggle_dataset()
    
    if args.prepare_kaggle:
        downloader.prepare_kaggle_dataset()
    
    if args.road_damage or args.all:
        downloader.download_road_damage_dataset()
    
    if not any([args.synthetic, args.kaggle, args.prepare_kaggle, args.road_damage, args.all]):
        print("No dataset option selected. Use --help for options.")
        print("\nQuick start for immediate training:")
        print("python scripts/prepare_datasets.py --synthetic --synthetic-samples 2000")

if __name__ == "__main__":
    main()