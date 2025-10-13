#!/usr/bin/env python3
"""
Process Kaggle Annotated Potholes Dataset for Training
Converts Pascal VOC XML annotations to classification and detection format
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
import random
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleDatasetProcessor:
    """Process Kaggle annotated potholes dataset"""
    
    def __init__(self, source_dir="data", output_dir="data/processed"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create output directories
        (self.output_dir / "classification" / "train" / "pothole").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "classification" / "train" / "no_pothole").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "classification" / "val" / "pothole").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "classification" / "val" / "no_pothole").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "classification" / "test" / "pothole").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "classification" / "test" / "no_pothole").mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / "detection" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "detection" / "annotations").mkdir(parents=True, exist_ok=True)
    
    def parse_xml_annotation(self, xml_path):
        """Parse Pascal VOC XML annotation"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotation = {
            'filename': root.find('filename').text,
            'width': int(root.find('size/width').text),
            'height': int(root.find('size/height').text),
            'objects': []
        }
        
        for obj in root.findall('object'):
            obj_data = {
                'name': obj.find('name').text,
                'bbox': {
                    'xmin': int(obj.find('bndbox/xmin').text),
                    'ymin': int(obj.find('bndbox/ymin').text),
                    'xmax': int(obj.find('bndbox/xmax').text),
                    'ymax': int(obj.find('bndbox/ymax').text)
                }
            }
            annotation['objects'].append(obj_data)
        
        return annotation
    
    def load_splits(self):
        """Load train/test splits from splits.json"""
        splits_file = self.source_dir / "annotations" / "splits.json"
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # Convert to image names (remove .xml extension)
        splits_images = {}
        for split_name, xml_files in splits.items():
            splits_images[split_name] = [f.replace('.xml', '.jpg') for f in xml_files]
        
        # Create validation split from training data (80/20 split)
        train_images = splits_images['train']
        random.shuffle(train_images)
        split_idx = int(0.8 * len(train_images))
        
        return {
            'train': train_images[:split_idx],
            'val': train_images[split_idx:],
            'test': splits_images['test']
        }
    
    def analyze_dataset(self):
        """Analyze the dataset to understand annotations"""
        annotations_dir = self.source_dir / "annotations"
        xml_files = list(annotations_dir.glob("*.xml"))
        xml_files = [f for f in xml_files if f.name != "splits.json"]
        
        stats = {
            'total_images': 0,
            'images_with_potholes': 0,
            'images_without_potholes': 0,
            'total_potholes': 0,
            'bbox_sizes': []
        }
        
        object_counts = defaultdict(int)
        
        for xml_file in xml_files:
            try:
                annotation = self.parse_xml_annotation(xml_file)
                stats['total_images'] += 1
                
                has_pothole = False
                for obj in annotation['objects']:
                    object_counts[obj['name']] += 1
                    
                    if obj['name'].lower() == 'pothole':
                        has_pothole = True
                        stats['total_potholes'] += 1
                        
                        # Calculate bbox size
                        bbox = obj['bbox']
                        width = bbox['xmax'] - bbox['xmin']
                        height = bbox['ymax'] - bbox['ymin']
                        stats['bbox_sizes'].append((width, height))
                
                if has_pothole:
                    stats['images_with_potholes'] += 1
                else:
                    stats['images_without_potholes'] += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {xml_file}: {e}")
        
        logger.info(f"Dataset Analysis:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Images with potholes: {stats['images_with_potholes']}")
        logger.info(f"  Images without potholes: {stats['images_without_potholes']}")
        logger.info(f"  Total pothole objects: {stats['total_potholes']}")
        logger.info(f"  Object types: {dict(object_counts)}")
        
        if stats['bbox_sizes']:
            bbox_sizes = np.array(stats['bbox_sizes'])
            logger.info(f"  Average bbox size: {bbox_sizes.mean(axis=0)}")
            logger.info(f"  Min bbox size: {bbox_sizes.min(axis=0)}")
            logger.info(f"  Max bbox size: {bbox_sizes.max(axis=0)}")
        
        return stats
    
    def create_classification_dataset(self):
        """Create classification dataset (pothole/no_pothole)"""
        logger.info("Creating classification dataset...")
        
        # Get image splits
        splits = self.load_splits()
        
        # Process each split
        for split_name, image_files in splits.items():
            logger.info(f"Processing {split_name} split: {len(image_files)} images")
            
            for image_file in image_files:
                # Find corresponding XML annotation
                xml_file = image_file.replace('.jpg', '.xml')
                xml_path = self.source_dir / "annotations" / xml_file
                
                if not xml_path.exists():
                    logger.warning(f"XML not found for {image_file}")
                    continue
                
                # Parse annotation
                annotation = self.parse_xml_annotation(xml_path)
                
                # Check if image has potholes
                has_pothole = any(obj['name'].lower() == 'pothole' for obj in annotation['objects'])
                
                # Copy to appropriate directory
                source_img = self.source_dir / "images" / "no_pothole" / image_file
                if not source_img.exists():
                    source_img = self.source_dir / "images" / "pothole" / image_file
                
                if source_img.exists():
                    if has_pothole:
                        target_dir = self.output_dir / "classification" / split_name / "pothole"
                    else:
                        target_dir = self.output_dir / "classification" / split_name / "no_pothole"
                    
                    target_path = target_dir / image_file
                    shutil.copy2(source_img, target_path)
                else:
                    logger.warning(f"Image file not found: {image_file}")
        
        # Count results
        for split_name in ['train', 'val', 'test']:
            pothole_count = len(list((self.output_dir / "classification" / split_name / "pothole").glob("*.jpg")))
            no_pothole_count = len(list((self.output_dir / "classification" / split_name / "no_pothole").glob("*.jpg")))
            logger.info(f"{split_name}: {pothole_count} pothole, {no_pothole_count} no_pothole images")
    
    def create_detection_dataset(self):
        """Create detection dataset in COCO format"""
        logger.info("Creating detection dataset...")
        
        # Get image splits
        splits = self.load_splits()
        
        # Process all images for detection
        all_images = []
        for split_images in splits.values():
            all_images.extend(split_images)
        
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'pothole', 'supercategory': 'defect'}
            ]
        }
        
        annotation_id = 1
        
        for idx, image_file in enumerate(all_images):
            # Find corresponding XML annotation
            xml_file = image_file.replace('.jpg', '.xml')
            xml_path = self.source_dir / "annotations" / xml_file
            
            if not xml_path.exists():
                continue
            
            # Parse annotation
            annotation = self.parse_xml_annotation(xml_path)
            
            # Copy image
            source_img = self.source_dir / "images" / "no_pothole" / image_file
            if not source_img.exists():
                source_img = self.source_dir / "images" / "pothole" / image_file
            
            if not source_img.exists():
                continue
            
            target_img = self.output_dir / "detection" / "images" / image_file
            shutil.copy2(source_img, target_img)
            
            # Add image info
            image_info = {
                'id': idx + 1,
                'file_name': image_file,
                'width': annotation['width'],
                'height': annotation['height']
            }
            coco_data['images'].append(image_info)
            
            # Add annotations
            for obj in annotation['objects']:
                if obj['name'].lower() == 'pothole':
                    bbox = obj['bbox']
                    coco_annotation = {
                        'id': annotation_id,
                        'image_id': idx + 1,
                        'category_id': 1,
                        'bbox': [
                            bbox['xmin'],
                            bbox['ymin'],
                            bbox['xmax'] - bbox['xmin'],
                            bbox['ymax'] - bbox['ymin']
                        ],
                        'area': (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin']),
                        'iscrowd': 0
                    }
                    coco_data['annotations'].append(coco_annotation)
                    annotation_id += 1
        
        # Save COCO annotation file
        coco_file = self.output_dir / "detection" / "annotations" / "instances_all.json"
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Detection dataset: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    def create_summary(self):
        """Create dataset summary"""
        summary = {
            'source': 'Kaggle - chitholian/annotated-potholes-dataset',
            'format': 'Pascal VOC XML',
            'processed_date': str(Path.cwd()),
            'datasets_created': ['classification', 'detection'],
            'classification': {},
            'detection': {}
        }
        
        # Count classification dataset
        for split in ['train', 'val', 'test']:
            pothole_count = len(list((self.output_dir / "classification" / split / "pothole").glob("*.jpg")))
            no_pothole_count = len(list((self.output_dir / "classification" / split / "no_pothole").glob("*.jpg")))
            summary['classification'][split] = {
                'pothole': pothole_count,
                'no_pothole': no_pothole_count,
                'total': pothole_count + no_pothole_count
            }
        
        # Count detection dataset
        coco_file = self.output_dir / "detection" / "annotations" / "instances_all.json"
        if coco_file.exists():
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            summary['detection'] = {
                'images': len(coco_data['images']),
                'annotations': len(coco_data['annotations']),
                'categories': len(coco_data['categories'])
            }
        
        # Save summary
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Dataset processing complete!")
        logger.info(f"Summary saved to: {summary_file}")
        
        return summary

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Kaggle Annotated Potholes Dataset')
    parser.add_argument('--source_dir', type=str, default='data', 
                       help='Source directory containing the Kaggle dataset')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed datasets')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze dataset without processing')
    
    args = parser.parse_args()
    
    processor = KaggleDatasetProcessor(args.source_dir, args.output_dir)
    
    # Analyze dataset
    stats = processor.analyze_dataset()
    
    if not args.analyze_only:
        # Create datasets
        processor.create_classification_dataset()
        processor.create_detection_dataset()
        
        # Create summary
        summary = processor.create_summary()
        
        print("\n" + "="*50)
        print("DATASET PROCESSING COMPLETE")
        print("="*50)
        print(f"Classification dataset created at: {processor.output_dir}/classification")
        print(f"Detection dataset created at: {processor.output_dir}/detection")
        print("\nNext steps:")
        print("1. Review the processed datasets")
        print("2. Update config/model_config.yaml")
        print("3. Start training: python src/train.py")

if __name__ == "__main__":
    main()