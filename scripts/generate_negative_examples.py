#!/usr/bin/env python3
"""
Generate negative examples for classification training
Creates no_pothole examples by cropping regions without potholes
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NegativeExampleGenerator:
    """Generate negative examples from positive images"""
    
    def __init__(self, source_dir="data", processed_dir="data/processed"):
        self.source_dir = Path(source_dir)
        self.processed_dir = Path(processed_dir)
        self.crop_size = (224, 224)  # Standard input size
        
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
            if obj.find('name').text.lower() == 'pothole':
                obj_data = {
                    'bbox': {
                        'xmin': int(obj.find('bndbox/xmin').text),
                        'ymin': int(obj.find('bndbox/ymin').text),
                        'xmax': int(obj.find('bndbox/xmax').text),
                        'ymax': int(obj.find('bndbox/ymax').text)
                    }
                }
                annotation['objects'].append(obj_data)
        
        return annotation
    
    def get_non_pothole_regions(self, image_size, pothole_bboxes, crop_size=(224, 224), num_crops=3):
        """Generate random crop regions that don't overlap with potholes"""
        width, height = image_size
        crop_w, crop_h = crop_size
        
        if width < crop_w or height < crop_h:
            return []
        
        valid_crops = []
        max_attempts = 50
        
        for _ in range(max_attempts):
            if len(valid_crops) >= num_crops:
                break
                
            # Random crop position
            x = random.randint(0, width - crop_w)
            y = random.randint(0, height - crop_h)
            
            crop_bbox = {
                'xmin': x,
                'ymin': y,
                'xmax': x + crop_w,
                'ymax': y + crop_h
            }
            
            # Check if crop overlaps with any pothole
            overlap = False
            for pothole_bbox in pothole_bboxes:
                if self.bbox_overlap(crop_bbox, pothole_bbox):
                    overlap = True
                    break
            
            if not overlap:
                valid_crops.append(crop_bbox)
        
        return valid_crops
    
    def bbox_overlap(self, bbox1, bbox2, threshold=0.1):
        """Check if two bounding boxes overlap significantly"""
        # Calculate intersection
        x1 = max(bbox1['xmin'], bbox2['xmin'])
        y1 = max(bbox1['ymin'], bbox2['ymin'])
        x2 = min(bbox1['xmax'], bbox2['xmax'])
        y2 = min(bbox1['ymax'], bbox2['ymax'])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        bbox1_area = (bbox1['xmax'] - bbox1['xmin']) * (bbox1['ymax'] - bbox1['ymin'])
        bbox2_area = (bbox2['xmax'] - bbox2['xmin']) * (bbox2['ymax'] - bbox2['ymin'])
        
        # Calculate IoU
        union_area = bbox1_area + bbox2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou > threshold
    
    def generate_negative_examples(self, split="train", target_ratio=1.0):
        """Generate negative examples for a specific split"""
        logger.info(f"Generating negative examples for {split} split...")
        
        pothole_dir = self.processed_dir / "classification" / split / "pothole"
        no_pothole_dir = self.processed_dir / "classification" / split / "no_pothole"
        
        pothole_images = list(pothole_dir.glob("*.jpg"))
        target_count = int(len(pothole_images) * target_ratio)
        
        logger.info(f"Target negative examples: {target_count}")
        
        generated_count = 0
        
        for img_path in pothole_images:
            if generated_count >= target_count:
                break
            
            # Load image
            try:
                image = Image.open(img_path)
                image_size = image.size  # (width, height)
                
                # Find corresponding XML annotation
                xml_file = img_path.name.replace('.jpg', '.xml')
                xml_path = self.source_dir / "annotations" / xml_file
                
                if not xml_path.exists():
                    continue
                
                # Parse annotation
                annotation = self.parse_xml_annotation(xml_path)
                pothole_bboxes = [obj['bbox'] for obj in annotation['objects']]
                
                # Generate non-pothole crops
                negative_crops = self.get_non_pothole_regions(
                    image_size, pothole_bboxes, self.crop_size, num_crops=2
                )
                
                # Save negative crops
                for i, crop_bbox in enumerate(negative_crops):
                    if generated_count >= target_count:
                        break
                    
                    crop = image.crop((
                        crop_bbox['xmin'],
                        crop_bbox['ymin'],
                        crop_bbox['xmax'],
                        crop_bbox['ymax']
                    ))
                    
                    # Resize to standard size
                    crop = crop.resize(self.crop_size)
                    
                    # Save crop
                    crop_name = f"neg_{img_path.stem}_{i}.jpg"
                    crop_path = no_pothole_dir / crop_name
                    crop.save(crop_path)
                    
                    generated_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
        
        logger.info(f"Generated {generated_count} negative examples for {split}")
        return generated_count
    
    def generate_all_splits(self):
        """Generate negative examples for all splits"""
        results = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / "classification" / split / "pothole"
            if split_dir.exists():
                count = self.generate_negative_examples(split, target_ratio=0.8)
                results[split] = count
        
        # Update summary
        self.update_summary(results)
        
        return results
    
    def update_summary(self, negative_counts):
        """Update dataset summary with negative example counts"""
        summary_file = self.processed_dir / "dataset_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # Update classification counts
        for split, neg_count in negative_counts.items():
            if 'classification' not in summary:
                summary['classification'] = {}
            if split not in summary['classification']:
                summary['classification'][split] = {}
            
            summary['classification'][split]['no_pothole'] = neg_count
            
            # Count actual positive examples
            pothole_dir = self.processed_dir / "classification" / split / "pothole"
            pos_count = len(list(pothole_dir.glob("*.jpg")))
            summary['classification'][split]['pothole'] = pos_count
            summary['classification'][split]['total'] = pos_count + neg_count
        
        # Save updated summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Updated summary: {summary_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate negative examples for classification')
    parser.add_argument('--source_dir', type=str, default='data',
                       help='Source directory containing original dataset')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Processed dataset directory')
    parser.add_argument('--target_ratio', type=float, default=0.8,
                       help='Ratio of negative to positive examples')
    
    args = parser.parse_args()
    
    generator = NegativeExampleGenerator(args.source_dir, args.processed_dir)
    results = generator.generate_all_splits()
    
    print("\n" + "="*50)
    print("NEGATIVE EXAMPLE GENERATION COMPLETE")
    print("="*50)
    for split, count in results.items():
        print(f"{split}: {count} negative examples generated")
    
    print("\nDataset is now ready for training!")

if __name__ == "__main__":
    main()