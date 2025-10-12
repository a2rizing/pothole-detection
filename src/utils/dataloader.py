"""
Dataset handling utilities for pothole detection
Supports loading, preprocessing, augmentation, and train/val/test splits
"""

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import logging

class PotholeDataset(Dataset):
    """
    Custom dataset class for pothole detection
    Supports both classification and object detection tasks
    """
    
    def __init__(self, 
                 image_dir, 
                 annotation_file=None, 
                 transform=None, 
                 task='classification',
                 image_size=(224, 224)):
        """
        Args:
            image_dir (str): Directory containing images
            annotation_file (str): Path to COCO format annotation file for object detection
            transform: Augmentation transforms
            task (str): 'classification' or 'detection'
            image_size (tuple): Target image size (width, height)
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.task = task
        self.image_size = image_size
        
        # Load data based on task
        if task == 'classification':
            self.samples = self._load_classification_data()
        elif task == 'detection':
            self.samples = self._load_detection_data()
        else:
            raise ValueError("Task must be 'classification' or 'detection'")
    
    def _load_classification_data(self):
        """Load data for classification task (pothole/no-pothole)"""
        samples = []
        
        # Assume directory structure: image_dir/pothole/ and image_dir/no_pothole/
        for class_name in ['pothole', 'no_pothole']:
            class_dir = os.path.join(self.image_dir, class_name)
            if os.path.exists(class_dir):
                label = 1 if class_name == 'pothole' else 0
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        samples.append({
                            'image_path': os.path.join(class_dir, img_file),
                            'label': label,
                            'severity': 0  # Default severity for classification
                        })
        
        return samples
    
    def _load_detection_data(self):
        """Load data for object detection task with COCO format annotations"""
        if not self.annotation_file or not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        with open(self.annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id to filename mapping
        image_info = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        samples = []
        for img_id, annotations in image_annotations.items():
            if img_id in image_info:
                image_path = os.path.join(self.image_dir, image_info[img_id]['file_name'])
                if os.path.exists(image_path):
                    samples.append({
                        'image_path': image_path,
                        'annotations': annotations,
                        'image_info': image_info[img_id]
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.task == 'classification':
            return self._get_classification_item(image, sample)
        else:
            return self._get_detection_item(image, sample)
    
    def _get_classification_item(self, image, sample):
        """Process item for classification task"""
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'severity': torch.tensor(sample['severity'], dtype=torch.float32)
        }
    
    def _get_detection_item(self, image, sample):
        """Process item for object detection task"""
        h, w = image.shape[:2]
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        severities = []
        
        for ann in sample['annotations']:
            # COCO format: [x, y, width, height]
            x, y, width, height = ann['bbox']
            
            # Convert to [x1, y1, x2, y2] format
            x1, y1, x2, y2 = x, y, x + width, y + height
            
            # Normalize coordinates
            x1 /= w
            y1 /= h
            x2 /= w
            y2 /= h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(ann.get('category_id', 1))  # Default to pothole class
            severities.append(ann.get('severity', 1.0))  # Default severity
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,))
        severities = torch.tensor(severities, dtype=torch.float32) if severities else torch.zeros((0,))
        
        # Apply transforms
        if self.transform:
            # For detection, we need to handle bbox transforms
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                class_labels=labels.tolist() if len(labels) > 0 else []
            )
            image = transformed['image']
            if 'bboxes' in transformed:
                boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'severities': severities,
            'image_id': torch.tensor(idx)
        }


def get_transforms(phase='train', image_size=(224, 224), task='classification'):
    """
    Get data augmentation transforms for different phases
    
    Args:
        phase (str): 'train', 'val', or 'test'
        image_size (tuple): Target image size
        task (str): 'classification' or 'detection'
    """
    
    if task == 'classification':
        if phase == 'train':
            return A.Compose([
                A.Resize(image_size[1], image_size[0]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.RandomGamma(p=0.2),
                A.CLAHE(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(image_size[1], image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    else:  # detection
        if phase == 'train':
            return A.Compose([
                A.Resize(image_size[1], image_size[0]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(p=0.3),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(image_size[1], image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))


def create_data_splits(image_dir, 
                      annotation_file=None, 
                      test_size=0.2, 
                      val_size=0.15, 
                      random_state=42):
    """
    Create train/validation/test splits
    
    Args:
        image_dir (str): Directory containing images
        annotation_file (str): Path to annotation file (for detection)
        test_size (float): Fraction of data for testing
        val_size (float): Fraction of remaining data for validation
        random_state (int): Random seed
    
    Returns:
        dict: Dictionary containing train/val/test image lists
    """
    
    # Get all image files
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Create splits
    train_files, temp_files = train_test_split(
        image_files, test_size=test_size, random_state=random_state
    )
    
    val_files, test_files = train_test_split(
        temp_files, test_size=val_size/(test_size), random_state=random_state
    )
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }


def get_dataloaders(image_dir,
                   annotation_file=None,
                   batch_size=32,
                   num_workers=4,
                   image_size=(224, 224),
                   task='classification',
                   test_size=0.2,
                   val_size=0.15):
    """
    Create train/validation/test dataloaders
    
    Args:
        image_dir (str): Directory containing images
        annotation_file (str): Path to annotation file
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        image_size (tuple): Target image size
        task (str): 'classification' or 'detection'
        test_size (float): Test set fraction
        val_size (float): Validation set fraction
    
    Returns:
        dict: Dictionary containing train/val/test dataloaders
    """
    
    # Get transforms
    train_transform = get_transforms('train', image_size, task)
    val_transform = get_transforms('val', image_size, task)
    
    # Create datasets
    train_dataset = PotholeDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=train_transform,
        task=task,
        image_size=image_size
    )
    
    val_dataset = PotholeDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=val_transform,
        task=task,
        image_size=image_size
    )
    
    test_dataset = PotholeDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=val_transform,
        task=task,
        image_size=image_size
    )
    
    # Split data
    splits = create_data_splits(image_dir, annotation_file, test_size, val_size)
    
    # Update datasets with splits
    train_dataset.samples = [s for s in train_dataset.samples 
                           if s['image_path'] in splits['train']]
    val_dataset.samples = [s for s in val_dataset.samples 
                         if s['image_path'] in splits['val']]
    test_dataset.samples = [s for s in test_dataset.samples 
                          if s['image_path'] in splits['test']]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def collate_fn_detection(batch):
    """
    Custom collate function for object detection batches
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'])
        targets.append({
            'boxes': item['boxes'],
            'labels': item['labels'],
            'severities': item['severities'],
            'image_id': item['image_id']
        })
    
    return torch.stack(images), targets


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the dataloader
    
    # For classification
    print("Testing classification dataloader...")
    try:
        dataloaders = get_dataloaders(
            image_dir="./data/images",
            task='classification',
            batch_size=16,
            image_size=(224, 224)
        )
        
        # Test a batch
        for batch in dataloaders['train']:
            print(f"Image batch shape: {batch['image'].shape}")
            print(f"Label batch shape: {batch['label'].shape}")
            break
            
    except Exception as e:
        print(f"Classification test failed: {e}")
    
    # For detection
    print("Testing detection dataloader...")
    try:
        dataloaders = get_dataloaders(
            image_dir="./data/images",
            annotation_file="./data/annotations.json",
            task='detection',
            batch_size=8,
            image_size=(416, 416)
        )
        
        # Test a batch
        for images, targets in dataloaders['train']:
            print(f"Image batch shape: {images.shape}")
            print(f"Number of targets: {len(targets)}")
            break
            
    except Exception as e:
        print(f"Detection test failed: {e}")