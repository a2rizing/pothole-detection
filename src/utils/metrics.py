"""
Metrics utilities for pothole detection
Includes accuracy, IoU, precision, recall, F1-score, and custom metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class MetricsCalculator:
    """
    Comprehensive metrics calculator for both classification and detection tasks
    """
    
    def __init__(self, task='classification', num_classes=2, device='cuda'):
        """
        Args:
            task (str): 'classification' or 'detection'
            num_classes (int): Number of classes (for classification)
            device (str): Device to run calculations on
        """
        self.task = task
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        if self.task == 'classification':
            self.predictions = []
            self.targets = []
            self.losses = []
        else:  # detection
            self.detection_results = []
            self.ground_truths = []
    
    def update_classification(self, predictions, targets, loss=None):
        """
        Update metrics for classification task
        
        Args:
            predictions (torch.Tensor): Model predictions (logits or probabilities)
            targets (torch.Tensor): Ground truth labels
            loss (float): Loss value for this batch
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
        
        # Convert logits to class predictions
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        self.predictions.extend(pred_classes.numpy())
        self.targets.extend(targets.numpy())
        
        if loss is not None:
            self.losses.append(loss)
    
    def update_detection(self, predictions, targets):
        """
        Update metrics for detection task
        
        Args:
            predictions (list): List of prediction dictionaries
            targets (list): List of target dictionaries
        """
        self.detection_results.extend(predictions)
        self.ground_truths.extend(targets)
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        """
        Compute all classification metrics
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        if not self.predictions or not self.targets:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro and micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            targets, predictions, average='micro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Per-class metrics
        class_metrics = {}
        for i in range(len(precision)):
            class_name = f'class_{i}'
            class_metrics[f'{class_name}_precision'] = float(precision[i])
            class_metrics[f'{class_name}_recall'] = float(recall[i])
            class_metrics[f'{class_name}_f1'] = float(f1[i])
            class_metrics[f'{class_name}_support'] = int(support[i])
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro),
            'confusion_matrix': cm.tolist(),
            **class_metrics
        }
        
        if self.losses:
            metrics['avg_loss'] = float(np.mean(self.losses))
        
        return metrics
    
    def compute_detection_metrics(self, iou_threshold=0.5) -> Dict[str, float]:
        """
        Compute detection metrics including mAP
        
        Args:
            iou_threshold (float): IoU threshold for positive detection
        
        Returns:
            dict: Dictionary containing detection metrics
        """
        if not self.detection_results or not self.ground_truths:
            return {}
        
        # Calculate IoU for all predictions
        all_ious = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(self.detection_results, self.ground_truths):
            pred_boxes = pred.get('boxes', torch.tensor([]))
            gt_boxes = gt.get('boxes', torch.tensor([]))
            
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            elif len(pred_boxes) == 0:
                false_negatives += len(gt_boxes)
                continue
            elif len(gt_boxes) == 0:
                false_positives += len(pred_boxes)
                continue
            
            # Calculate IoU matrix
            iou_matrix = self._calculate_iou_matrix(pred_boxes, gt_boxes)
            
            # Find best matches
            max_ious = torch.max(iou_matrix, dim=1)[0]
            matched_gt = torch.max(iou_matrix, dim=1)[1]
            
            # Count TP, FP, FN
            for i, iou in enumerate(max_ious):
                if iou >= iou_threshold:
                    true_positives += 1
                    all_ious.append(float(iou))
                else:
                    false_positives += 1
            
            # Unmatched ground truth boxes are false negatives
            matched_gt_unique = torch.unique(matched_gt[max_ious >= iou_threshold])
            false_negatives += len(gt_boxes) - len(matched_gt_unique)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = np.mean(all_ious) if all_ious else 0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_iou': float(avg_iou),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _calculate_iou_matrix(self, boxes1, boxes2):
        """
        Calculate IoU matrix between two sets of boxes
        
        Args:
            boxes1 (torch.Tensor): First set of boxes [N, 4]
            boxes2 (torch.Tensor): Second set of boxes [M, 4]
        
        Returns:
            torch.Tensor: IoU matrix [N, M]
        """
        # Calculate intersection areas
        x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union = area1[:, None] + area2[None, :] - intersection
        
        # Calculate IoU
        iou = intersection / torch.clamp(union, min=1e-6)
        
        return iou
    
    def compute_metrics(self, **kwargs) -> Dict[str, float]:
        """
        Compute metrics based on task type
        
        Returns:
            dict: Computed metrics
        """
        if self.task == 'classification':
            return self.compute_classification_metrics()
        else:
            return self.compute_detection_metrics(**kwargs)
    
    def plot_confusion_matrix(self, save_path=None, class_names=None):
        """
        Plot confusion matrix for classification task
        
        Args:
            save_path (str): Path to save the plot
            class_names (list): List of class names
        """
        if self.task != 'classification' or not self.predictions:
            print("Confusion matrix only available for classification task")
            return
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        box1 (list/array): [x1, y1, x2, y2]
        box2 (list/array): [x1, y1, x2, y2]
    
    Returns:
        float: IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_map(predictions, ground_truths, iou_thresholds=None, confidence_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection
    
    Args:
        predictions (list): List of prediction dictionaries
        ground_truths (list): List of ground truth dictionaries
        iou_thresholds (list): List of IoU thresholds (default: [0.5:0.95:0.05])
        confidence_threshold (float): Confidence threshold for predictions
    
    Returns:
        dict: mAP metrics
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    aps = []
    
    for iou_thresh in iou_thresholds:
        ap = calculate_average_precision(
            predictions, ground_truths, iou_thresh, confidence_threshold
        )
        aps.append(ap)
    
    return {
        'mAP': np.mean(aps),
        'mAP_50': aps[0] if len(aps) > 0 else 0.0,
        'mAP_75': aps[5] if len(aps) > 5 else 0.0,
        'APs': aps
    }


def calculate_average_precision(predictions, ground_truths, iou_threshold, confidence_threshold):
    """
    Calculate Average Precision (AP) for a single IoU threshold
    
    Args:
        predictions (list): List of prediction dictionaries
        ground_truths (list): List of ground truth dictionaries
        iou_threshold (float): IoU threshold
        confidence_threshold (float): Confidence threshold
    
    Returns:
        float: Average Precision
    """
    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Filter predictions by confidence
        if 'scores' in pred:
            valid_indices = pred['scores'] > confidence_threshold
            filtered_pred = {
                'boxes': pred['boxes'][valid_indices],
                'scores': pred['scores'][valid_indices],
                'image_id': i
            }
        else:
            filtered_pred = {
                'boxes': pred['boxes'],
                'scores': torch.ones(len(pred['boxes'])),
                'image_id': i
            }
        
        all_predictions.append(filtered_pred)
        
        gt_with_id = {
            'boxes': gt['boxes'],
            'image_id': i
        }
        all_ground_truths.append(gt_with_id)
    
    # Sort predictions by confidence score
    sorted_predictions = []
    for pred in all_predictions:
        for j in range(len(pred['boxes'])):
            sorted_predictions.append({
                'box': pred['boxes'][j],
                'score': pred['scores'][j],
                'image_id': pred['image_id']
            })
    
    sorted_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate precision and recall
    tp = np.zeros(len(sorted_predictions))
    fp = np.zeros(len(sorted_predictions))
    
    # Keep track of matched ground truths
    matched_gts = set()
    
    for i, pred in enumerate(sorted_predictions):
        pred_box = pred['box']
        pred_image_id = pred['image_id']
        
        # Find ground truths for this image
        gt_boxes = all_ground_truths[pred_image_id]['boxes']
        
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            gt_key = (pred_image_id, j)
            if gt_key in matched_gts:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            gt_key = (pred_image_id, best_gt_idx)
            if gt_key not in matched_gts:
                tp[i] = 1
                matched_gts.add(gt_key)
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / max(1, sum(len(gt['boxes']) for gt in all_ground_truths))
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap


def severity_mae(predictions, targets):
    """
    Calculate Mean Absolute Error for severity estimation
    
    Args:
        predictions (torch.Tensor): Predicted severity values
        targets (torch.Tensor): Target severity values
    
    Returns:
        float: MAE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    return np.mean(np.abs(predictions - targets))


def severity_rmse(predictions, targets):
    """
    Calculate Root Mean Square Error for severity estimation
    
    Args:
        predictions (torch.Tensor): Predicted severity values
        targets (torch.Tensor): Target severity values
    
    Returns:
        float: RMSE value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    return np.sqrt(np.mean((predictions - targets) ** 2))


# Example usage and testing
if __name__ == "__main__":
    # Test classification metrics
    print("Testing classification metrics...")
    
    # Create dummy data
    predictions = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
    targets = torch.tensor([1, 0, 1, 0])
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(task='classification', num_classes=2)
    metrics_calc.update_classification(predictions, targets, loss=0.5)
    
    # Compute metrics
    metrics = metrics_calc.compute_metrics()
    print("Classification metrics:", metrics)
    
    # Test detection metrics
    print("\nTesting detection metrics...")
    
    # Create dummy detection data
    pred_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]])
    gt_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]])
    
    pred_dict = {'boxes': pred_boxes}
    gt_dict = {'boxes': gt_boxes}
    
    metrics_calc_det = MetricsCalculator(task='detection')
    metrics_calc_det.update_detection([pred_dict], [gt_dict])
    
    # Compute detection metrics
    det_metrics = metrics_calc_det.compute_metrics()
    print("Detection metrics:", det_metrics)
    
    # Test IoU calculation
    box1 = [0.1, 0.1, 0.3, 0.3]
    box2 = [0.2, 0.2, 0.4, 0.4]
    iou = calculate_iou(box1, box2)
    print(f"\nIoU between {box1} and {box2}: {iou:.3f}")