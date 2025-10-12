"""
Real-time inference script for pothole detection
Supports live camera feed, video files, and image processing with visualization
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pothole_net import create_model
from utils.dataloader import get_transforms

class PotholeDetectionInference:
    """
    Real-time pothole detection inference class
    Supports classification and detection tasks with visualization
    """
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = 'potholeNet',
                 device: str = 'auto',
                 conf_threshold: float = 0.5,
                 input_size: Tuple[int, int] = (224, 224),
                 task: str = 'classification'):
        """
        Args:
            model_path (str): Path to trained model checkpoint
            model_type (str): Type of model to load
            device (str): Device to run inference on ('auto', 'cpu', 'cuda')
            conf_threshold (float): Confidence threshold for predictions
            input_size (tuple): Input image size (H, W)
            task (str): Task type ('classification' or 'detection')
        """
        
        self.model_path = model_path
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.task = task
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Setup transforms
        self.transform = get_transforms('val', input_size, task)
        
        # Class names (can be loaded from config)
        self.class_names = ['No Pothole', 'Pothole']
        self.colors = [(0, 255, 0), (0, 0, 255)]  # Green for no pothole, Red for pothole
        
        # Performance tracking
        self.fps_counter = []
        self.frame_count = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration if available
            if 'config' in checkpoint:
                model_config = checkpoint['config']
                self.logger.info(f"Loaded model config: {model_config}")
            else:
                model_config = {}
            
            # Create model
            if self.task == 'classification':
                model = create_model(
                    model_type=self.model_type,
                    num_classes=2,
                    input_size=self.input_size,
                    with_depth=False
                )
            else:  # detection
                model = create_model(
                    model_type='detector',
                    num_classes=2,
                    input_size=self.input_size
                )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.logger.info(f"Model loaded successfully from {self.model_path}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image (np.ndarray): Input image in BGR format
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def predict_classification(self, image: np.ndarray) -> Dict:
        """
        Perform classification prediction
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            dict: Prediction results
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(input_tensor)
            inference_time = time.time() - start_time
        
        # Process outputs
        if isinstance(outputs, dict):
            classification_output = outputs['classification']
            severity_output = outputs.get('severity', None)
        else:
            classification_output = outputs
            severity_output = None
        
        # Get probabilities and predictions
        probabilities = F.softmax(classification_output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        results = {
            'predicted_class': int(predicted_class.item()),
            'confidence': float(confidence.item()),
            'probabilities': probabilities.squeeze().cpu().numpy().tolist(),
            'class_name': self.class_names[int(predicted_class.item())],
            'inference_time': inference_time
        }
        
        if severity_output is not None:
            results['severity'] = float(severity_output.item())
        
        return results
    
    def predict_detection(self, image: np.ndarray) -> Dict:
        """
        Perform object detection prediction
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            dict: Detection results
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)
        h, w = image.shape[:2]
        
        # Inference
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(input_tensor)
            inference_time = time.time() - start_time
        
        # Process detection outputs
        # This is a simplified version - actual implementation depends on model architecture
        detections = []
        
        if isinstance(outputs, dict):
            # Extract detection components
            cls_output = outputs.get('classification')
            reg_output = outputs.get('regression')
            obj_output = outputs.get('objectness')
            severity_output = outputs.get('severity')
            
            # Process outputs (this is model-specific)
            # For now, return dummy detection for demonstration
            if cls_output is not None and obj_output is not None:
                # Apply confidence thresholding
                obj_scores = torch.sigmoid(obj_output)
                cls_scores = F.softmax(cls_output, dim=1)
                
                # Simple non-maximum suppression would go here
                # For now, just return high-confidence detections
                pass
        
        results = {
            'detections': detections,
            'inference_time': inference_time,
            'num_detections': len(detections)
        }
        
        return results
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Perform prediction based on task type
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            dict: Prediction results
        """
        if self.task == 'classification':
            return self.predict_classification(image)
        else:
            return self.predict_detection(image)
    
    def draw_classification_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw classification results on image
        
        Args:
            image (np.ndarray): Input image
            results (dict): Prediction results
        
        Returns:
            np.ndarray: Image with visualizations
        """
        h, w = image.shape[:2]
        
        # Create overlay
        overlay = image.copy()
        
        # Draw background for text
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw class prediction
        class_name = results['class_name']
        confidence = results['confidence']
        color = self.colors[results['predicted_class']]
        
        cv2.putText(image, f"Class: {class_name}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"Confidence: {confidence:.3f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw severity if available
        if 'severity' in results:
            severity = results['severity']
            cv2.putText(image, f"Severity: {severity:.3f}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw FPS
        if len(self.fps_counter) > 0:
            fps = 1.0 / np.mean(self.fps_counter[-30:])  # Average over last 30 frames
            cv2.putText(image, f"FPS: {fps:.1f}", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def draw_detection_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image (np.ndarray): Input image
            results (dict): Detection results
        
        Returns:
            np.ndarray: Image with visualizations
        """
        h, w = image.shape[:2]
        
        # Draw detections
        for detection in results['detections']:
            bbox = detection.get('bbox', [])
            confidence = detection.get('confidence', 0)
            class_id = detection.get('class_id', 0)
            
            if len(bbox) == 4 and confidence > self.conf_threshold:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                color = self.colors[class_id]
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{self.class_names[class_id]}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw info
        cv2.putText(image, f"Detections: {results['num_detections']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame
        
        Args:
            frame (np.ndarray): Input frame
        
        Returns:
            tuple: (processed_frame, results)
        """
        # Make prediction
        results = self.predict(frame)
        
        # Draw results
        if self.task == 'classification':
            processed_frame = self.draw_classification_results(frame, results)
        else:
            processed_frame = self.draw_detection_results(frame, results)
        
        # Update FPS counter
        self.fps_counter.append(results['inference_time'])
        if len(self.fps_counter) > 100:  # Keep only last 100 frames
            self.fps_counter.pop(0)
        
        self.frame_count += 1
        
        return processed_frame, results
    
    def run_camera(self, camera_id: int = 0):
        """
        Run inference on live camera feed
        
        Args:
            camera_id (int): Camera device ID
        """
        self.logger.info(f"Starting camera inference on device {camera_id}")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Display
                cv2.imshow('Pothole Detection', processed_frame)
                
                # Log periodic results
                if self.frame_count % 30 == 0:
                    avg_fps = 1.0 / np.mean(self.fps_counter[-30:]) if self.fps_counter else 0
                    self.logger.info(f"Frame {self.frame_count}, Avg FPS: {avg_fps:.1f}")
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Run inference on video file
        
        Args:
            video_path (str): Path to input video
            output_path (str, optional): Path to save output video
        """
        self.logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Write frame if output specified
                if writer:
                    writer.write(processed_frame)
                
                # Display progress
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    avg_fps = 1.0 / np.mean(self.fps_counter[-30:]) if self.fps_counter else 0
                    self.logger.info(f"Progress: {progress:.1f}%, Processing FPS: {avg_fps:.1f}")
                
                frame_idx += 1
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            cap.release()
            if writer:
                writer.release()
                self.logger.info(f"Output video saved: {output_path}")
    
    def run_image(self, image_path: str, output_path: Optional[str] = None):
        """
        Run inference on single image
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save output image
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image {image_path}")
        
        # Process image
        processed_image, results = self.process_frame(image)
        
        # Log results
        self.logger.info(f"Results: {results}")
        
        # Save output if specified
        if output_path:
            cv2.imwrite(output_path, processed_image)
            self.logger.info(f"Output image saved: {output_path}")
        
        # Display image
        cv2.imshow('Pothole Detection', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Pothole Detection Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='potholeNet',
                       choices=['potholeNet', 'mini', 'detector'],
                       help='Type of model to load')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'detection'],
                       help='Task type')
    parser.add_argument('--input_type', type=str, required=True,
                       choices=['camera', 'video', 'image'],
                       help='Input type')
    parser.add_argument('--input_path', type=str,
                       help='Path to input file (for video/image)')
    parser.add_argument('--output_path', type=str,
                       help='Path to save output')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224],
                       help='Input image size (H W)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_type in ['video', 'image'] and not args.input_path:
        parser.error(f"--input_path is required for {args.input_type} input")
    
    # Create inference object
    inference = PotholeDetectionInference(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        conf_threshold=args.conf_threshold,
        input_size=tuple(args.input_size),
        task=args.task
    )
    
    # Run inference based on input type
    try:
        if args.input_type == 'camera':
            inference.run_camera(args.camera_id)
        elif args.input_type == 'video':
            inference.run_video(args.input_path, args.output_path)
        elif args.input_type == 'image':
            inference.run_image(args.input_path, args.output_path)
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())