#!/usr/bin/env python3
"""
🔍 SLIDING WINDOW OBJECT DETECTION
=================================
Convert your classification model to object detection using sliding windows
Detects pothole locations in real-time camera feed

Usage: python scripts/sliding_window_detection.py
"""

import cv2
import torch
import numpy as np
import sys
import time
from datetime import datetime
import os

# Add src to path
sys.path.append('src')

try:
    from models.pothole_net import SimplePotholeNet, MiniPotholeNet
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️ Model classes not available. Using mock predictions.")

class SlidingWindowDetector:
    def __init__(self, model_path='models/pothole_detection_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Detection parameters
        self.window_size = (224, 224)  # Model input size
        self.stride = 56  # Step size (1/4 of window for overlap)
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.3  # Non-maximum suppression
        
        print(f"🔍 Initializing Sliding Window Detector...")
        print(f"📱 Device: {self.device}")
        print(f"🪟 Window size: {self.window_size}")
        print(f"👣 Stride: {self.stride}")
        
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained classification model"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                print("💡 Will use mock predictions for demo")
                return
                
            if not MODEL_AVAILABLE:
                print("⚠️ Model classes not available. Using mock predictions.")
                return
                
            print(f"📂 Loading model from: {model_path}")
            
            self.model = SimplePotholeNet()
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                accuracy = checkpoint.get('accuracy', 'Unknown')
                print(f"✅ Model loaded! Training accuracy: {accuracy}")
            else:
                self.model.load_state_dict(checkpoint)
                print("✅ Model loaded successfully!")
                
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("💡 Will use mock predictions")
            
    def preprocess_window(self, window):
        """Preprocess window for classification"""
        # Resize to model input size
        window_resized = cv2.resize(window, self.window_size)
        
        # Convert BGR to RGB
        window_rgb = cv2.cvtColor(window_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        window_normalized = window_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        window_tensor = torch.FloatTensor(window_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return window_tensor
        
    def classify_window(self, window):
        """Classify a single window"""
        try:
            if self.model_loaded and MODEL_AVAILABLE:
                window_tensor = self.preprocess_window(window)
                
                with torch.no_grad():
                    outputs = self.model(window_tensor.to(self.device))
                    probabilities = torch.softmax(outputs, dim=1)
                    pothole_prob = probabilities[0][1].item()  # Probability of pothole class
                    
                return pothole_prob
            else:
                # Mock prediction
                return np.random.uniform(0.1, 0.9)
                
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return 0.0
            
    def non_maximum_suppression(self, boxes, scores, threshold=0.3):
        """Apply NMS to remove overlapping detections"""
        if len(boxes) == 0:
            return []
            
        # Convert to numpy arrays
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Sort by scores
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Pick the detection with highest score
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
                
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            
            # Calculate intersection area
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate union area
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = current_area + remaining_areas - intersection
            
            # Calculate IoU
            iou = intersection / union
            
            # Keep detections with IoU below threshold
            indices = indices[1:][iou < threshold]
            
        return keep
        
    def detect_potholes(self, frame):
        """Detect potholes in frame using sliding window"""
        height, width = frame.shape[:2]
        detections = []
        
        # Slide window across the frame
        for y in range(0, height - self.window_size[1], self.stride):
            for x in range(0, width - self.window_size[0], self.stride):
                # Extract window
                window = frame[y:y+self.window_size[1], x:x+self.window_size[0]]
                
                # Skip if window is too small
                if window.shape[0] < self.window_size[1] or window.shape[1] < self.window_size[0]:
                    continue
                    
                # Classify window
                confidence = self.classify_window(window)
                
                # Add detection if confidence is high enough
                if confidence > self.confidence_threshold:
                    detections.append({
                        'bbox': (x, y, x + self.window_size[0], y + self.window_size[1]),
                        'confidence': confidence,
                        'center': (x + self.window_size[0]//2, y + self.window_size[1]//2)
                    })
        
        # Apply non-maximum suppression
        if detections:
            boxes = [d['bbox'] for d in detections]
            scores = [d['confidence'] for d in detections]
            
            keep_indices = self.non_maximum_suppression(boxes, scores, self.nms_threshold)
            detections = [detections[i] for i in keep_indices]
            
        return detections
        
    def draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            center = detection['center']
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            # Draw confidence score
            text = f"Pothole: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (bbox[0], bbox[1] - text_size[1] - 10), 
                         (bbox[0] + text_size[0], bbox[1]), (0, 0, 255), -1)
            cv2.putText(frame, text, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            
        # Add detection count
        count_text = f"Potholes detected: {len(detections)}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
        
    def run_real_time_detection(self, camera_id=0):
        """Run real-time pothole detection on camera feed"""
        print(f"📹 Starting real-time detection on camera {camera_id}")
        print("Press 'q' to quit, 's' to save frame, 'c' to adjust confidence threshold")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
                
            frame_count += 1
            
            # Detect potholes
            detections = self.detect_potholes(frame)
            total_detections += len(detections)
            
            # Draw results
            frame_with_detections = self.draw_detections(frame.copy(), detections)
            
            # Add FPS counter
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame_with_detections, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add threshold info
            cv2.putText(frame_with_detections, f"Threshold: {self.confidence_threshold:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Real-time Pothole Detection', frame_with_detections)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"detection_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_detections)
                print(f"💾 Saved frame: {filename}")
            elif key == ord('c'):
                print(f"Current threshold: {self.confidence_threshold:.2f}")
                try:
                    new_threshold = float(input("Enter new confidence threshold (0.0-1.0): "))
                    if 0.0 <= new_threshold <= 1.0:
                        self.confidence_threshold = new_threshold
                        print(f"✅ Threshold updated to: {self.confidence_threshold:.2f}")
                    else:
                        print("❌ Invalid threshold. Must be between 0.0 and 1.0")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*50)
        print("📊 DETECTION SESSION SUMMARY")
        print("="*50)
        print(f"🎬 Total frames processed: {frame_count}")
        print(f"🕳️ Total detections: {total_detections}")
        print(f"📈 Average detections per frame: {total_detections/frame_count:.2f}")
        print(f"⏱️ Session duration: {elapsed_time:.1f} seconds")
        print(f"🚀 Average FPS: {fps:.1f}")

def main():
    """Main function"""
    print("🔍 Starting Sliding Window Pothole Detection")
    print("=" * 50)
    
    detector = SlidingWindowDetector()
    detector.run_real_time_detection()

if __name__ == "__main__":
    main()