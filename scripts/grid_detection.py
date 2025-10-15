#!/usr/bin/env python3
"""
🎯 GRID-BASED OBJECT DETECTION
=============================
Fast object detection by dividing frame into grid cells
Uses your classification model without retraining

Usage: python scripts/grid_detection.py
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

class GridDetector:
    def __init__(self, model_path='models/pothole_detection_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Grid parameters
        self.grid_rows = 4
        self.grid_cols = 6
        self.confidence_threshold = 0.75
        
        print(f"🎯 Initializing Grid-Based Detector...")
        print(f"📱 Device: {self.device}")
        print(f"🔢 Grid: {self.grid_rows}x{self.grid_cols}")
        
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
                
            self.model = SimplePotholeNet()
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model loaded successfully!")
            else:
                self.model.load_state_dict(checkpoint)
                print("✅ Model loaded successfully!")
                
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("💡 Will use mock predictions")
            
    def preprocess_cell(self, cell):
        """Preprocess grid cell for classification"""
        # Resize to model input size
        cell_resized = cv2.resize(cell, (224, 224))
        
        # Convert BGR to RGB
        cell_rgb = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        cell_normalized = cell_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        cell_tensor = torch.FloatTensor(cell_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return cell_tensor
        
    def classify_cell(self, cell):
        """Classify a grid cell"""
        try:
            if self.model_loaded and MODEL_AVAILABLE:
                cell_tensor = self.preprocess_cell(cell)
                
                with torch.no_grad():
                    outputs = self.model(cell_tensor.to(self.device))
                    probabilities = torch.softmax(outputs, dim=1)
                    pothole_prob = probabilities[0][1].item()
                    
                return pothole_prob
            else:
                # Mock prediction
                return np.random.uniform(0.0, 1.0)
                
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return 0.0
            
    def detect_in_grid(self, frame):
        """Detect potholes using grid-based approach"""
        height, width = frame.shape[:2]
        
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        detections = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate cell boundaries
                y1 = row * cell_height
                y2 = (row + 1) * cell_height if row < self.grid_rows - 1 else height
                x1 = col * cell_width
                x2 = (col + 1) * cell_width if col < self.grid_cols - 1 else width
                
                # Extract cell
                cell = frame[y1:y2, x1:x2]
                
                # Classify cell
                confidence = self.classify_cell(cell)
                
                # Add detection if confidence is high enough
                if confidence > self.confidence_threshold:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': (x1 + (x2-x1)//2, y1 + (y2-y1)//2),
                        'grid_pos': (row, col)
                    })
                    
        return detections
        
    def draw_grid_and_detections(self, frame, detections):
        """Draw grid and detection results"""
        height, width = frame.shape[:2]
        
        cell_height = height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # Draw grid lines
        for i in range(1, self.grid_rows):
            y = i * cell_height
            cv2.line(frame, (0, y), (width, y), (128, 128, 128), 1)
            
        for i in range(1, self.grid_cols):
            x = i * cell_width
            cv2.line(frame, (x, 0), (x, height), (128, 128, 128), 1)
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            center = detection['center']
            grid_pos = detection['grid_pos']
            
            # Highlight detection cell
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            
            # Draw confidence score
            text = f"P: {confidence:.2f}"
            cv2.putText(frame, text, (bbox[0] + 5, bbox[1] + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw grid position
            pos_text = f"({grid_pos[0]},{grid_pos[1]})"
            cv2.putText(frame, pos_text, (bbox[0] + 5, bbox[1] + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw center point
            cv2.circle(frame, center, 8, (0, 255, 0), -1)
            
        # Add detection count
        count_text = f"Grid Detections: {len(detections)}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
        
    def run_real_time_detection(self, camera_id=0):
        """Run real-time grid-based detection"""
        print(f"📹 Starting real-time grid detection on camera {camera_id}")
        print("Press 'q' to quit, 's' to save frame, '+'/'-' to adjust threshold")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
            detections = self.detect_in_grid(frame)
            total_detections += len(detections)
            
            # Draw results
            frame_with_detections = self.draw_grid_and_detections(frame.copy(), detections)
            
            # Add FPS counter
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame_with_detections, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add threshold info
            cv2.putText(frame_with_detections, f"Threshold: {self.confidence_threshold:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Grid-Based Pothole Detection', frame_with_detections)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"grid_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_detections)
                print(f"💾 Saved frame: {filename}")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                print(f"📈 Threshold increased to: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
                print(f"📉 Threshold decreased to: {self.confidence_threshold:.2f}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*50)
        print("📊 GRID DETECTION SESSION SUMMARY")
        print("="*50)
        print(f"🎬 Total frames processed: {frame_count}")
        print(f"🕳️ Total detections: {total_detections}")
        print(f"📈 Average detections per frame: {total_detections/frame_count:.2f}")
        print(f"⏱️ Session duration: {elapsed_time:.1f} seconds")
        print(f"🚀 Average FPS: {fps:.1f}")

def main():
    """Main function"""
    print("🎯 Starting Grid-Based Pothole Detection")
    print("=" * 50)
    
    detector = GridDetector()
    detector.run_real_time_detection()

if __name__ == "__main__":
    main()