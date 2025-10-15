#!/usr/bin/env python3
"""
Live Pothole Detection System
Simple real-time detection using existing classification model
Shows: Pothole presence (Yes/No) + Count
"""

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import model - handle different possible locations
try:
    from src.models.simple_cnn import SimplePotholeNet
except ImportError:
    # Try alternative import paths
    try:
        import sys
        sys.path.append(str(parent_dir / "src"))
        from models.simple_cnn import SimplePotholeNet
    except ImportError:
        # Define model inline if import fails
        import torch.nn as nn
        
        class SimplePotholeNet(nn.Module):
            def __init__(self, num_classes=2):
                super(SimplePotholeNet, self).__init__()
                
                # Feature extractor
                self.features = nn.Sequential(
                    # First conv block
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Second conv block
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Third conv block
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Fourth conv block
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(256 * 7 * 7, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

class LivePotholeDetector:
    def __init__(self, model_path, confidence_threshold=0.75):
        """
        Initialize the live pothole detector
        
        Args:
            model_path: Path to trained model
            confidence_threshold: Minimum confidence for detection
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = SimplePotholeNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Statistics
        self.total_frames = 0
        self.frames_with_potholes = 0
        self.total_potholes = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def analyze_frame(self, frame):
        """
        Analyze frame using grid-based approach
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (has_potholes, pothole_count, detections)
        """
        height, width = frame.shape[:2]
        
        # Grid configuration - simple 3x3 grid
        grid_rows, grid_cols = 3, 3
        cell_height = height // grid_rows
        cell_width = width // grid_cols
        
        detections = []
        pothole_count = 0
        
        # Analyze each grid cell
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Extract cell
                y1 = row * cell_height
                y2 = (row + 1) * cell_height if row < grid_rows - 1 else height
                x1 = col * cell_width
                x2 = (col + 1) * cell_width if col < grid_cols - 1 else width
                
                cell = frame[y1:y2, x1:x2]
                
                # Predict
                confidence = self.predict_cell(cell)
                
                if confidence > self.confidence_threshold:
                    pothole_count += 1
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'position': f"Row{row+1}_Col{col+1}"
                    })
        
        has_potholes = pothole_count > 0
        return has_potholes, pothole_count, detections
    
    def predict_cell(self, cell):
        """
        Predict pothole probability for a cell
        
        Args:
            cell: Image cell
            
        Returns:
            float: Confidence score
        """
        try:
            # Preprocess
            if cell.size == 0:
                return 0.0
                
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(cell_rgb).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                pothole_prob = probabilities[0][1].item()  # Class 1 = pothole
                
            return pothole_prob
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0
    
    def draw_detections(self, frame, detections, has_potholes, pothole_count):
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            detections: List of detection results
            has_potholes: Boolean indicating presence
            pothole_count: Number of potholes
            
        Returns:
            frame: Annotated frame
        """
        # Create copy for drawing
        result_frame = frame.copy()
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw rectangle
            color = (0, 0, 255)  # Red for potholes
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add confidence text
            conf_text = f"{confidence:.2f}"
            cv2.putText(result_frame, conf_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Status overlay
        self.draw_status_overlay(result_frame, has_potholes, pothole_count)
        
        return result_frame
    
    def draw_status_overlay(self, frame, has_potholes, pothole_count):
        """Draw status information overlay"""
        height, width = frame.shape[:2]
        
        # Status box background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status_text = "🔴 POTHOLES DETECTED" if has_potholes else "✅ ROAD CLEAR"
        status_color = (0, 0, 255) if has_potholes else (0, 255, 0)
        
        cv2.putText(frame, status_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Count and stats
        cv2.putText(frame, f"Pothole Count: {pothole_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Confidence Threshold: {self.confidence_threshold:.2f}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        current_fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Calculate every 30 frames
            current_time = time.time()
            fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            self.fps_counter = 0
            return fps
        return 0.0
    
    def run_live_detection(self, source=0):
        """
        Run live detection from camera or video
        
        Args:
            source: Camera index (0) or video file path
        """
        print(f"🎥 Starting live detection from source: {source}")
        
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {source}")
            return
        
        print("📹 Camera opened successfully!")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  '+' - Increase sensitivity")
        print("  '-' - Decrease sensitivity")
        print()
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📺 End of video or camera disconnected")
                    break
                
                frame_count += 1
                self.total_frames += 1
                
                # Analyze frame
                start_time = time.time()
                has_potholes, pothole_count, detections = self.analyze_frame(frame)
                processing_time = (time.time() - start_time) * 1000
                
                # Update statistics
                if has_potholes:
                    self.frames_with_potholes += 1
                    self.total_potholes += pothole_count
                
                # Draw results
                result_frame = self.draw_detections(frame, detections, has_potholes, pothole_count)
                
                # Show frame
                cv2.imshow('Live Pothole Detection', result_frame)
                
                # Print periodic status
                if frame_count % 30 == 0:
                    detection_rate = (self.frames_with_potholes / self.total_frames) * 100
                    print(f"Frame {frame_count}: {'🔴 POTHOLES' if has_potholes else '✅ CLEAR'} | "
                          f"Count: {pothole_count} | Processing: {processing_time:.1f}ms | "
                          f"Detection Rate: {detection_rate:.1f}%")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"pothole_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"💾 Frame saved: {filename}")
                elif key == ord('+'):
                    # Increase sensitivity (lower threshold)
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"🔧 Sensitivity increased: {self.confidence_threshold:.2f}")
                elif key == ord('-'):
                    # Decrease sensitivity (higher threshold)
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"🔧 Sensitivity decreased: {self.confidence_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Detection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            self.print_session_summary()
    
    def print_session_summary(self):
        """Print final session statistics"""
        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print("="*50)
        print(f"Total Frames Processed: {self.total_frames}")
        print(f"Frames with Potholes: {self.frames_with_potholes}")
        print(f"Total Potholes Detected: {self.total_potholes}")
        
        if self.total_frames > 0:
            detection_rate = (self.frames_with_potholes / self.total_frames) * 100
            avg_potholes = self.total_potholes / self.total_frames
            print(f"Detection Rate: {detection_rate:.2f}%")
            print(f"Average Potholes per Frame: {avg_potholes:.2f}")
        
        print("="*50)

def main():
    """Main function"""
    # Model path
    model_path = Path("models/pothole_detection_model.pth")
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please ensure the model is trained and saved at the correct location.")
        return
    
    # Initialize detector
    detector = LivePotholeDetector(
        model_path=model_path,
        confidence_threshold=0.75
    )
    
    # Choose source
    print("📹 Live Pothole Detection System")
    print("1. Use default camera (0)")
    print("2. Use video file")
    
    choice = input("Enter choice (1/2) or press Enter for camera: ").strip()
    
    if choice == "2":
        video_path = input("Enter video file path: ").strip()
        if os.path.exists(video_path):
            detector.run_live_detection(video_path)
        else:
            print(f"❌ Video file not found: {video_path}")
    else:
        # Default to camera
        detector.run_live_detection(0)

if __name__ == "__main__":
    main()