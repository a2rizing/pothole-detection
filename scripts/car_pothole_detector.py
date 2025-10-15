#!/usr/bin/env python3
"""
🚗 CAR-DEPLOYED POTHOLE DETECTION SYSTEM
=======================================
Real-time pothole detection for vehicle deployment
Uses your existing classification model to provide:
1. Pothole presence detection (Yes/No)
2. Pothole count in camera feed
3. Alert system for vehicle control

Usage: python scripts/car_pothole_detector.py
"""

import cv2
import torch
import numpy as np
import sys
import time
import json
from datetime import datetime
import os
from collections import deque
import threading
import queue

# Add src to path
sys.path.append('src')

try:
    from models.pothole_net import SimplePotholeNet, MiniPotholeNet
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️ Model classes not available. Using mock predictions.")

class CarPotholeDetector:
    def __init__(self, model_path='models/pothole_detection_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Detection parameters optimized for car deployment
        self.grid_rows = 3  # Focus on road area
        self.grid_cols = 4  # Good balance of speed vs precision
        self.confidence_threshold = 0.75
        self.min_confidence_for_count = 0.80  # Higher threshold for counting
        
        # Alert system parameters
        self.alert_cooldown = 2.0  # Seconds between alerts
        self.last_alert_time = 0
        self.detection_history = deque(maxlen=10)  # Last 10 frames
        
        # Real-time statistics
        self.session_stats = {
            'total_frames': 0,
            'frames_with_potholes': 0,
            'total_potholes_detected': 0,
            'max_potholes_in_frame': 0,
            'session_start_time': time.time()
        }
        
        print(f"🚗 Initializing Car Pothole Detection System...")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Grid: {self.grid_rows}x{self.grid_cols} (optimized for road)")
        print(f"📊 Confidence threshold: {self.confidence_threshold}")
        
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
                # Mock prediction with road-realistic patterns
                # Simulate less frequent potholes (15% chance)
                return np.random.uniform(0.0, 1.0) if np.random.random() < 0.15 else np.random.uniform(0.0, 0.7)
                
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return 0.0
            
    def analyze_road_frame(self, frame):
        """Analyze road frame for potholes - main detection function"""
        height, width = frame.shape[:2]
        
        # Focus on road area (bottom 2/3 of frame)
        road_start_y = height // 3
        road_frame = frame[road_start_y:, :]
        road_height, road_width = road_frame.shape[:2]
        
        cell_height = road_height // self.grid_rows
        cell_width = road_width // self.grid_cols
        
        detections = []
        high_confidence_detections = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate cell boundaries
                y1 = row * cell_height
                y2 = (row + 1) * cell_height if row < self.grid_rows - 1 else road_height
                x1 = col * cell_width
                x2 = (col + 1) * cell_width if col < self.grid_cols - 1 else road_width
                
                # Extract cell
                cell = road_frame[y1:y2, x1:x2]
                
                # Classify cell
                confidence = self.classify_cell(cell)
                
                # Add detection if confidence is high enough
                if confidence > self.confidence_threshold:
                    detection = {
                        'bbox': (x1, y1 + road_start_y, x2, y2 + road_start_y),  # Adjust for full frame
                        'confidence': confidence,
                        'center': (x1 + (x2-x1)//2, y1 + road_start_y + (y2-y1)//2),
                        'grid_pos': (row, col),
                        'road_position': self.get_road_position(col, self.grid_cols)
                    }
                    detections.append(detection)
                    
                    # Count high-confidence detections
                    if confidence > self.min_confidence_for_count:
                        high_confidence_detections.append(detection)
        
        # Analysis results
        analysis = {
            'has_potholes': len(detections) > 0,
            'pothole_count': len(high_confidence_detections),  # Use high-confidence for counting
            'total_detections': len(detections),
            'detections': detections,
            'high_confidence_detections': high_confidence_detections,
            'max_confidence': max([d['confidence'] for d in detections]) if detections else 0.0,
            'road_safety_level': self.calculate_safety_level(high_confidence_detections)
        }
        
        return analysis
        
    def get_road_position(self, col, total_cols):
        """Get road position description"""
        if col < total_cols // 3:
            return "LEFT"
        elif col >= 2 * total_cols // 3:
            return "RIGHT"
        else:
            return "CENTER"
            
    def calculate_safety_level(self, detections):
        """Calculate road safety level"""
        count = len(detections)
        if count == 0:
            return "SAFE"
        elif count <= 2:
            return "CAUTION"
        elif count <= 4:
            return "WARNING"
        else:
            return "DANGER"
            
    def should_trigger_alert(self, analysis):
        """Determine if alert should be triggered"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
            
        # Alert conditions
        if analysis['pothole_count'] > 0:
            # High confidence potholes detected
            if analysis['max_confidence'] > 0.85 or analysis['pothole_count'] >= 2:
                self.last_alert_time = current_time
                return True
                
        return False
        
    def trigger_vehicle_alert(self, analysis):
        """Trigger vehicle control alerts"""
        safety_level = analysis['road_safety_level']
        pothole_count = analysis['pothole_count']
        
        print(f"🚨 VEHICLE ALERT: {safety_level} - {pothole_count} potholes detected!")
        
        # You can add actual vehicle control logic here:
        # - Slow down vehicle
        # - Activate warning lights
        # - Send GPS coordinates
        # - Alert driver with sound/vibration
        
        if safety_level == "DANGER":
            print("🛑 DANGER: Multiple potholes ahead - SLOW DOWN!")
        elif safety_level == "WARNING":
            print("⚠️ WARNING: Significant road damage detected")
        elif safety_level == "CAUTION":
            print("⚠️ CAUTION: Pothole detected ahead")
            
    def update_statistics(self, analysis):
        """Update session statistics"""
        self.session_stats['total_frames'] += 1
        
        if analysis['has_potholes']:
            self.session_stats['frames_with_potholes'] += 1
            
        if analysis['pothole_count'] > 0:
            self.session_stats['total_potholes_detected'] += analysis['pothole_count']
            self.session_stats['max_potholes_in_frame'] = max(
                self.session_stats['max_potholes_in_frame'], 
                analysis['pothole_count']
            )
            
        # Add to detection history
        self.detection_history.append({
            'timestamp': time.time(),
            'pothole_count': analysis['pothole_count'],
            'has_potholes': analysis['has_potholes'],
            'safety_level': analysis['road_safety_level']
        })
        
    def draw_car_interface(self, frame, analysis):
        """Draw car-friendly interface overlay"""
        height, width = frame.shape[:2]
        
        # Draw road analysis area
        road_start_y = height // 3
        cv2.line(frame, (0, road_start_y), (width, road_start_y), (255, 255, 0), 2)
        cv2.putText(frame, "ROAD ANALYSIS AREA", (10, road_start_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw grid overlay
        self.draw_road_grid(frame, road_start_y)
        
        # Draw detections
        for detection in analysis['detections']:
            bbox = detection['bbox']
            confidence = detection['confidence']
            road_pos = detection['road_position']
            
            # Color based on confidence
            if confidence > self.min_confidence_for_count:
                color = (0, 0, 255)  # Red for high confidence
                thickness = 3
            else:
                color = (0, 165, 255)  # Orange for medium confidence
                thickness = 2
                
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Label
            label = f"{road_pos}: {confidence:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw main status panel
        self.draw_status_panel(frame, analysis)
        
        return frame
        
    def draw_road_grid(self, frame, road_start_y):
        """Draw grid overlay on road area"""
        height, width = frame.shape[:2]
        road_height = height - road_start_y
        
        cell_height = road_height // self.grid_rows
        cell_width = width // self.grid_cols
        
        # Draw grid lines
        for i in range(1, self.grid_rows):
            y = road_start_y + i * cell_height
            cv2.line(frame, (0, y), (width, y), (128, 128, 128), 1)
            
        for i in range(1, self.grid_cols):
            x = i * cell_width
            cv2.line(frame, (x, road_start_y), (x, height), (128, 128, 128), 1)
            
    def draw_status_panel(self, frame, analysis):
        """Draw main status information panel"""
        height, width = frame.shape[:2]
        
        # Status panel background
        panel_height = 120
        cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        # Main status
        has_potholes = "YES" if analysis['has_potholes'] else "NO"
        pothole_count = analysis['pothole_count']
        safety_level = analysis['road_safety_level']
        
        # Color based on safety level
        if safety_level == "SAFE":
            status_color = (0, 255, 0)  # Green
        elif safety_level == "CAUTION":
            status_color = (0, 255, 255)  # Yellow
        elif safety_level == "WARNING":
            status_color = (0, 165, 255)  # Orange
        else:  # DANGER
            status_color = (0, 0, 255)  # Red
        
        # Status text
        cv2.putText(frame, f"POTHOLES: {has_potholes}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"COUNT: {pothole_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"SAFETY: {safety_level}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Session statistics (top right)
        stats_x = width - 300
        cv2.rectangle(frame, (stats_x, 10), (width - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (stats_x, 10), (width - 10, 100), (255, 255, 255), 2)
        
        frames_processed = self.session_stats['total_frames']
        frames_with_potholes = self.session_stats['frames_with_potholes']
        total_potholes = self.session_stats['total_potholes_detected']
        
        cv2.putText(frame, f"FRAMES: {frames_processed}", (stats_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"W/ POTHOLES: {frames_with_potholes}", (stats_x + 10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"TOTAL COUNT: {total_potholes}", (stats_x + 10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS (bottom right)
        elapsed_time = time.time() - self.session_stats['session_start_time']
        fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    def run_car_detection(self, camera_id=0, save_log=True):
        """Run real-time car pothole detection"""
        print(f"🚗 Starting Car Pothole Detection System...")
        print("📹 Camera feed will be analyzed for:")
        print("   1. Pothole presence (Yes/No)")
        print("   2. Number of potholes in view")
        print("   3. Road safety level assessment")
        print("\nControls:")
        print("   'q' - Quit system")
        print("   's' - Save current frame")
        print("   '+'/'-' - Adjust confidence threshold")
        print("   'r' - Reset statistics")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
            
        # Set camera properties for car deployment
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
        
        # Log file for car deployment
        if save_log:
            log_filename = f"car_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            detection_log = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
            
            # Analyze frame
            analysis = self.analyze_road_frame(frame)
            
            # Update statistics
            self.update_statistics(analysis)
            
            # Check for alerts
            if self.should_trigger_alert(analysis):
                self.trigger_vehicle_alert(analysis)
            
            # Draw interface
            frame_with_interface = self.draw_car_interface(frame.copy(), analysis)
            
            # Display
            cv2.imshow('Car Pothole Detection System', frame_with_interface)
            
            # Log data for car deployment
            if save_log:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'has_potholes': analysis['has_potholes'],
                    'pothole_count': analysis['pothole_count'],
                    'safety_level': analysis['road_safety_level'],
                    'max_confidence': analysis['max_confidence']
                }
                detection_log.append(log_entry)
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"car_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_interface)
                print(f"💾 Saved frame: {filename}")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                print(f"📈 Threshold increased to: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
                print(f"📉 Threshold decreased to: {self.confidence_threshold:.2f}")
            elif key == ord('r'):
                self.session_stats = {
                    'total_frames': 0,
                    'frames_with_potholes': 0,
                    'total_potholes_detected': 0,
                    'max_potholes_in_frame': 0,
                    'session_start_time': time.time()
                }
                print("🔄 Statistics reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final log
        if save_log and detection_log:
            with open(log_filename, 'w') as f:
                json.dump(detection_log, f, indent=2)
            print(f"💾 Detection log saved: {log_filename}")
        
        # Print final session summary
        self.print_session_summary()
        
    def print_session_summary(self):
        """Print final session summary"""
        elapsed_time = time.time() - self.session_stats['session_start_time']
        
        print("\n" + "="*60)
        print("🚗 CAR DEPLOYMENT SESSION SUMMARY")
        print("="*60)
        print(f"⏱️ Session duration: {elapsed_time:.1f} seconds")
        print(f"🎬 Total frames processed: {self.session_stats['total_frames']}")
        print(f"🕳️ Frames with potholes: {self.session_stats['frames_with_potholes']}")
        print(f"📊 Total potholes detected: {self.session_stats['total_potholes_detected']}")
        print(f"📈 Max potholes in single frame: {self.session_stats['max_potholes_in_frame']}")
        
        if self.session_stats['total_frames'] > 0:
            detection_rate = (self.session_stats['frames_with_potholes'] / self.session_stats['total_frames']) * 100
            avg_fps = self.session_stats['total_frames'] / elapsed_time
            print(f"📈 Detection rate: {detection_rate:.1f}%")
            print(f"🚀 Average FPS: {avg_fps:.1f}")
        
        print("✅ Car deployment session completed!")

def main():
    """Main function for car deployment"""
    print("🚗 CAR POTHOLE DETECTION SYSTEM")
    print("=" * 50)
    print("This system provides:")
    print("1. ✅ Pothole presence detection (Yes/No)")
    print("2. 🔢 Number of potholes in camera feed")
    print("3. 🚨 Vehicle alert system")
    print("4. 📊 Real-time road safety assessment")
    print()
    
    detector = CarPotholeDetector()
    detector.run_car_detection()

if __name__ == "__main__":
    main()