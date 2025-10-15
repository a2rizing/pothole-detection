#!/usr/bin/env python3
"""
Complete Raspberry Pi Pothole Detection System
Integrates camera, detection, GPS, and mapping
"""

import rospy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import Point
import json
import time
import os
import sys
import sqlite3
import datetime
import folium
import serial
import threading
from pathlib import Path

# Model definition (included to avoid import issues)
class SimplePotholeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimplePotholeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class GPSReader:
    """GPS data reader for Raspberry Pi"""
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.current_location = {'lat': 0.0, 'lon': 0.0, 'timestamp': None}
        self.running = False
        
    def start(self):
        """Start GPS reading thread"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.gps_thread = threading.Thread(target=self._read_gps)
            self.gps_thread.daemon = True
            self.gps_thread.start()
            print(f"GPS reader started on {self.port}")
        except Exception as e:
            print(f"GPS connection failed: {e}")
            # Use dummy GPS for testing
            self.current_location = {'lat': 37.7749, 'lon': -122.4194, 'timestamp': datetime.datetime.now()}
    
    def _read_gps(self):
        """Read GPS data continuously"""
        while self.running:
            try:
                if self.serial_conn:
                    line = self.serial_conn.readline().decode('ascii', errors='replace')
                    if line.startswith('$GPGGA'):
                        # Parse NMEA GPS data
                        parts = line.split(',')
                        if len(parts) > 6 and parts[2] and parts[4]:
                            lat = float(parts[2][:2]) + float(parts[2][2:]) / 60
                            if parts[3] == 'S':
                                lat = -lat
                            lon = float(parts[4][:3]) + float(parts[4][3:]) / 60
                            if parts[5] == 'W':
                                lon = -lon
                            self.current_location = {
                                'lat': lat,
                                'lon': lon,
                                'timestamp': datetime.datetime.now()
                            }
                time.sleep(0.1)
            except Exception as e:
                print(f"GPS read error: {e}")
                time.sleep(1)
    
    def get_location(self):
        """Get current GPS location"""
        return self.current_location
    
    def stop(self):
        """Stop GPS reading"""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()

class PotholeDatabase:
    """SQLite database for storing pothole detections"""
    def __init__(self, db_path='pothole_detections.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                latitude REAL,
                longitude REAL,
                confidence REAL,
                grid_position TEXT,
                frame_path TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_detection(self, lat, lon, confidence, grid_pos, frame_path=None):
        """Save pothole detection to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (timestamp, latitude, longitude, confidence, grid_position, frame_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.datetime.now(), lat, lon, confidence, grid_pos, frame_path))
        conn.commit()
        conn.close()
    
    def get_all_detections(self):
        """Get all detections from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM detections ORDER BY timestamp DESC')
        results = cursor.fetchall()
        conn.close()
        return results

class PiPotholeDetector:
    """Main Raspberry Pi Pothole Detection System"""
    
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = SimplePotholeNet(num_classes=2)
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Model loaded from {model_path}")
            else:
                print(f"Model file not found: {model_path}")
                print("Using untrained model - please copy your trained model!")
        except Exception as e:
            print(f"Error loading model: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize components
        self.gps = GPSReader()
        self.database = PotholeDatabase()
        
        # Detection parameters
        self.confidence_threshold = 0.75
        self.grid_size = (3, 4)  # rows, cols
        self.input_size = (224, 224)
        
        # Statistics
        self.total_frames = 0
        self.frames_with_potholes = 0
        self.total_potholes = 0
        
        # Create output directories
        os.makedirs('detection_output', exist_ok=True)
        os.makedirs('maps', exist_ok=True)
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        image = cv2.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        return image.to(self.device)
    
    def detect_potholes_grid(self, frame):
        """Detect potholes using grid-based approach"""
        detections = []
        h, w = frame.shape[:2]
        
        # Focus on bottom 2/3 of frame (road area)
        road_frame = frame[h//3:, :]
        road_h, road_w = road_frame.shape[:2]
        
        rows, cols = self.grid_size
        cell_h = road_h // rows
        cell_w = road_w // cols
        
        for i in range(rows):
            for j in range(cols):
                y1 = i * cell_h
                y2 = min((i + 1) * cell_h, road_h)
                x1 = j * cell_w
                x2 = min((j + 1) * cell_w, road_w)
                
                cell = road_frame[y1:y2, x1:x2]
                
                if cell.shape[0] > 50 and cell.shape[1] > 50:  # Skip tiny cells
                    # Preprocess and predict
                    input_tensor = self.preprocess_image(cell)
                    
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probs = F.softmax(outputs, dim=1)
                        confidence = probs[0][1].item()  # Pothole confidence
                    
                    if confidence > self.confidence_threshold:
                        # Convert back to original frame coordinates
                        abs_y1 = y1 + h//3
                        abs_y2 = y2 + h//3
                        abs_x1 = x1
                        abs_x2 = x2
                        
                        detection = {
                            'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                            'confidence': confidence,
                            'grid_pos': f"R{i}C{j}",
                            'position': 'LEFT' if j == 0 else 'RIGHT' if j == cols-1 else 'CENTER'
                        }
                        detections.append(detection)
        
        return detections
    
    def save_detection_frame(self, frame, detections):
        """Save frame with detection annotations"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_output/pothole_detection_{timestamp}.jpg"
        
        # Draw detections
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw confidence text
            text = f"Pothole {confidence:.2f}"
            cv2.putText(annotated_frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imwrite(filename, annotated_frame)
        return filename
    
    def create_map_visualization(self):
        """Create HTML map with all pothole detections"""
        detections = self.database.get_all_detections()
        
        if not detections:
            print("No detections to map")
            return None
        
        # Calculate center point
        lats = [det[2] for det in detections]
        lons = [det[3] for det in detections]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
        
        # Add markers for each detection
        for det in detections:
            timestamp, lat, lon, confidence, grid_pos = det[1:6]
            
            # Color based on confidence
            color = 'red' if confidence > 0.85 else 'orange' if confidence > 0.75 else 'yellow'
            
            folium.Marker(
                [lat, lon],
                popup=f"Pothole detected<br>Time: {timestamp}<br>Confidence: {confidence:.2f}<br>Grid: {grid_pos}",
                tooltip=f"Pothole ({confidence:.2f})",
                icon=folium.Icon(color=color, icon='exclamation-triangle')
            ).add_to(m)
        
        # Save map
        map_filename = f"maps/pothole_map_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        m.save(map_filename)
        print(f"Map saved: {map_filename}")
        return map_filename
    
    def run_detection(self, camera_id=0):
        """Run live pothole detection"""
        print("Starting Raspberry Pi Pothole Detection System...")
        
        # Start GPS
        self.gps.start()
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        print("Detection started. Press 'q' to quit, 's' to save frame, 'm' to create map")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                self.total_frames += 1
                
                # Detect potholes
                detections = self.detect_potholes_grid(frame)
                
                # Process detections
                pothole_count = len(detections)
                has_potholes = pothole_count > 0
                
                if has_potholes:
                    self.frames_with_potholes += 1
                    self.total_potholes += pothole_count
                    
                    # Get GPS location
                    location = self.gps.get_location()
                    
                    # Save to database
                    for det in detections:
                        self.database.save_detection(
                            location['lat'], location['lon'], 
                            det['confidence'], det['grid_pos']
                        )
                    
                    print(f"🚨 POTHOLES DETECTED: {pothole_count} potholes found!")
                    for det in detections:
                        print(f"   - {det['position']} lane, confidence: {det['confidence']:.3f}")
                
                # Draw live annotations
                display_frame = frame.copy()
                
                # Draw detections
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f"{det['confidence']:.2f}"
                    cv2.putText(display_frame, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw status info
                status_text = f"Potholes: {'YES' if has_potholes else 'NO'} | Count: {pothole_count}"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not has_potholes else (0, 0, 255), 2)
                
                stats_text = f"Total: {self.total_potholes} potholes in {self.frames_with_potholes}/{self.total_frames} frames"
                cv2.putText(display_frame, stats_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # GPS info
                location = self.gps.get_location()
                gps_text = f"GPS: {location['lat']:.6f}, {location['lon']:.6f}"
                cv2.putText(display_frame, gps_text, (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show frame
                cv2.imshow('Pi Pothole Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and has_potholes:
                    filename = self.save_detection_frame(frame, detections)
                    print(f"Frame saved: {filename}")
                elif key == ord('m'):
                    map_file = self.create_map_visualization()
                    if map_file:
                        print(f"Map created: {map_file}")
                
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.gps.stop()
            
            # Final statistics
            print(f"\n=== DETECTION SUMMARY ===")
            print(f"Total frames processed: {self.total_frames}")
            print(f"Frames with potholes: {self.frames_with_potholes}")
            print(f"Total potholes detected: {self.total_potholes}")
            print(f"Detection rate: {(self.frames_with_potholes/self.total_frames)*100:.1f}%")
            
            # Create final map
            final_map = self.create_map_visualization()
            if final_map:
                print(f"Final map: {final_map}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Raspberry Pi Pothole Detection System')
    parser.add_argument('--model', default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    args = parser.parse_args()
    
    detector = PiPotholeDetector(model_path=args.model)
    detector.run_detection(camera_id=args.camera)

if __name__ == '__main__':
    main()