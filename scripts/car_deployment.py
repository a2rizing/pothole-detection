#!/usr/bin/env python3
"""
Simplified Raspberry Pi Pothole Detection System (No ROS)
Complete standalone system for car deployment
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import os
import sqlite3
import datetime
import folium
import serial
import threading
from pathlib import Path

# Model definition
class SimplePotholeNet(nn.Module):
    """Lightweight CNN for pothole detection"""
    
    def __init__(self, num_classes=2):
        super(SimplePotholeNet, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CarPotholeDetector:
    """Simplified Raspberry Pi Pothole Detection for Car Deployment"""
    
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize statistics first (before model loading)
        self.reset_stats()
        
        # Load model
        self.model = SimplePotholeNet(num_classes=2)
        self.model_loaded = False
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"✅ Model loaded from {model_path}")
                self.model_loaded = True
            else:
                print(f"❌ Model file not found: {model_path}")
                print("Please copy your trained model to the Pi!")
                print("⚠️  Running in demo mode without detection...")
                self.model_loaded = False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Running in demo mode without detection...")
            self.model_loaded = False
        
        self.model.to(self.device)
        if self.model_loaded:
            self.model.eval()
        
        # Detection parameters
        self.confidence_threshold = 0.75
        self.grid_size = (3, 4)  # rows, cols
        self.input_size = (224, 224)
        
        # Create output directories
        os.makedirs('detection_output', exist_ok=True)
        os.makedirs('maps', exist_ok=True)
        
        # Database for storing detections
        self.init_database()
        
        # Initialize GPS
        self.init_gps()
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.total_frames = 0
        self.frames_with_potholes = 0
        self.total_potholes = 0
        self.session_start = time.time()
    
    def init_database(self):
        """Initialize SQLite database for detections"""
        conn = sqlite3.connect('pothole_detections.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                latitude REAL,
                longitude REAL,
                confidence REAL,
                grid_position TEXT,
                pothole_count INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def init_gps(self):
        """Initialize GPS module (Neo 6M)"""
        self.gps_enabled = False
        self.current_gps = {'lat': 37.7749, 'lon': -122.4194}  # Default coordinates
        
        try:
            # Try to connect to GPS module on Raspberry Pi
            self.gps_serial = serial.Serial('/dev/ttyS0', 9600, timeout=1)
            self.gps_enabled = True
            print("📡 GPS Module connected!")
        except Exception as e:
            print(f"📡 GPS not available: {e}")
            print("📍 Using simulated GPS coordinates")
            self.gps_serial = None
    
    def read_gps(self):
        """Read GPS coordinates from Neo 6M module"""
        if not self.gps_enabled or not self.gps_serial:
            return self.current_gps
        
        try:
            # Read NMEA sentence
            line = self.gps_serial.readline().decode('ascii', errors='ignore')
            
            # Parse GPGGA sentence (Global Positioning System Fix Data)
            if line.startswith('$GPGGA'):
                parts = line.split(',')
                if len(parts) > 6 and parts[2] and parts[4]:
                    # Convert latitude
                    lat_raw = float(parts[2])
                    lat_deg = int(lat_raw / 100)
                    lat_min = lat_raw - (lat_deg * 100)
                    lat = lat_deg + (lat_min / 60)
                    if parts[3] == 'S':
                        lat = -lat
                    
                    # Convert longitude
                    lon_raw = float(parts[4])
                    lon_deg = int(lon_raw / 100)
                    lon_min = lon_raw - (lon_deg * 100)
                    lon = lon_deg + (lon_min / 60)
                    if parts[5] == 'W':
                        lon = -lon
                    
                    self.current_gps = {'lat': lat, 'lon': lon}
                    
        except Exception as e:
            # Keep using last known coordinates
            pass
        
        return self.current_gps
    
    def save_detection(self, lat, lon, confidence, grid_pos, count):
        """Save detection to database"""
        conn = sqlite3.connect('pothole_detections.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (timestamp, latitude, longitude, confidence, grid_position, pothole_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.datetime.now(), lat, lon, confidence, grid_pos, count))
        conn.commit()
        conn.close()
    
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
        
        # If model not loaded, return empty detections (demo mode)
        if not self.model_loaded:
            return detections
            
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
                
                if cell.shape[0] > 50 and cell.shape[1] > 50:
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
    
    def create_detection_map(self):
        """Create HTML map with all detections"""
        conn = sqlite3.connect('pothole_detections.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM detections ORDER BY timestamp DESC')
        detections = cursor.fetchall()
        conn.close()
        
        if not detections:
            print("No detections to map")
            return None
        
        # Calculate center point
        lats = [det[2] for det in detections]
        lons = [det[3] for det in detections]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16,
                      tiles='OpenStreetMap')
        
        # Add title
        title_html = '''
        <h3 align="center" style="font-size:20px"><b>🚗 Pothole Detection Results</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add markers for each detection
        for i, det in enumerate(detections):
            _, timestamp, lat, lon, confidence, grid_pos, count = det
            
            # Color based on confidence
            if confidence > 0.85:
                color = 'red'
                icon = 'exclamation-triangle'
            elif confidence > 0.75:
                color = 'orange'
                icon = 'warning'
            else:
                color = 'yellow'
                icon = 'info-sign'
            
            folium.Marker(
                [lat, lon],
                popup=f"""
                <b>Pothole Detection #{i+1}</b><br>
                📅 Time: {timestamp}<br>
                🎯 Confidence: {confidence:.2f}<br>
                📍 Grid: {grid_pos}<br>
                🔢 Count: {count}
                """,
                tooltip=f"Pothole ({confidence:.2f})",
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)
        
        # Add route line if enough points
        if len(detections) > 1:
            route_points = [[det[2], det[3]] for det in detections]
            folium.PolyLine(
                route_points,
                weight=3,
                color='blue',
                opacity=0.8,
                popup='Detection Route'
            ).add_to(m)
        
        # Save map
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        map_filename = f"maps/pothole_detection_map_{timestamp}.html"
        m.save(map_filename)
        print(f"🗺️  Map saved: {map_filename}")
        return map_filename
    
    def run_live_detection(self, camera_id=0, headless=False):
        """Run live pothole detection on camera feed"""
        print("🚗 Starting Car Pothole Detection System...")
        print("📹 Initializing camera...")
        
        # Check if running headless (no display)
        if headless or os.environ.get('DISPLAY') is None:
            print("🖥️  Running in headless mode (no window display)")
            import cv2
            cv2.namedWindow = lambda *args, **kwargs: None  # Disable window creation
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            print("🔧 Try different camera IDs: 0, 1, 2...")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📷 Camera initialized: {width}x{height} @ {actual_fps} FPS")
        print("\n🎮 Controls:")
        print("   'q' - Quit")
        print("   's' - Save current frame") 
        print("   'm' - Create detection map")
        print("   'r' - Reset statistics")
        print("\n🚀 Starting detection...\n")
        
        frame_time = 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                self.total_frames += 1
                
                # Detect potholes
                detections = self.detect_potholes_grid(frame)
                
                # Process results
                pothole_count = len(detections)
                has_potholes = pothole_count > 0
                
                if has_potholes:
                    self.frames_with_potholes += 1
                    self.total_potholes += pothole_count
                    
                    # Get current GPS coordinates
                    current_gps = self.read_gps()
                    
                    # Save to database with real GPS
                    for det in detections:
                        self.save_detection(
                            current_gps['lat'],
                            current_gps['lon'],
                            det['confidence'],
                            det['grid_pos'],
                            pothole_count
                        )
                    
                    # Print detection alert
                    print(f"🚨 POTHOLE ALERT! {pothole_count} pothole(s) detected")
                    for det in detections:
                        print(f"   📍 {det['position']} lane - Confidence: {det['confidence']:.3f}")
                
                # Create display frame
                display_frame = frame.copy()
                
                # Draw grid overlay
                h, w = frame.shape[:2]
                road_h = (h * 2) // 3
                rows, cols = self.grid_size
                cell_h = road_h // rows
                cell_w = w // cols
                
                # Draw grid lines
                for i in range(1, rows):
                    y = h//3 + i * cell_h
                    cv2.line(display_frame, (0, y), (w, y), (100, 100, 100), 1)
                for j in range(1, cols):
                    x = j * cell_w
                    cv2.line(display_frame, (x, h//3), (x, h), (100, 100, 100), 1)
                
                # Draw road area boundary
                cv2.line(display_frame, (0, h//3), (w, h//3), (255, 255, 0), 2)
                
                # Draw detections
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Confidence text
                    text = f"{det['confidence']:.2f}"
                    cv2.putText(display_frame, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Status overlay
                status_color = (0, 0, 255) if has_potholes else (0, 255, 0)
                status_text = f"POTHOLES: {'YES' if has_potholes else 'NO'} | COUNT: {pothole_count}"
                cv2.putText(display_frame, status_text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                
                # Statistics
                if self.total_frames > 0:
                    detection_rate = (self.frames_with_potholes / self.total_frames) * 100
                    stats_text = f"Session: {self.total_potholes} total | {detection_rate:.1f}% frames"
                    cv2.putText(display_frame, stats_text, (10, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # FPS counter
                fps_text = f"FPS: {1/frame_time:.1f}" if frame_time > 0 else "FPS: --"
                cv2.putText(display_frame, fps_text, (10, height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # GPS info (real or simulated)
                gps_status = "📡 LIVE" if self.gps_enabled else "📍 SIM"
                current_gps = self.read_gps()
                gps_text = f"GPS {gps_status}: {current_gps['lat']:.4f}, {current_gps['lon']:.4f}"
                cv2.putText(display_frame, gps_text, (10, height-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show frame (only if not headless)
                if not headless and os.environ.get('DISPLAY'):
                    cv2.imshow('🚗 Pi Pothole Detection', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n🛑 Stopping detection...")
                        break
                    elif key == ord('s'):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"detection_output/frame_{timestamp}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"💾 Frame saved: {filename}")
                    elif key == ord('m'):
                        map_file = self.create_detection_map()
                    elif key == ord('r'):
                        self.reset_stats()
                        print("📊 Statistics reset")
                else:
                    # Headless mode - just print status
                    if has_potholes:
                        print(f"🚨 FRAME {self.total_frames}: {pothole_count} pothole(s) detected")
                    
                    # Auto-quit after some frames in headless mode for testing
                    if self.total_frames >= 100:  # Run for 100 frames then stop
                        print(f"\n🛑 Headless demo complete after {self.total_frames} frames")
                        break
                
                # Calculate frame time
                frame_time = time.time() - start_time
                
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        
        finally:
            # Cleanup
            cap.release()
            if not headless and os.environ.get('DISPLAY'):
                cv2.destroyAllWindows()
            
            # Close GPS connection
            if hasattr(self, 'gps_serial') and self.gps_serial:
                self.gps_serial.close()
                print("📡 GPS connection closed")
            
            # Final summary
            total_time = time.time() - self.session_start
            avg_fps = self.total_frames / total_time if total_time > 0 else 0
            
            print(f"\n{'='*50}")
            print(f"🎯 DETECTION SESSION COMPLETE")
            print(f"{'='*50}")
            print(f"⏱️  Session duration: {total_time:.1f} seconds")
            print(f"📊 Total frames: {self.total_frames}")
            print(f"🚨 Frames with potholes: {self.frames_with_potholes}")
            print(f"🕳️  Total potholes: {self.total_potholes}")
            print(f"📈 Detection rate: {(self.frames_with_potholes/self.total_frames)*100:.1f}%")
            print(f"🎬 Average FPS: {avg_fps:.1f}")
            
            # Create final map
            if self.total_potholes > 0:
                print(f"\n🗺️  Creating final detection map...")
                final_map = self.create_detection_map()
                print(f"✅ Ready for teacher demo: {final_map}")
            else:
                print(f"\n📍 No potholes detected in this session")
            
            return True

def main():
    """Main function for car deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='🚗 Car Pothole Detection System')
    parser.add_argument('--model', default='models/best_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera ID (try 0, 1, 2 if camera not working)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display window (for SSH/remote)')
    args = parser.parse_args()
    
    print("🚗 Car Pothole Detection System")
    print("="*40)
    
    detector = CarPotholeDetector(model_path=args.model)
    success = detector.run_live_detection(camera_id=args.camera, headless=args.headless)
    
    if success:
        print("\n✅ System ready for car deployment!")
    else:
        print("\n❌ Setup incomplete. Check camera and model file.")

if __name__ == '__main__':
    main()