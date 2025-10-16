#!/usr/bin/env python3
"""
Simple Pothole Detection Demo - No ML Required!
Detects dark regions (potholes) in camera feed
Perfect for controlled demo environments
"""

import cv2
import numpy as np
import time
import os

class SimplePotholeDemo:
    def __init__(self):
        self.frame_count = 0
        self.pothole_detections = 0
        
    def detect_dark_regions(self, frame):
        """Detect dark regions that could be potholes"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # Threshold to find dark regions (potholes are darker than road)
        # Adjust threshold value (50-80) based on your lighting
        _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (adjust min/max for your pothole)
            if 500 < area < 50000:  # Adjust these values!
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (not too thin/wide)
                aspect_ratio = w / float(h)
                if 0.3 < aspect_ratio < 3.0:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(area / 10000, 1.0)  # Fake confidence
                    })
        
        return detections
    
    def run_demo(self, camera_id=0):
        """Run simple pothole detection demo"""
        print("=" * 50)
        print("🚗 SIMPLE POTHOLE DETECTION DEMO")
        print("=" * 50)
        print("📹 Opening camera...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Camera ready!")
        print("\n🎮 Controls:")
        print("   'q' - Quit")
        print("   's' - Save frame")
        print("   '+' - Increase threshold (lighter)")
        print("   '-' - Decrease threshold (darker)")
        print("\n🚀 Starting detection...\n")
        
        threshold = 70  # Adjustable threshold
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Detect potholes
                detections = self.detect_dark_regions(frame)
                
                # Draw detections
                for det in detections:
                    x, y, w, h = det['bbox']
                    confidence = det['confidence']
                    
                    # Draw bounding box
                    color = (0, 0, 255)  # Red
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw label
                    label = f"POTHOLE {confidence:.2f}"
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    self.pothole_detections += 1
                
                # Status overlay
                if len(detections) > 0:
                    status = f"POTHOLES DETECTED: {len(detections)}"
                    color = (0, 0, 255)
                    print(f"🚨 Frame {self.frame_count}: {len(detections)} pothole(s) detected!")
                else:
                    status = "NO POTHOLES"
                    color = (0, 255, 0)
                
                cv2.putText(frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # FPS counter
                elapsed = time.time() - start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Threshold info
                cv2.putText(frame, f"Threshold: {threshold}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Frame counter
                cv2.putText(frame, f"Frame: {self.frame_count}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Pothole Detection Demo', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Stopping demo...")
                    break
                elif key == ord('s'):
                    filename = f"pothole_frame_{self.frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Saved: {filename}")
                elif key == ord('+') or key == ord('='):
                    threshold = min(threshold + 5, 150)
                    print(f"🔆 Threshold increased: {threshold}")
                elif key == ord('-') or key == ord('_'):
                    threshold = max(threshold - 5, 20)
                    print(f"🔅 Threshold decreased: {threshold}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Summary
            elapsed = time.time() - start_time
            print("\n" + "=" * 50)
            print("📊 DEMO SUMMARY")
            print("=" * 50)
            print(f"⏱️  Duration: {elapsed:.1f} seconds")
            print(f"📹 Total frames: {self.frame_count}")
            print(f"🚨 Pothole detections: {self.pothole_detections}")
            print(f"🎬 Average FPS: {self.frame_count/elapsed:.1f}")
            print("=" * 50)

if __name__ == "__main__":
    demo = SimplePotholeDemo()
    demo.run_demo(camera_id=0)
