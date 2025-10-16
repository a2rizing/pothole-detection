#!/usr/bin/env python3
"""
Brown/Grey Pothole Detection - For black background
Detects brown sand and grey rocks (pothole materials)
"""
import cv2
import numpy as np
import time

class ColorPotholeDemo:
    def __init__(self):
        self.frame_count = 0
        self.pothole_detections = 0
        
    def detect_pothole_colors(self, frame):
        """Detect brown sand and grey rocks on black background"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Brown range (sand) - adjust these values if needed!
        brown_lower = np.array([10, 30, 30])    
        brown_upper = np.array([30, 255, 200])
        
        # Grey range (rocks) - adjust these values if needed!
        grey_lower = np.array([0, 0, 60])       
        grey_upper = np.array([180, 50, 180])
        
        # Create masks
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        grey_mask = cv2.inRange(hsv, grey_lower, grey_upper)
        
        # Combine (pothole = brown OR grey)
        combined_mask = cv2.bitwise_or(brown_mask, grey_mask)
        
        # Clean up noise
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size range for pothole (adjust if needed!)
            if 300 < area < 100000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Accept various shapes
                if 0.2 < aspect_ratio < 5.0:
                    detections.append({
                        'bbox': (x, y, w, h), 
                        'area': area
                    })
        
        return detections, combined_mask
    
    def run_demo(self, camera_id=0):
        print("=" * 50)
        print("🚗 COLOR POTHOLE DETECTION")
        print("Detecting: BROWN sand + GREY rocks")
        print("=" * 50)
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Camera ready!")
        print("\n🎮 Controls:")
        print("   'q' - Quit")
        print("   's' - Save frame")
        print("   'm' - Toggle mask view")
        print("\n🚀 Starting detection...\n")
        
        start_time = time.time()
        show_mask = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            self.frame_count += 1
            
            # Detect potholes
            detections, mask = self.detect_pothole_colors(frame)
            
            # Draw detections
            for det in detections:
                x, y, w, h = det['bbox']
                
                # Red bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                
                # Label
                label = f"POTHOLE {det['area']:.0f}px"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if len(detections) > 0:
                    print(f"🚨 Frame {self.frame_count}: {len(detections)} pothole(s) detected!")
            
            # Status overlay
            if len(detections) > 0:
                status = f"POTHOLES: {len(detections)}"
                color = (0, 0, 255)
            else:
                status = "NO POTHOLES"
                color = (0, 255, 0)
            
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # FPS
            fps = self.frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Frame count
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show frame or mask
            if show_mask:
                # Show color detection mask (for debugging)
                mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow('Pothole Detection', mask_color)
            else:
                cv2.imshow('Pothole Detection', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopping demo...")
                break
            elif key == ord('s'):
                filename = f"pothole_frame_{self.frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
            elif key == ord('m'):
                show_mask = not show_mask
                print(f"👁️  Mask view: {'ON' if show_mask else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print("📊 DEMO SUMMARY")
        print("=" * 50)
        print(f"⏱️  Duration: {elapsed:.1f} seconds")
        print(f"📹 Total frames: {self.frame_count}")
        print(f"🎬 Average FPS: {self.frame_count/elapsed:.1f}")
        print("=" * 50)

if __name__ == "__main__":
    demo = ColorPotholeDemo()
    demo.run_demo(camera_id=0)
