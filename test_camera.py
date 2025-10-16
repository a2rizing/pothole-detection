#!/usr/bin/env python3
"""Quick camera test for debugging"""
import cv2
import time

print("🎥 Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened")
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Add text overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Camera Test', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n🛑 Quit pressed")
            break
        
        # Print status every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"📊 Frame {frame_count} | FPS: {fps:.1f}")
        
except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n✅ Test complete: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
