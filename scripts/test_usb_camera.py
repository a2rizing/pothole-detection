#!/usr/bin/env python3
"""
USB Camera Testing Utility for Pothole Detection System
Tests external 1080p USB camera functionality and performance
"""

import cv2
import numpy as np
import time
import argparse
import sys

def list_cameras():
    """List available cameras"""
    print("Scanning for available cameras...")
    available_cameras = []
    
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc >> 8 * j) & 0xFF) for j in range(4)])
                
                available_cameras.append({
                    'id': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'fourcc': fourcc_str
                })
                
                print(f"Camera {i}: {width}x{height} @ {fps} FPS, FOURCC: {fourcc_str}")
            cap.release()
    
    if not available_cameras:
        print("No cameras found!")
        return None
    
    return available_cameras

def test_camera_performance(camera_id=0, width=1920, height=1080, duration=30):
    """Test camera performance"""
    print(f"Testing camera {camera_id} at {width}x{height} for {duration} seconds...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Try to set MJPG codec for better performance
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Verify settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"Actual settings: {actual_width}x{actual_height} @ {actual_fps} FPS, FOURCC: {fourcc_str}")
    
    # Warm up camera
    print("Warming up camera...")
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frames during warmup")
            cap.release()
            return False
    
    # Performance test
    frame_count = 0
    start_time = time.time()
    processing_times = []
    
    print("Starting performance test...")
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            processing_time = time.time() - loop_start
            processing_times.append(processing_time)
            
            # Display frame (optional)
            if frame_count % 30 == 0:  # Every 30 frames
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                avg_processing_time = np.mean(processing_times[-30:]) * 1000
                
                print(f"Frame {frame_count}: {current_fps:.1f} FPS, "
                      f"Avg processing: {avg_processing_time:.1f}ms")
                
                # Optionally show frame
                cv2.imshow('Camera Test', cv2.resize(frame, (640, 360)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    avg_processing_time = np.mean(processing_times) * 1000
    max_processing_time = np.max(processing_times) * 1000
    min_processing_time = np.min(processing_times) * 1000
    
    print(f"\n=== Performance Results ===")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average processing time: {avg_processing_time:.2f}ms")
    print(f"Min processing time: {min_processing_time:.2f}ms")
    print(f"Max processing time: {max_processing_time:.2f}ms")
    
    # Performance assessment
    if avg_fps >= 25:
        print("✅ Excellent performance for real-time processing")
    elif avg_fps >= 20:
        print("✅ Good performance, suitable for pothole detection")
    elif avg_fps >= 15:
        print("⚠️  Acceptable performance, may need optimization")
    else:
        print("❌ Poor performance, consider reducing resolution or frame rate")
    
    return True

def test_camera_settings(camera_id=0):
    """Test different camera settings"""
    print(f"Testing various settings for camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return False
    
    # Test different resolutions
    resolutions = [
        (1920, 1080),  # 1080p
        (1280, 720),   # 720p
        (640, 480),    # VGA
        (320, 240)     # QVGA
    ]
    
    print("\nTesting resolutions:")
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ret, frame = cap.read()
        if ret:
            print(f"  {width}x{height} -> {actual_width}x{actual_height} ✅")
        else:
            print(f"  {width}x{height} -> Failed ❌")
    
    # Test camera controls
    print("\nTesting camera controls:")
    controls = [
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness", -100, 100),
        (cv2.CAP_PROP_CONTRAST, "Contrast", -100, 100),
        (cv2.CAP_PROP_SATURATION, "Saturation", -100, 100),
        (cv2.CAP_PROP_HUE, "Hue", -180, 180),
        (cv2.CAP_PROP_GAIN, "Gain", 0, 100),
        (cv2.CAP_PROP_EXPOSURE, "Exposure", -15, 15),
    ]
    
    for prop, name, min_val, max_val in controls:
        try:
            current_val = cap.get(prop)
            # Test setting a value
            test_val = (min_val + max_val) / 2
            cap.set(prop, test_val)
            new_val = cap.get(prop)
            
            if abs(new_val - test_val) < 1:
                print(f"  {name}: ✅ (Current: {current_val}, Test: {new_val})")
            else:
                print(f"  {name}: ⚠️  (Current: {current_val}, Read-only or limited)")
            
            # Reset to original value
            cap.set(prop, current_val)
            
        except Exception as e:
            print(f"  {name}: ❌ Error: {e}")
    
    cap.release()
    return True

def capture_test_images(camera_id=0, width=1920, height=1080, count=5):
    """Capture test images for quality assessment"""
    print(f"Capturing {count} test images...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Warm up
    for _ in range(10):
        cap.read()
    
    for i in range(count):
        ret, frame = cap.read()
        if ret:
            filename = f"test_image_{i+1}_{width}x{height}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        else:
            print(f"Failed to capture image {i+1}")
    
    cap.release()
    print(f"Test images saved. Check image quality for pothole detection suitability.")
    return True

def main():
    parser = argparse.ArgumentParser(description='USB Camera Testing Utility')
    parser.add_argument('--list', action='store_true', help='List available cameras')
    parser.add_argument('--test', type=int, help='Test camera performance (camera ID)')
    parser.add_argument('--settings', type=int, help='Test camera settings (camera ID)')
    parser.add_argument('--capture', type=int, help='Capture test images (camera ID)')
    parser.add_argument('--width', type=int, default=1920, help='Camera width')
    parser.add_argument('--height', type=int, default=1080, help='Camera height')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    parser.add_argument('--count', type=int, default=5, help='Number of test images')
    
    args = parser.parse_args()
    
    if args.list:
        list_cameras()
    elif args.test is not None:
        test_camera_performance(args.test, args.width, args.height, args.duration)
    elif args.settings is not None:
        test_camera_settings(args.settings)
    elif args.capture is not None:
        capture_test_images(args.capture, args.width, args.height, args.count)
    else:
        # Run all tests
        print("Running comprehensive camera tests...")
        cameras = list_cameras()
        if cameras:
            camera_id = cameras[0]['id']
            test_camera_settings(camera_id)
            test_camera_performance(camera_id, args.width, args.height, 10)
            capture_test_images(camera_id, args.width, args.height, 3)

if __name__ == "__main__":
    main()