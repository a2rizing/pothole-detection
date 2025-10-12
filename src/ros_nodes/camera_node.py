#!/usr/bin/env python3
"""
ROS Camera Node for Pothole Detection System
Captures and publishes camera frames
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import threading
import time

class CameraNode:
    """
    ROS node for camera data acquisition and publishing
    """
    
    def __init__(self):
        """Initialize camera node"""
        rospy.init_node('camera_node', anonymous=True)
        
        # Parameters
        self.camera_id = rospy.get_param('~camera_id', 0)
        self.frame_rate = rospy.get_param('~frame_rate', 30)
        self.image_width = rospy.get_param('~image_width', 1920)
        self.image_height = rospy.get_param('~image_height', 1080)
        self.compressed = rospy.get_param('~compressed', True)
        self.publish_raw = rospy.get_param('~publish_raw', True)
        self.fourcc = rospy.get_param('~fourcc', 'MJPG')
        self.buffer_size = rospy.get_param('~buffer_size', 1)
        
        # Camera settings for USB camera
        self.auto_exposure = rospy.get_param('~auto_exposure', True)
        self.brightness = rospy.get_param('~brightness', 0)
        self.contrast = rospy.get_param('~contrast', 0)
        self.saturation = rospy.get_param('~saturation', 0)
        self.hue = rospy.get_param('~hue', 0)
        
        # Initialize camera
        self.cap = None
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        # Publishers
        if self.compressed:
            self.image_pub = rospy.Publisher('/camera/image_compressed', 
                                           CompressedImage, queue_size=1)
        
        if self.publish_raw:
            self.raw_image_pub = rospy.Publisher('/camera/image_raw', 
                                               Image, queue_size=1)
        
        # Status publisher
        self.status_pub = rospy.Publisher('/camera/status', 
                                        Image, queue_size=1)  # Using Image msg for simplicity
        
        # Initialize camera
        self.initialize_camera()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.daemon = True
        self.running = True
        
        rospy.loginfo(f"USB Camera node initialized with camera {self.camera_id}")
        rospy.loginfo(f"Publishing at {self.frame_rate} FPS")
        rospy.loginfo(f"Resolution: {self.image_width}x{self.image_height}")
        rospy.loginfo(f"FOURCC: {self.fourcc}, Compressed: {self.compressed}")
    
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                rospy.logerr(f"Cannot open USB camera {self.camera_id}")
                return False
            
            # Set camera properties for USB camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
            
            # Set FOURCC codec for better performance with USB cameras
            if self.fourcc:
                fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # USB camera specific settings
            if hasattr(cv2, 'CAP_PROP_AUTO_EXPOSURE'):
                if self.auto_exposure:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure on
                else:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
            
            # Adjust image quality settings
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.saturation)
            self.cap.set(cv2.CAP_PROP_HUE, self.hue)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            
            # Convert fourcc back to string for logging
            fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            rospy.loginfo(f"USB Camera settings - Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            rospy.loginfo(f"FOURCC: {fourcc_str}, Buffer size: {self.buffer_size}")
            
            # Warm up camera (some USB cameras need this)
            rospy.loginfo("Warming up USB camera...")
            for _ in range(10):
                ret, frame = self.cap.read()
                if ret:
                    break
                time.sleep(0.1)
            
            if not ret:
                rospy.logerr("Failed to get initial frame from USB camera")
                return False
            
            rospy.loginfo("USB camera initialized successfully")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize USB camera: {e}")
            return False
    
    def capture_loop(self):
        """Main capture loop running in separate thread"""
        rate = rospy.Rate(self.frame_rate)
        
        while self.running and not rospy.is_shutdown():
            try:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    rospy.logwarn("Failed to capture frame from USB camera")
                    # Try to reinitialize camera if multiple failures
                    if hasattr(self, '_consecutive_failures'):
                        self._consecutive_failures += 1
                    else:
                        self._consecutive_failures = 1
                    
                    if self._consecutive_failures > 10:
                        rospy.logerr("Too many consecutive failures, reinitializing camera...")
                        self.cap.release()
                        time.sleep(1)
                        if self.initialize_camera():
                            self._consecutive_failures = 0
                        else:
                            rospy.logerr("Failed to reinitialize camera")
                            break
                    continue
                else:
                    self._consecutive_failures = 0
                
                # Create timestamp
                timestamp = rospy.Time.now()
                
                # Publish compressed image
                if self.compressed:
                    self.publish_compressed_image(frame, timestamp)
                
                # Publish raw image
                if self.publish_raw:
                    self.publish_raw_image(frame, timestamp)
                
                # Update counters
                self.frame_count += 1
                self.fps_counter += 1
                
                # Log FPS periodically
                current_time = time.time()
                if current_time - self.last_fps_time >= 5.0:  # Every 5 seconds
                    fps = self.fps_counter / (current_time - self.last_fps_time)
                    rospy.loginfo(f"Camera FPS: {fps:.1f}, Total frames: {self.frame_count}")
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"Error in capture loop: {e}")
                break
    
    def publish_compressed_image(self, frame, timestamp):
        """Publish compressed image"""
        try:
            # Encode image to JPEG
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            # Create compressed image message
            msg = CompressedImage()
            msg.header.stamp = timestamp
            msg.header.frame_id = "camera"
            msg.format = "jpeg"
            msg.data = buffer.tobytes()
            
            self.image_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish compressed image: {e}")
    
    def publish_raw_image(self, frame, timestamp):
        """Publish raw image"""
        try:
            # Convert to ROS Image message
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = timestamp
            msg.header.frame_id = "camera"
            
            self.raw_image_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish raw image: {e}")
    
    def start(self):
        """Start the camera node"""
        if self.cap is None or not self.cap.isOpened():
            rospy.logerr("Camera not initialized")
            return
        
        # Start capture thread
        self.capture_thread.start()
        
        # Keep node alive
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Camera node shutting down...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Cleanup resources"""
        rospy.loginfo("Shutting down camera node...")
        
        self.running = False
        
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        rospy.loginfo("Camera node shutdown complete")


def main():
    """Main function"""
    try:
        camera_node = CameraNode()
        camera_node.start()
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera node interrupted")
    except Exception as e:
        rospy.logerr(f"Camera node error: {e}")


if __name__ == '__main__':
    main()