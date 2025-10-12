#!/usr/bin/env python3
"""
ROS Detection Node for Pothole Detection System
Subscribes to camera feed, runs CNN inference, and publishes results
"""

import rospy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import json
import time
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pothole_net import create_model
from utils.dataloader import get_transforms

# Custom message types (would normally be in separate .msg files)
class PotholeDetectionResult:
    """Custom message for pothole detection results"""
    def __init__(self):
        self.header = None
        self.detected = False
        self.confidence = 0.0
        self.severity = 0.0
        self.class_name = ""
        self.bounding_box = []  # [x1, y1, x2, y2]
        self.processing_time = 0.0


class DetectionNode:
    """
    ROS node for pothole detection using CNN
    """
    
    def __init__(self):
        """Initialize detection node"""
        rospy.init_node('detection_node', anonymous=True)
        
        # Parameters
        self.model_path = rospy.get_param('~model_path', '/path/to/model.pth')
        self.model_type = rospy.get_param('~model_type', 'potholeNet')
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        self.input_size = rospy.get_param('~input_size', [224, 224])
        self.device = rospy.get_param('~device', 'auto')
        self.task = rospy.get_param('~task', 'classification')
        self.publish_visualization = rospy.get_param('~publish_visualization', True)
        
        # Initialize components
        self.bridge = CvBridge()
        self.model = None
        self.transform = None
        self.class_names = ['No Pothole', 'Pothole']
        self.colors = [(0, 255, 0), (0, 0, 255)]
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.last_log_time = time.time()
        
        # Setup device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.load_model()
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, 
                                        self.image_callback, queue_size=1)
        self.compressed_sub = rospy.Subscriber('/camera/image_compressed', CompressedImage, 
                                             self.compressed_callback, queue_size=1)
        
        # Publishers
        self.detection_pub = rospy.Publisher('/pothole_detection/results', 
                                           String, queue_size=10)
        self.confidence_pub = rospy.Publisher('/pothole_detection/confidence', 
                                            Float32, queue_size=10)
        self.severity_pub = rospy.Publisher('/pothole_detection/severity', 
                                          Float32, queue_size=10)
        self.detected_pub = rospy.Publisher('/pothole_detection/detected', 
                                          Bool, queue_size=10)
        
        if self.publish_visualization:
            self.viz_pub = rospy.Publisher('/pothole_detection/visualization', 
                                         Image, queue_size=1)
        
        rospy.loginfo(f"Detection node initialized")
        rospy.loginfo(f"Model: {self.model_type}, Device: {self.device}")
        rospy.loginfo(f"Task: {self.task}, Confidence threshold: {self.confidence_threshold}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                rospy.logerr(f"Model file not found: {self.model_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model
            if self.task == 'classification':
                self.model = create_model(
                    model_type=self.model_type,
                    num_classes=2,
                    input_size=tuple(self.input_size),
                    with_depth=False
                )
            else:
                self.model = create_model(
                    model_type='detector',
                    num_classes=2,
                    input_size=tuple(self.input_size)
                )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = get_transforms('val', tuple(self.input_size), self.task)
            
            rospy.loginfo(f"Model loaded successfully from {self.model_path}")
            rospy.loginfo(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            return False
    
    def preprocess_image(self, cv_image):
        """Preprocess image for inference"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            transformed = self.transform(image=image_rgb)
            image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            rospy.logerr(f"Failed to preprocess image: {e}")
            return None
    
    def run_inference(self, input_tensor):
        """Run model inference"""
        try:
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(input_tensor)
                processing_time = time.time() - start_time
            
            return outputs, processing_time
            
        except Exception as e:
            rospy.logerr(f"Inference failed: {e}")
            return None, 0
    
    def process_classification_results(self, outputs, processing_time):
        """Process classification results"""
        try:
            if isinstance(outputs, dict):
                classification_output = outputs['classification']
                severity_output = outputs.get('severity', None)
            else:
                classification_output = outputs
                severity_output = None
            
            # Get probabilities and predictions
            probabilities = F.softmax(classification_output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            results = {
                'detected': int(predicted_class.item()) == 1,  # 1 = pothole
                'confidence': float(confidence.item()),
                'predicted_class': int(predicted_class.item()),
                'class_name': self.class_names[int(predicted_class.item())],
                'probabilities': probabilities.squeeze().cpu().numpy().tolist(),
                'processing_time': processing_time,
                'timestamp': rospy.Time.now().to_sec()
            }
            
            if severity_output is not None:
                results['severity'] = float(severity_output.item())
            else:
                results['severity'] = 0.0
            
            return results
            
        except Exception as e:
            rospy.logerr(f"Failed to process classification results: {e}")
            return None
    
    def process_detection_results(self, outputs, processing_time):
        """Process detection results"""
        try:
            # This would be implemented based on the specific detection model output
            # For now, return a simplified result
            results = {
                'detected': False,
                'confidence': 0.0,
                'severity': 0.0,
                'detections': [],
                'processing_time': processing_time,
                'timestamp': rospy.Time.now().to_sec()
            }
            
            return results
            
        except Exception as e:
            rospy.logerr(f"Failed to process detection results: {e}")
            return None
    
    def publish_results(self, results):
        """Publish detection results"""
        try:
            # Publish detection result as JSON string
            result_json = json.dumps(results)
            self.detection_pub.publish(String(data=result_json))
            
            # Publish individual values
            self.confidence_pub.publish(Float32(data=results['confidence']))
            self.severity_pub.publish(Float32(data=results.get('severity', 0.0)))
            self.detected_pub.publish(Bool(data=results['detected']))
            
        except Exception as e:
            rospy.logerr(f"Failed to publish results: {e}")
    
    def create_visualization(self, cv_image, results):
        """Create visualization of detection results"""
        try:
            vis_image = cv_image.copy()
            h, w = vis_image.shape[:2]
            
            if self.task == 'classification':
                # Draw classification results
                class_name = results.get('class_name', 'Unknown')
                confidence = results.get('confidence', 0.0)
                severity = results.get('severity', 0.0)
                
                # Background for text
                cv2.rectangle(vis_image, (10, 10), (400, 120), (0, 0, 0), -1)
                
                # Text
                color = self.colors[results.get('predicted_class', 0)]
                cv2.putText(vis_image, f"Class: {class_name}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(vis_image, f"Confidence: {confidence:.3f}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, f"Severity: {severity:.3f}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            else:
                # Draw detection results (bounding boxes)
                detections = results.get('detections', [])
                for detection in detections:
                    # This would draw bounding boxes for detected potholes
                    pass
            
            # Add processing time and frame count
            cv2.putText(vis_image, f"Processing: {results['processing_time']*1000:.1f}ms", 
                       (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_image, f"Frame: {self.frame_count}", 
                       (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return vis_image
            
        except Exception as e:
            rospy.logerr(f"Failed to create visualization: {e}")
            return cv_image
    
    def image_callback(self, msg):
        """Callback for raw image messages"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image(cv_image, msg.header)
            
        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")
    
    def compressed_callback(self, msg):
        """Callback for compressed image messages"""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                self.process_image(cv_image, msg.header)
            
        except Exception as e:
            rospy.logerr(f"Compressed callback error: {e}")
    
    def process_image(self, cv_image, header):
        """Main image processing function"""
        if self.model is None:
            rospy.logwarn("Model not loaded, skipping detection")
            return
        
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(cv_image)
            if input_tensor is None:
                return
            
            # Run inference
            outputs, processing_time = self.run_inference(input_tensor)
            if outputs is None:
                return
            
            # Process results based on task
            if self.task == 'classification':
                results = self.process_classification_results(outputs, processing_time)
            else:
                results = self.process_detection_results(outputs, processing_time)
            
            if results is None:
                return
            
            # Publish results
            self.publish_results(results)
            
            # Create and publish visualization
            if self.publish_visualization:
                vis_image = self.create_visualization(cv_image, results)
                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
                    vis_msg.header = header
                    self.viz_pub.publish(vis_msg)
                except Exception as e:
                    rospy.logwarn(f"Failed to publish visualization: {e}")
            
            # Update statistics
            self.frame_count += 1
            self.total_processing_time += processing_time
            
            # Log performance periodically
            current_time = time.time()
            if current_time - self.last_log_time >= 10.0:  # Every 10 seconds
                avg_processing_time = self.total_processing_time / self.frame_count
                rospy.loginfo(f"Processed {self.frame_count} frames, "
                             f"Avg processing time: {avg_processing_time*1000:.1f}ms")
                self.last_log_time = current_time
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def run(self):
        """Run the detection node"""
        rospy.loginfo("Detection node running...")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Detection node shutting down...")


def main():
    """Main function"""
    try:
        detection_node = DetectionNode()
        detection_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Detection node interrupted")
    except Exception as e:
        rospy.logerr(f"Detection node error: {e}")


if __name__ == '__main__':
    main()