# USB Camera Deployment Checklist

## Pre-Deployment Testing

### 1. Camera Detection
```bash
# List connected USB devices
lsusb

# Check video devices
ls /dev/video*

# Test camera with v4l2-ctl
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### 2. Camera Testing Utility
```bash
# Test your USB camera
python3 scripts/test_usb_camera.py --list
python3 scripts/test_usb_camera.py --test 0 --width 1920 --height 1080
python3 scripts/test_usb_camera.py --settings 0
python3 scripts/test_usb_camera.py --capture 0 --count 5
```

### 3. Performance Verification
- [ ] Camera detected at 1920x1080 resolution
- [ ] Achieving ≥20 FPS for real-time processing
- [ ] MJPG codec working properly
- [ ] Auto-exposure and brightness controls functional
- [ ] Test images show good quality for pothole detection

## ROS System Testing

### 4. Individual Node Testing
```bash
# Terminal 1: Start roscore
roscore

# Terminal 2: Test camera node
rosrun pothole_detection camera_node.py

# Terminal 3: Check camera topics
rostopic list
rostopic hz /camera/image_raw
rostopic echo /camera/camera_info

# Test image display
rosrun image_view image_view image:=/camera/image_raw
```

### 5. Detection Pipeline Testing
```bash
# Terminal 1: roscore
roscore

# Terminal 2: Camera node
rosrun pothole_detection camera_node.py

# Terminal 3: Detection node (after training model)
rosrun pothole_detection detection_node.py

# Terminal 4: GPS node
rosrun pothole_detection gps_node.py

# Terminal 5: Check detection topics
rostopic echo /pothole_detection/results
```

## Hardware Setup Verification

### 6. Raspberry Pi Optimization
- [ ] USB 3.0 ports used for camera (blue ports)
- [ ] Adequate power supply (≥3A for Pi 4B + USB camera)
- [ ] Camera mount secure and stable
- [ ] SD card class 10 or better for data logging
- [ ] GPS module connected via UART/USB

### 7. Camera Mounting
- [ ] Camera positioned for road surface view
- [ ] Mounting stable during vehicle movement
- [ ] Lens clean and protected from weather
- [ ] USB cable secured and protected
- [ ] No vibration affecting image quality

## Software Configuration

### 8. Configuration Files
- [ ] `config/raspberry_pi_config.yaml` - USB camera settings verified
- [ ] `config/ros_config.yaml` - 1920x1080 resolution configured
- [ ] `config/model_config.yaml` - Model optimized for Pi performance

### 9. Model Training & Deployment
- [ ] Dataset collected and labeled for local road conditions
- [ ] Model trained and validated (accuracy >85%)
- [ ] Model converted to optimized format (TensorRT/ONNX if needed)
- [ ] Inference speed ≥15 FPS on target Raspberry Pi

## Deployment Steps

### 10. Final Deployment
```bash
# Install and setup
bash scripts/setup_raspberry_pi.sh

# Test complete system
python3 scripts/test_usb_camera.py
roslaunch pothole_detection full_system.launch

# Start monitoring
python3 -c "
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

def image_callback(msg):
    print(f'Camera: {msg.width}x{msg.height}')

def detection_callback(msg):
    print(f'Detection: {msg.data}')

rospy.init_node('monitor')
rospy.Subscriber('/camera/image_raw', Image, image_callback)
rospy.Subscriber('/pothole_detection/results', String, detection_callback)
rospy.spin()
"
```

## Troubleshooting Checklist

### 11. Common Issues
- [ ] **No camera detected**: Check USB connection, try different port
- [ ] **Low FPS**: Reduce resolution or switch to MJPG codec
- [ ] **Poor image quality**: Adjust brightness/contrast, clean lens
- [ ] **ROS node crashes**: Check permissions, camera not in use by other apps
- [ ] **High CPU usage**: Enable hardware acceleration, optimize model

### 12. Performance Monitoring
```bash
# Monitor system resources
htop

# Check camera performance
python3 scripts/test_usb_camera.py --test 0 --duration 60

# Monitor ROS topics
rostopic hz /camera/image_raw
rostopic bw /camera/image_raw
```

## Production Readiness

### 13. Final Checks
- [ ] System runs stable for ≥30 minutes continuous operation
- [ ] Memory usage stable (no memory leaks)
- [ ] All error conditions handled gracefully
- [ ] Logging configured for debugging
- [ ] Auto-start configured for deployment

### 14. Documentation
- [ ] Hardware setup documented
- [ ] Software configuration documented  
- [ ] Troubleshooting guide created
- [ ] Performance benchmarks recorded

## Post-Deployment

### 15. Monitoring & Maintenance
- [ ] Log rotation configured
- [ ] System health monitoring setup
- [ ] Regular model retraining scheduled
- [ ] Hardware maintenance schedule established

---

**Ready for Deployment**: Check all boxes above before deploying to production vehicle.