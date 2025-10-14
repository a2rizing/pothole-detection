# 🍓 Raspberry Pi Deployment Guide - Complete Setup

## 📋 **Pre-Deployment Checklist**

### ✅ **What to Include in Repository:**
- ✅ All source code (`src/`, `scripts/`, `config/`)
- ✅ Documentation (README.md, guides)  
- ✅ Configuration files (YAML configs)
- ✅ Requirements file
- ❌ **Exclude**: Dataset images (too large)
- ❌ **Exclude**: Trained models (download separately)
- ❌ **Exclude**: Virtual environments

### 🎯 **Repository Status Check:**
```bash
# Before pushing to GitHub, check what will be included:
git status
git add .
git commit -m "Complete pothole detection system ready for Pi deployment"
git push origin main
```

## 🍓 **Raspberry Pi Setup Steps**

### **Step 1: Fresh OS Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git python3-pip python3-venv curl wget

# Install ROS Noetic (if not already installed)
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-noetic-desktop-base ros-noetic-image-transport ros-noetic-cv-bridge

# Install camera and GPS dependencies
sudo apt install -y v4l-utils fswebcam gpsd gpsd-clients python3-serial
```

### **Step 2: Clone Repository**
```bash
# Clone your repository
cd ~/
git clone https://github.com/a2rizing/pothole-detection.git
cd pothole-detection

# Check what was cloned
ls -la
```

### **Step 3: Python Environment Setup**
```bash
# Create virtual environment
python3 -m venv pothole_env
source pothole_env/bin/activate

# Install PyTorch for ARM (Raspberry Pi)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip3 install opencv-python pillow numpy pyyaml tqdm matplotlib seaborn scikit-learn pandas
pip3 install rospkg rospy-message-converter

# For ROS Python packages
pip3 install catkin_pkg empy
```

### **Step 4: Download Trained Model**
```bash
# Create models directory
mkdir -p models

# Option A: Download from your cloud storage (recommended)
# Upload your model to Google Drive/Dropbox and download:
# wget "https://your-cloud-link/pothole_detection_model.pth" -O models/pothole_detection_model.pth

# Option B: Transfer via SCP from your Windows machine
# From Windows: scp models/pothole_detection_model.pth pi@raspberry-pi-ip:~/pothole-detection/models/

# Option C: Re-train on Pi (slower but works)
# You can re-train with the lightweight dataset if needed
```

### **Step 5: Hardware Setup**
```bash
# Test USB camera
lsusb  # Check if camera is detected
v4l2-ctl --list-devices  # List video devices
v4l2-ctl -d /dev/video0 --list-formats-ext  # Check camera capabilities

# Test camera capture
fswebcam -r 1920x1080 --jpeg 90 test_camera.jpg
ls -la test_camera.jpg

# Test GPS (if connected)
sudo systemctl start gpsd
sudo systemctl enable gpsd
cgps  # Should show GPS data
```

### **Step 6: ROS Workspace Setup**
```bash
# Setup ROS environment
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
echo "export ROS_MASTER_URI=http://localhost:11311" >> ~/.bashrc
source ~/.bashrc

# Test ROS installation
roscore &  # Start ROS master
rostopic list  # Should show ROS topics
pkill roscore  # Stop ROS master
```

### **Step 7: Configuration Update**
```bash
# Update camera device ID in config files
nano config/raspberry_pi_config.yaml

# Update paths in ROS config
nano config/ros_config.yaml

# Test configuration
python3 check_training_ready.py  # Verify setup
```

### **Step 8: System Testing**
```bash
# Test individual components
cd ~/pothole-detection

# 1. Test camera node
source pothole_env/bin/activate
roscore &
python3 src/ros_nodes/camera_node.py &

# In another terminal:
rostopic list  # Should see /camera/image_raw
rostopic hz /camera/image_raw  # Check frame rate

# 2. Test GPS node (if hardware available)
python3 src/ros_nodes/gps_node.py &
rostopic echo /gps/fix  # Check GPS data

# 3. Test detection node (requires model)
python3 src/ros_nodes/detection_node.py &
rostopic echo /pothole_detection/results

# Stop all nodes
pkill -f ros_nodes
pkill roscore
```

## 🚀 **Deployment Options**

### **Option A: Quick Development Testing**
```bash
# For development and testing
source pothole_env/bin/activate
cd ~/pothole-detection

# Start system manually
roscore &
sleep 2
python3 src/ros_nodes/camera_node.py &
python3 src/ros_nodes/detection_node.py &
python3 src/ros_nodes/gps_node.py &
python3 src/ros_nodes/mapping_node.py &

# Monitor system
rostopic list
rostopic hz /camera/image_raw
```

### **Option B: Production Deployment**
```bash
# Run the setup script for production
chmod +x scripts/setup_raspberry_pi.sh
sudo bash scripts/setup_raspberry_pi.sh

# This will:
# - Create systemd services
# - Setup auto-start on boot
# - Configure logging
# - Setup monitoring
```

## 📊 **Performance Optimization for Pi**

### **Memory Optimization:**
```bash
# Increase GPU memory split
sudo nano /boot/config.txt
# Add: gpu_mem=128

# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### **Model Optimization:**
```python
# Use the MiniPotholeNet for better Pi performance
# Edit config/model_config.yaml:
# model:
#   type: "mini"  # Instead of "potholeNet"
```

## 🔍 **Troubleshooting Common Issues**

### **Camera Issues:**
```bash
# Check camera permissions
sudo usermod -a -G video $USER
logout  # Then log back in

# Test different camera settings
python3 scripts/test_usb_camera.py --test 0 --width 1280 --height 720
```

### **ROS Issues:**
```bash
# Check ROS environment
printenv | grep ROS
rosnode list
rosnode info /camera_node
```

### **Performance Issues:**
```bash
# Monitor system resources
htop  # Check CPU/Memory usage
iostat 1  # Check disk I/O
vcgencmd measure_temp  # Check temperature
```

## 📈 **Data Collection on Pi**

### **Do You Need More Images?**

**Current Status:** Your model has **89.06% accuracy** - this is excellent!

**Recommendations:**

**🎯 For Development/Testing:** Your current dataset is sufficient
- 89% accuracy is production-ready
- Good generalization across validation/test sets
- Balanced dataset with proper augmentation

**📸 For Real-World Deployment:** Consider collecting local data
```bash
# Collect local road images for fine-tuning
mkdir -p data/local_collection
python3 scripts/collect_local_data.py --output_dir data/local_collection --duration 30

# Fine-tune model with local data if needed
python3 src/train.py --pretrained models/pothole_detection_model.pth --data_dir data/local_collection --epochs 10
```

**When to Collect More Data:**
- ✅ If accuracy drops below 85% in real-world testing
- ✅ If you encounter different road types than in training data
- ✅ If lighting conditions are very different
- ❌ Not needed immediately - test with current model first

## ✅ **Final Deployment Checklist**

```bash
# 1. Repository ready
[ ] .gitignore configured
[ ] Code pushed to GitHub
[ ] Documentation complete

# 2. Pi setup
[ ] Fresh OS installed and updated
[ ] Repository cloned
[ ] Python environment created
[ ] Dependencies installed
[ ] Model downloaded/transferred

# 3. Hardware testing
[ ] Camera working (1080p capture)
[ ] GPS module connected (if available)
[ ] ROS nodes communicating

# 4. System integration
[ ] All nodes starting successfully
[ ] Real-time inference working
[ ] Database logging functional

# 5. Production ready
[ ] Auto-start on boot configured
[ ] Monitoring and logging setup
[ ] Performance optimized for Pi
```

## 🎯 **Next Steps Priority Order:**

1. **📤 Push Code to GitHub** (exclude large files via .gitignore)
2. **🍓 Clone on Raspberry Pi** 
3. **🐍 Setup Python Environment** (Pi-optimized packages)
4. **📱 Transfer/Download Model** (27MB file)
5. **📷 Test Camera System** (1080p USB camera)
6. **🚀 Run System Integration Tests**
7. **📊 Test Real-World Performance** 
8. **🔧 Optimize for Production** (auto-start, monitoring)

**You have everything you need for successful Pi deployment!** Your 89% accuracy model is production-ready. 🎉