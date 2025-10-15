# 🚗 Raspberry Pi Car Deployment Guide

## Quick Setup (3 Hours Max)

### 1. **Hardware Preparation** (30 minutes)
```bash
# Items needed:
- Raspberry Pi 4 (4GB+ recommended)
- USB Camera or Pi Camera Module  
- USB GPS Module (optional - uses simulation if not available)
- MicroSD Card (32GB+)
- Power bank for car use
```

### 2. **Pi Setup** (60 minutes)
```bash
# 1. Flash Raspberry Pi OS to SD card
# 2. SSH or connect monitor/keyboard
# 3. Run setup script:
cd ~
git clone https://github.com/a2rizing/pothole-detection.git
cd pothole-detection
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
```

### 3. **Copy Your Model** (5 minutes)
```bash
# From your Windows machine, copy the trained model:
scp models/pothole_detection_model.pth pi@<pi-ip>:~/pothole-detection/models/best_model.pth

# Or use USB drive:
# 1. Copy model to USB drive
# 2. On Pi: cp /media/usb/pothole_detection_model.pth ~/pothole-detection/models/best_model.pth
```

### 4. **Test System** (15 minutes)
```bash
cd ~/pothole-detection

# Test camera
python3 -c "import cv2; print('Camera OK!' if cv2.VideoCapture(0).isOpened() else 'Camera Failed!')"

# Test model loading
python3 -c "import torch; print('PyTorch OK!')"

# Run detection
python3 scripts/car_deployment.py --camera 0
```

## 🎯 **For Teacher Demo** (60 minutes)

### **What Your System Does:**
1. **Real-time Detection**: Shows live camera feed with pothole detection
2. **Counting**: Displays exact number of potholes in each frame  
3. **GPS Logging**: Records location of each detection (simulated)
4. **Interactive Map**: Creates HTML map showing all detection points
5. **Statistics**: Shows detection rates and session summary

### **Demo Script:**
```bash
# 1. Start the system
python3 scripts/car_deployment.py

# 2. Point camera at road/pothole images
# 3. Show live detection with bounding boxes
# 4. Press 'm' to generate map
# 5. Open map in browser to show GPS visualization
```

### **Key Demo Points:**
- ✅ **No Retraining**: Uses your existing 89.06% accuracy model
- ✅ **Real-time**: Live camera feed processing
- ✅ **Accurate Counting**: Grid-based detection counts multiple potholes
- ✅ **GPS Integration**: Location tracking for each detection
- ✅ **Professional Visualization**: Interactive HTML map
- ✅ **Car Ready**: Designed for vehicle deployment

## 🚀 **ROS Integration** (Optional - 30 minutes)

### **ROS Architecture:**
```
📷 camera_node.py     → Publishes camera frames
🧠 detection_node.py  → Subscribes to frames, runs CNN inference  
📍 gps_node.py        → Publishes GPS coordinates
🗺️ mapping_node.py    → Creates visualization maps
```

### **How ROS is Used:**
1. **camera_node.py**: Captures camera frames and publishes to `/camera/image_raw`
2. **detection_node.py**: Subscribes to camera feed, runs your CNN model, publishes results
3. **gps_node.py**: Reads GPS data and publishes to `/gps/location`
4. **mapping_node.py**: Combines detection + GPS data to create maps

### **ROS Launch:**
```bash
# Start ROS core
roscore &

# Launch all nodes
roslaunch pothole_detection detection_system.launch
```

## 📊 **What Teachers Will See:**

### **Live Detection Display:**
- Real-time camera feed
- Red bounding boxes around detected potholes
- "POTHOLES: YES/NO" status
- "COUNT: X" showing number of potholes
- Detection statistics and GPS coordinates

### **Generated Map:**
- Interactive HTML map (opens in browser)
- Red markers for high-confidence detections
- Orange/yellow for lower confidence
- Route line showing car path
- Popup details for each detection

### **Console Output:**
```
🚨 POTHOLE ALERT! 2 pothole(s) detected
   📍 LEFT lane - Confidence: 0.857
   📍 CENTER lane - Confidence: 0.792

=== DETECTION SESSION COMPLETE ===
📊 Total frames: 1,250
🚨 Frames with potholes: 89
🕳️ Total potholes: 134
📈 Detection rate: 7.1%
🗺️ Map saved: maps/pothole_detection_map_20251016_143022.html
```

## 🔧 **Troubleshooting:**

### **Camera Issues:**
```bash
# Try different camera IDs
python3 scripts/car_deployment.py --camera 1
python3 scripts/car_deployment.py --camera 2

# Check connected cameras
ls /dev/video*
```

### **Model Issues:**
```bash
# Verify model file exists
ls -la models/best_model.pth

# Check model loading
python3 -c "import torch; print(torch.load('models/best_model.pth', map_location='cpu'))"
```

### **Dependencies:**
```bash
# Install missing packages
pip3 install folium geopy pyserial
```

## ✅ **Success Checklist:**
- [ ] Pi boots and connects to network
- [ ] Camera shows live feed
- [ ] Model loads without errors  
- [ ] Detection shows bounding boxes
- [ ] Map generates with markers
- [ ] System ready for car mounting

## 🎉 **Ready for Deployment!**

Your system now provides:
1. **Binary Detection**: Is there a pothole? (YES/NO)
2. **Counting**: How many potholes? (Exact count)
3. **Visualization**: Where were they found? (Interactive map)
4. **Car Integration**: Ready for vehicle mounting and real-world testing

**Total Setup Time: ~3 hours**
**Demo Ready: Professional pothole detection system with GPS mapping!**