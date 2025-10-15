#!/bin/bash
# Raspberry Pi Setup Script for Pothole Detection System

echo "=== Raspberry Pi Pothole Detection Setup ==="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and essential packages
echo "🐍 Installing Python packages..."
sudo apt install -y python3-pip python3-opencv python3-serial python3-dev
sudo apt install -y git curl wget unzip

# Install GPS tools
echo "📍 Installing GPS tools..."
sudo apt install -y gpsd gpsd-clients

# Install ROS Noetic (if needed)
echo "🤖 Installing ROS Noetic..."
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-noetic-desktop-full
sudo apt install -y ros-noetic-cv-bridge ros-noetic-image-transport
sudo apt install -y ros-noetic-sensor-msgs ros-noetic-geometry-msgs

# Source ROS
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install Python packages
echo "📚 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install opencv-python numpy matplotlib folium geopy
pip3 install pyserial pynmea2 gpsd-py3 requests
pip3 install RPi.GPIO picamera

# Setup project
echo "📂 Setting up project..."
cd ~
git clone https://github.com/a2rizing/pothole-detection.git
cd pothole-detection

# Create necessary directories
mkdir -p detection_output maps logs

# Copy model file (you need to copy this manually)
echo "⚠️  IMPORTANT: Copy your trained model to models/best_model.pth"
echo "   You can use scp: scp models/pothole_detection_model.pth pi@<pi-ip>:~/pothole-detection/models/best_model.pth"

# Setup GPS (if USB GPS is used)
echo "📍 Setting up GPS..."
sudo systemctl stop gpsd
sudo systemctl disable gpsd
echo 'DEVICES="/dev/ttyUSB0"' | sudo tee -a /etc/default/gpsd
echo 'GPSD_OPTIONS="-n"' | sudo tee -a /etc/default/gpsd
sudo systemctl enable gpsd
sudo systemctl start gpsd

# Camera permissions
echo "📷 Setting up camera permissions..."
sudo usermod -a -G video $USER

# Make scripts executable
chmod +x scripts/*.py

echo "✅ Setup complete!"
echo ""
echo "🔧 Manual steps needed:"
echo "1. Copy your trained model: scp models/pothole_detection_model.pth pi@<pi-ip>:~/pothole-detection/models/best_model.pth"
echo "2. Connect USB camera and GPS module"
echo "3. Test camera: python3 -c 'import cv2; print(cv2.VideoCapture(0).isOpened())'"
echo "4. Test GPS: gpspipe -r"
echo "5. Run detection: python3 scripts/pi_pothole_detector.py"
echo ""
echo "🚗 Ready for deployment!"