#!/bin/bash

# Pothole Detection System Setup Script for Raspberry Pi
# This script sets up the complete environment for the pothole detection system

set -e  # Exit on any error

echo "==================================="
echo "Pothole Detection System Setup"
echo "==================================="

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Warning: This script is optimized for Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Variables
PROJECT_DIR="/home/pi/pothole-detection"
VENV_DIR="/home/pi/pothole_env"
ROS_WS="/home/pi/catkin_ws"

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    sqlite3 \
    libsqlite3-dev \
    v4l-utils \
    fswebcam \
    guvcview

# Install ROS Noetic (if not already installed)
if ! command -v roscore &> /dev/null; then
    echo "Installing ROS Noetic..."
    
    # Add ROS repository
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
    
    sudo apt update
    sudo apt install -y ros-noetic-desktop-full
    
    # Initialize rosdep
    sudo rosdep init
    rosdep update
    
    # Add ROS to bashrc
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    
    # Install additional ROS packages
    sudo apt install -y \
        ros-noetic-cv-bridge \
        ros-noetic-image-transport \
        ros-noetic-sensor-msgs \
        ros-noetic-geometry-msgs \
        ros-noetic-std-msgs \
        ros-noetic-navigation \
        ros-noetic-robot-localization
fi

# Create Python virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for Raspberry Pi)
echo "Installing PyTorch for Raspberry Pi..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Clone or update project repository
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning project repository..."
    git clone https://github.com/your-username/pothole-detection.git $PROJECT_DIR
else
    echo "Updating project repository..."
    cd $PROJECT_DIR
    git pull
fi

cd $PROJECT_DIR

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set up ROS workspace
echo "Setting up ROS workspace..."
if [ ! -d "$ROS_WS" ]; then
    mkdir -p $ROS_WS/src
    cd $ROS_WS
    catkin_make
    echo "source $ROS_WS/devel/setup.bash" >> ~/.bashrc
fi

# Create symbolic link to project in ROS workspace
if [ ! -L "$ROS_WS/src/pothole_detection" ]; then
    ln -s $PROJECT_DIR $ROS_WS/src/pothole_detection
fi

# Build ROS workspace
cd $ROS_WS
catkin_make

# Set up camera (USB camera configuration)
echo "Configuring USB camera access..."
sudo usermod -a -G video pi    # For camera access

# Test USB camera availability
echo "Testing USB camera..."
if lsusb | grep -i camera > /dev/null || lsusb | grep -i webcam > /dev/null; then
    echo "USB camera detected!"
    # Test camera capture
    if command -v fswebcam &> /dev/null; then
        echo "Testing camera capture..."
        fswebcam --no-banner -r 1280x720 --jpeg 85 -D 1 test_capture.jpg
        if [ -f "test_capture.jpg" ]; then
            echo "Camera test successful!"
            rm test_capture.jpg
        else
            echo "Camera test failed - check camera connection"
        fi
    else
        echo "Installing fswebcam for camera testing..."
        sudo apt install -y fswebcam
    fi
else
    echo "Warning: No USB camera detected. Please connect your 1080p USB camera."
fi

# Set up serial for GPS (enable UART)
echo "Configuring GPS serial interface..."
sudo raspi-config nonint do_serial 0
echo "enable_uart=1" | sudo tee -a /boot/config.txt

# Create directories
echo "Creating project directories..."
mkdir -p $PROJECT_DIR/data/images
mkdir -p $PROJECT_DIR/models
mkdir -p $PROJECT_DIR/logs
mkdir -p $PROJECT_DIR/results

# Set up systemd services
echo "Setting up system services..."

# Camera node service
sudo tee /etc/systemd/system/pothole-camera.service > /dev/null <<EOF
[Unit]
Description=Pothole Detection Camera Node
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=ROS_MASTER_URI=http://localhost:11311
ExecStartPre=/bin/bash -c 'source $VENV_DIR/bin/activate && source /opt/ros/noetic/setup.bash && source $ROS_WS/devel/setup.bash'
ExecStart=/bin/bash -c 'source $VENV_DIR/bin/activate && source /opt/ros/noetic/setup.bash && source $ROS_WS/devel/setup.bash && python3 src/ros_nodes/camera_node.py'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Detection node service
sudo tee /etc/systemd/system/pothole-detection.service > /dev/null <<EOF
[Unit]
Description=Pothole Detection Node
After=network.target pothole-camera.service

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=ROS_MASTER_URI=http://localhost:11311
ExecStartPre=/bin/bash -c 'source $VENV_DIR/bin/activate && source /opt/ros/noetic/setup.bash && source $ROS_WS/devel/setup.bash'
ExecStart=/bin/bash -c 'source $VENV_DIR/bin/activate && source /opt/ros/noetic/setup.bash && source $ROS_WS/devel/setup.bash && python3 src/ros_nodes/detection_node.py'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# GPS node service
sudo tee /etc/systemd/system/pothole-gps.service > /dev/null <<EOF
[Unit]
Description=Pothole Detection GPS Node
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=ROS_MASTER_URI=http://localhost:11311
ExecStartPre=/bin/bash -c 'source $VENV_DIR/bin/activate && source /opt/ros/noetic/setup.bash && source $ROS_WS/devel/setup.bash'
ExecStart=/bin/bash -c 'source $VENV_DIR/bin/activate && source /opt/ros/noetic/setup.bash && source $ROS_WS/devel/setup.bash && python3 src/ros_nodes/gps_node.py'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Mapping node service
sudo tee /etc/systemd/system/pothole-mapping.service > /dev/null <<EOF
[Unit]
Description=Pothole Detection Mapping Node
After=network.target pothole-detection.service pothole-gps.service

[Service]
Type=simple
User=pi
WorkingDirectory=$PROJECT_DIR
Environment=ROS_MASTER_URI=http://localhost:11311
ExecStartPre=/bin/bash -c 'source $VENV_DIR/bin/activate && source /opt/ros/noetic/setup.bash && source $ROS_WS/devel/setup.bash'
ExecStart=/bin/bash -c 'source $VENV_DIR/bin/activate && source /opt/ros/noetic/setup.bash && source $ROS_WS/devel/setup.bash && python3 src/ros_nodes/mapping_node.py'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ROS Master service
sudo tee /etc/systemd/system/roscore.service > /dev/null <<EOF
[Unit]
Description=ROS Master
After=network.target

[Service]
Type=forking
User=pi
ExecStart=/bin/bash -c 'source /opt/ros/noetic/setup.bash && roscore'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable roscore.service
sudo systemctl enable pothole-camera.service
sudo systemctl enable pothole-detection.service
sudo systemctl enable pothole-gps.service
sudo systemctl enable pothole-mapping.service

# Set permissions
echo "Setting permissions..."
sudo chown -R pi:pi $PROJECT_DIR
sudo usermod -a -G dialout pi  # For GPS serial access
sudo usermod -a -G video pi    # For camera access

# Create launch script
echo "Creating launch script..."
tee $PROJECT_DIR/launch_system.sh > /dev/null <<EOF
#!/bin/bash

# Launch Pothole Detection System

echo "Starting Pothole Detection System..."

# Source environment
source $VENV_DIR/bin/activate
source /opt/ros/noetic/setup.bash
source $ROS_WS/devel/setup.bash

# Start ROS master
echo "Starting ROS master..."
roscore &
sleep 5

# Start nodes
echo "Starting camera node..."
python3 $PROJECT_DIR/src/ros_nodes/camera_node.py &

echo "Starting GPS node..."
python3 $PROJECT_DIR/src/ros_nodes/gps_node.py &

echo "Starting detection node..."
python3 $PROJECT_DIR/src/ros_nodes/detection_node.py &

echo "Starting mapping node..."
python3 $PROJECT_DIR/src/ros_nodes/mapping_node.py &

echo "All nodes started!"
echo "Press Ctrl+C to stop the system"

# Wait for interrupt
trap 'echo "Stopping system..."; kill $(jobs -p); exit' INT
wait
EOF

chmod +x $PROJECT_DIR/launch_system.sh

# Create stop script
tee $PROJECT_DIR/stop_system.sh > /dev/null <<EOF
#!/bin/bash

echo "Stopping Pothole Detection System..."

# Stop services
sudo systemctl stop pothole-mapping.service
sudo systemctl stop pothole-detection.service
sudo systemctl stop pothole-gps.service
sudo systemctl stop pothole-camera.service
sudo systemctl stop roscore.service

# Kill any remaining processes
pkill -f "ros_nodes"
pkill -f "roscore"

echo "System stopped."
EOF

chmod +x $PROJECT_DIR/stop_system.sh

# Create status script
tee $PROJECT_DIR/check_status.sh > /dev/null <<EOF
#!/bin/bash

echo "Pothole Detection System Status:"
echo "================================="

echo "ROS Master:"
sudo systemctl status roscore.service --no-pager -l

echo -e "\nCamera Node:"
sudo systemctl status pothole-camera.service --no-pager -l

echo -e "\nDetection Node:"
sudo systemctl status pothole-detection.service --no-pager -l

echo -e "\nGPS Node:"
sudo systemctl status pothole-gps.service --no-pager -l

echo -e "\nMapping Node:"
sudo systemctl status pothole-mapping.service --no-pager -l

echo -e "\nROS Topics:"
source /opt/ros/noetic/setup.bash
rostopic list 2>/dev/null || echo "ROS Master not running"
EOF

chmod +x $PROJECT_DIR/check_status.sh

# Configure automatic startup (optional)
read -p "Do you want to enable automatic startup on boot? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo systemctl start roscore.service
    sudo systemctl start pothole-camera.service
    sudo systemctl start pothole-detection.service
    sudo systemctl start pothole-gps.service
    sudo systemctl start pothole-mapping.service
    echo "Automatic startup enabled."
fi

# Final configuration reminder
echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Copy your trained model to: $PROJECT_DIR/models/"
echo "2. Update model path in: $PROJECT_DIR/config/ros_config.yaml"
echo "3. Configure GPS serial port if different from /dev/ttyUSB0"
echo "4. Test USB camera with: v4l2-ctl --list-devices && fswebcam test.jpg"
echo "5. Start system with: $PROJECT_DIR/launch_system.sh"
echo "6. Check status with: $PROJECT_DIR/check_status.sh"
echo ""
echo "Reboot recommended to apply all changes."

# Offer to reboot
read -p "Reboot now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi

echo "Setup script completed!"