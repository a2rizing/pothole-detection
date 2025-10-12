# ROS-Based Pothole Detection System

A comprehensive, low-cost pothole detection and mapping system designed for Raspberry Pi using computer vision, GPS tracking, and ROS (Robot Operating System). This system combines a custom lightweight CNN with real-time processing capabilities to identify potholes and map their locations for road maintenance applications.

## 🎯 Project Overview

This project addresses the critical challenge of road damage detection, particularly in regions with varying road conditions. By leveraging affordable hardware and optimized AI models, it provides:

- **Real-time pothole detection** using custom CNN architecture
- **GPS-based location mapping** with severity assessment
- **Modular ROS architecture** for easy deployment and maintenance
- **Raspberry Pi optimization** for portable, low-cost operation
- **Comprehensive data logging** and visualization

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Node   │    │      GPS Node   │    │  Detection Node │
│                 │    │                 │    │                 │
│ • Pi Camera     │    │ • Neo-6M Module │    │ • Custom CNN    │
│ • Image Capture │    │ • Location Data │    │ • Inference     │
│ • ROS Publisher │    │ • ROS Publisher │    │ • Classification│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Mapping Node   │
                    │                 │
                    │ • Data Fusion   │
                    │ • SQLite DB     │
                    │ • Map Updates   │
                    │ • Web Interface │
                    └─────────────────┘
```

## 📁 Project Structure

```
pothole-detection/
├── src/
│   ├── models/
│   │   └── pothole_net.py          # Custom CNN architectures
│   ├── utils/
│   │   ├── dataloader.py           # Dataset handling & preprocessing
│   │   └── metrics.py              # Performance metrics
│   ├── ros_nodes/
│   │   ├── camera_node.py          # Camera data acquisition
│   │   ├── detection_node.py       # AI inference node
│   │   ├── gps_node.py            # GPS data processing
│   │   └── mapping_node.py        # Location mapping & storage
│   ├── train.py                   # Training pipeline
│   └── inference.py               # Standalone inference script
├── config/
│   ├── model_config.yaml          # Model & training configuration
│   ├── ros_config.yaml            # ROS nodes configuration
│   └── raspberry_pi_config.yaml  # Hardware configuration
├── data/                          # Dataset directory
├── models/                        # Trained model storage
├── logs/                          # System logs
├── scripts/
│   └── setup_raspberry_pi.sh     # Automated setup script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Features

### Computer Vision
- **Custom Lightweight CNN**: Optimized for Raspberry Pi performance
- **Multiple Model Variants**: PotholeNet, MiniPotholeNet, Detection models
- **Real-time Inference**: 15-30 FPS on Raspberry Pi 4
- **Severity Assessment**: Quantitative pothole severity scoring
- **Model Optimization**: Quantization and pruning support

### Hardware Integration
- **Raspberry Pi Compatible**: Optimized for Pi 3B+, 4B, Zero 2W
- **Camera Support**: External 1080p USB cameras, Pi Camera v1.3, v2.1, HQ Camera
- **GPS Integration**: Neo-6M module with NMEA parsing
- **Flexible Connectivity**: WiFi, Ethernet, cellular options

### ROS Integration
- **Modular Architecture**: Independent, communicating nodes
- **Standard Interfaces**: Sensor_msgs, geometry_msgs compatibility
- **Real-time Communication**: Efficient topic-based messaging
- **Service-oriented**: Systemd integration for reliability

### Data Management
- **SQLite Database**: Local pothole storage and tracking
- **GPS Correlation**: Automatic location tagging
- **Cloud Sync**: Optional remote data synchronization
- **Visualization**: Real-time mapping and analysis tools

## 📋 Requirements

### Hardware Requirements
- **Raspberry Pi**: 3B+ (minimum), 4B (recommended)
- **Camera**: External 1080p USB camera (recommended) or Pi Camera
- **GPS Module**: Neo-6M or compatible UART GPS
- **Storage**: 32GB+ microSD card (Class 10)
- **Power**: 5V 3A power supply
- **Optional**: IMU, accelerometer for enhanced detection

### Software Requirements
- **OS**: Raspberry Pi OS (64-bit recommended)
- **Python**: 3.8+
- **ROS**: Noetic (Ubuntu 20.04) or equivalent
- **PyTorch**: 1.12+ (CPU optimized)
- **OpenCV**: 4.5+

## 🛠️ Installation

### Quick Setup (Raspberry Pi)

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/pothole-detection.git
cd pothole-detection
```

2. **Run automated setup**:
```bash
chmod +x scripts/setup_raspberry_pi.sh
./scripts/setup_raspberry_pi.sh
```

3. **Configure the system**:
```bash
# Edit configuration files
nano config/ros_config.yaml
nano config/raspberry_pi_config.yaml
```

4. **Add your trained model**:
```bash
# Copy your trained model to the models directory
cp /path/to/your/model.pth models/
```

### Manual Installation

1. **System Dependencies**:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git cmake build-essential
sudo apt install libopencv-dev python3-opencv sqlite3
```

2. **ROS Installation**:
```bash
# Install ROS Noetic
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-noetic-desktop-full
```

3. **Python Environment**:
```bash
python3 -m venv pothole_env
source pothole_env/bin/activate
pip install -r requirements.txt
```

4. **Hardware Configuration**:
```bash
# USB camera detection and testing
v4l2-ctl --list-devices
fswebcam -r 1920x1080 --jpeg 90 test.jpg

# Enable UART for GPS
sudo raspi-config nonint do_serial 0
echo "enable_uart=1" | sudo tee -a /boot/config.txt
```

## 🎓 Model Training

### Dataset Preparation

1. **Organize your dataset**:
```
data/
├── images/
│   ├── pothole/        # Pothole images
│   └── no_pothole/     # Normal road images
└── annotations.json    # Optional: for object detection
```

2. **Train the model**:
```bash
# Classification training
python src/train.py --config config/model_config.yaml

# Custom configuration
python src/train.py \
    --data_dir ./data/images \
    --model_type potholeNet \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

3. **Model optimization** (optional):
```bash
# Quantize model for Raspberry Pi
python scripts/optimize_model.py \
    --model_path models/best_model.pth \
    --output_path models/quantized_model.pth \
    --quantize
```

## 🚦 Usage

### Starting the System

1. **Launch all nodes**:
```bash
./launch_system.sh
```

2. **Start individual nodes**:
```bash
# Terminal 1: ROS Master
roscore

# Terminal 2: Camera Node
python src/ros_nodes/camera_node.py

# Terminal 3: GPS Node
python src/ros_nodes/gps_node.py

# Terminal 4: Detection Node
python src/ros_nodes/detection_node.py

# Terminal 5: Mapping Node
python src/ros_nodes/mapping_node.py
```

3. **Check system status**:
```bash
./check_status.sh
```

### Standalone Inference

```bash
# Live camera
python src/inference.py \
    --model_path models/best_model.pth \
    --input_type camera \
    --camera_id 0

# Video file
python src/inference.py \
    --model_path models/best_model.pth \
    --input_type video \
    --input_path input_video.mp4 \
    --output_path output_video.mp4

# Single image
python src/inference.py \
    --model_path models/best_model.pth \
    --input_type image \
    --input_path image.jpg \
    --output_path result.jpg
```

### System Services

```bash
# Enable automatic startup
sudo systemctl enable roscore.service
sudo systemctl enable pothole-camera.service
sudo systemctl enable pothole-detection.service
sudo systemctl enable pothole-gps.service
sudo systemctl enable pothole-mapping.service

# Manual control
sudo systemctl start pothole-detection.service
sudo systemctl stop pothole-detection.service
sudo systemctl restart pothole-detection.service
```

## 📊 Configuration

### Model Configuration (`config/model_config.yaml`)
```yaml
model:
  type: "potholeNet"
  num_classes: 2
  input_size: [416, 416]  # Larger input for 1080p camera
  
training:
  epochs: 100
  batch_size: 16  # Reduced for higher resolution
  learning_rate: 0.001
  optimizer: "adam"
```

### ROS Configuration (`config/ros_config.yaml`)
```yaml
detection_node:
  model_path: "/path/to/model.pth"
  confidence_threshold: 0.5
  device: "auto"

gps_node:
  serial_port: "/dev/ttyUSB0"
  baud_rate: 9600
```

### Hardware Configuration (`config/raspberry_pi_config.yaml`)
```yaml
hardware:
  camera:
    type: "usb_camera"
    resolution: [1920, 1080]  # Full HD
    framerate: 30
    device_id: 0
    fourcc: "MJPG"
    
  gps:
    model: "neo-6m"
    interface: "uart"
```

## 📈 Performance Optimization

### Raspberry Pi Optimization

1. **GPU Memory Split**:
```bash
# Allocate 128MB to GPU
sudo raspi-config nonint do_memory_split 128
```

2. **CPU Governor**:
```bash
# Performance mode
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

3. **Thermal Management**:
```bash
# Monitor temperature
vcgencmd measure_temp

# Add heatsink/fan for sustained performance
```

### Model Optimization

1. **Quantization**:
```python
# INT8 quantization for faster inference
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

2. **Pruning**:
```python
# Remove unnecessary connections
torch.nn.utils.prune.global_unstructured(
    parameters_to_prune, pruning_method=torch.nn.utils.prune.L1Unstructured,
    amount=0.2
)
```

## 🔍 Monitoring and Debugging

### System Monitoring

```bash
# ROS topics
rostopic list
rostopic echo /pothole_detection/results

# System resources
htop
iotop

# GPU temperature
vcgencmd measure_temp

# Camera test
raspistill -o test.jpg
```

### Log Analysis

```bash
# System logs
journalctl -u pothole-detection.service -f

# ROS logs
tail -f ~/.ros/log/latest/rosout.log

# Application logs
tail -f logs/detection.log
```

### Performance Profiling

```bash
# Model inference time
python src/inference.py --model_path models/model.pth --benchmark

# Memory usage
python scripts/profile_memory.py

# FPS measurement
rostopic hz /camera/image_raw
```

## 🛡️ Troubleshooting

### Common Issues

1. **Camera not detected**:
```bash
# Check USB camera connection
v4l2-ctl --list-devices
lsusb | grep -i camera

# Test camera capture
fswebcam -r 1920x1080 --jpeg 90 test.jpg

# Check camera permissions
ls -l /dev/video*
```

2. **GPS not working**:
```bash
# Check serial connection
sudo minicom -D /dev/ttyUSB0 -b 9600

# Verify UART is enabled
dmesg | grep tty
```

3. **Model loading errors**:
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Verify model file
python -c "import torch; torch.load('models/model.pth')"
```

4. **ROS communication issues**:
```bash
# Check ROS master
echo $ROS_MASTER_URI
rosnode list

# Network configuration
rostopic list
```

### Performance Issues

1. **Low FPS**:
   - Reduce image resolution
   - Use quantized model
   - Optimize preprocessing pipeline
   - Check thermal throttling

2. **High memory usage**:
   - Increase swap space
   - Reduce batch size
   - Enable model optimization
   - Monitor memory leaks

3. **Detection accuracy**:
   - Retrain with local data
   - Adjust confidence threshold
   - Fine-tune model parameters
   - Improve lighting conditions

## 🤝 Contributing

We welcome contributions to improve the system! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes and test thoroughly**
4. **Commit with clear messages**: `git commit -m "Add new feature"`
5. **Push to your fork**: `git push origin feature/new-feature`
6. **Create a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{pothole_detection_2024,
  title={ROS-Based Pothole Detection System for Raspberry Pi},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/pothole-detection}
}
```

## 🙏 Acknowledgments

- **OpenCV community** for computer vision libraries
- **PyTorch team** for deep learning framework
- **ROS community** for robotics middleware
- **Raspberry Pi Foundation** for affordable computing platform
- **Research contributors** in road condition monitoring

## 📞 Support

For questions, issues, or support:

- **Issues**: [GitHub Issues](https://github.com/your-username/pothole-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/pothole-detection/discussions)
- **Email**: your.email@example.com

## 🔮 Future Enhancements

- [ ] **Edge deployment optimization** with TensorRT/ONNX
- [ ] **Multi-sensor fusion** with IMU and accelerometer data
- [ ] **Cloud-based training pipeline** for continuous improvement
- [ ] **Mobile app integration** for real-time monitoring
- [ ] **Advanced mapping features** with route optimization
- [ ] **Machine learning operations (MLOps)** pipeline
- [ ] **Integration with municipal systems** and APIs

---

⭐ **Star this repository** if you find it helpful!

🐛 **Report bugs** and 💡 **suggest features** through GitHub Issues.

🤝 **Contribute** to make road infrastructure monitoring more accessible worldwide!