# 📡 GPS Module Setup Guide - Neo 6M

## 🔌 **Hardware Connections**

### **Neo 6M GPS Module → Raspberry Pi**
```
GPS Module    →    Raspberry Pi
VCC           →    5V (Pin 2 or 4)
GND           →    Ground (Pin 6, 9, 14, 20, 25, 30, 34, 39)
TX            →    GPIO 14 (Pin 8) - RX on Pi
RX            →    GPIO 15 (Pin 10) - TX on Pi
```

### **📍 Wiring Diagram**
```
Raspberry Pi 40-Pin Header:
     3V3  (1) (2)  5V     ← Connect VCC here
   GPIO2  (3) (4)  5V
   GPIO3  (5) (6)  GND    ← Connect GND here
   GPIO4  (7) (8)  GPIO14 ← Connect GPS TX here
     GND  (9) (10) GPIO15 ← Connect GPS RX here
```

## ⚙️ **Software Setup**

### 1. Enable Serial on Raspberry Pi
```bash
sudo raspi-config
# Navigate to: Interface Options → Serial Port
# "Would you like a login shell accessible over serial?" → NO
# "Would you like the serial port hardware enabled?" → YES
# Reboot when prompted
```

### 2. Edit Boot Config
```bash
sudo nano /boot/config.txt
# Add these lines:
enable_uart=1
dtoverlay=disable-bt
```

### 3. Test GPS Connection
```bash
# Install GPS utilities
sudo apt-get install gpsd gpsd-clients python3-gps

# Test raw GPS data
cat /dev/ttyS0

# You should see NMEA sentences like:
# $GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47
```

## 🧪 **Testing Your GPS**

### Test Script:
```python
import serial
import time

# Test GPS connection
def test_gps():
    try:
        ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
        print("📡 GPS Module connected!")
        
        for i in range(10):
            line = ser.readline().decode('ascii', errors='ignore')
            if line.startswith('$GPGGA'):
                print(f"GPS Data: {line.strip()}")
        
        ser.close()
        return True
    except Exception as e:
        print(f"❌ GPS Error: {e}")
        return False

if __name__ == "__main__":
    test_gps()
```

## 🎯 **What You Need for Demo**

### **Hardware Required:**
1. ✅ **Raspberry Pi** (any model with GPIO)
2. ✅ **Pi Camera** (USB camera or Pi Camera module)
3. ✅ **Neo 6M GPS Module**
4. ✅ **Jumper wires** (4 wires for GPS connection)
5. ✅ **Power supply** for Raspberry Pi

### **Software Ready:**
1. ✅ **car_deployment.py** - Main detection system
2. ✅ **Automatic GPS integration** - Will switch from simulation to real GPS
3. ✅ **Interactive maps** - Folium-powered visualization
4. ✅ **Database logging** - SQLite for detection history

## 📦 **Clone-Ready Status:**

Your code is **100% ready to clone** with:
- ✅ Proper `.gitignore` (excludes large files)
- ✅ Updated `requirements.txt` (all dependencies)
- ✅ Working model architecture
- ✅ Pi deployment scripts
- ✅ GPS integration ready

## 🚗 **Teleop Motor Integration (Later)**

When you're ready for motor control, you'll just need to add:
```python
# In car_deployment.py, when pothole detected:
if pothole_count > 0:
    # Your existing detection logic
    
    # ADD: Motor control commands
    slow_down_vehicle()  # Your teleop function
    alert_driver()       # Your alert system
```

**No changes needed to current detection system!** 🎯