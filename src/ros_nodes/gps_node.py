#!/usr/bin/env python3
"""
ROS GPS Node for Pothole Detection System
Reads GPS coordinates from Neo-6M module and publishes location data
"""

import rospy
import serial
import time
import threading
from std_msgs.msg import String, Float64
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import NavSatFix, NavSatStatus
import pynmea2
import json

class GPSNode:
    """
    ROS node for GPS data acquisition using Neo-6M module
    """
    
    def __init__(self):
        """Initialize GPS node"""
        rospy.init_node('gps_node', anonymous=True)
        
        # Parameters
        self.serial_port = rospy.get_param('~serial_port', '/dev/ttyUSB0')
        self.baud_rate = rospy.get_param('~baud_rate', 9600)
        self.timeout = rospy.get_param('~timeout', 1.0)
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)  # Hz
        
        # GPS state
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.speed = 0.0
        self.heading = 0.0
        self.fix_quality = 0
        self.num_satellites = 0
        self.hdop = 0.0
        self.last_fix_time = None
        self.gps_connected = False
        
        # Serial connection
        self.serial_connection = None
        
        # Threading
        self.running = True
        self.data_lock = threading.Lock()
        
        # Publishers
        self.navsat_pub = rospy.Publisher('/gps/fix', NavSatFix, queue_size=10)
        self.position_pub = rospy.Publisher('/gps/position', PointStamped, queue_size=10)
        self.raw_pub = rospy.Publisher('/gps/raw', String, queue_size=10)
        self.status_pub = rospy.Publisher('/gps/status', String, queue_size=10)
        
        # Initialize GPS connection
        self.connect_gps()
        
        # Start GPS reading thread
        self.gps_thread = threading.Thread(target=self.gps_read_loop)
        self.gps_thread.daemon = True
        
        # Start publishing thread
        self.pub_thread = threading.Thread(target=self.publish_loop)
        self.pub_thread.daemon = True
        
        rospy.loginfo(f"GPS node initialized")
        rospy.loginfo(f"Serial port: {self.serial_port}, Baud rate: {self.baud_rate}")
    
    def connect_gps(self):
        """Connect to GPS module via serial"""
        try:
            self.serial_connection = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            
            if self.serial_connection.is_open:
                self.gps_connected = True
                rospy.loginfo(f"GPS connected on {self.serial_port}")
                return True
            else:
                rospy.logerr(f"Failed to open GPS serial port {self.serial_port}")
                return False
                
        except serial.SerialException as e:
            rospy.logerr(f"GPS serial connection error: {e}")
            self.gps_connected = False
            return False
        except Exception as e:
            rospy.logerr(f"GPS connection error: {e}")
            self.gps_connected = False
            return False
    
    def parse_nmea_sentence(self, sentence):
        """Parse NMEA sentence and update GPS data"""
        try:
            if sentence.startswith('$'):
                msg = pynmea2.parse(sentence)
                
                with self.data_lock:
                    # GGA - Fix information
                    if isinstance(msg, pynmea2.GGA):
                        if msg.latitude and msg.longitude:
                            self.latitude = float(msg.latitude)
                            self.longitude = float(msg.longitude)
                            self.altitude = float(msg.altitude) if msg.altitude else 0.0
                            self.fix_quality = int(msg.gps_qual) if msg.gps_qual else 0
                            self.num_satellites = int(msg.num_sats) if msg.num_sats else 0
                            self.hdop = float(msg.horizontal_dil) if msg.horizontal_dil else 0.0
                            self.last_fix_time = rospy.Time.now()
                    
                    # RMC - Recommended minimum
                    elif isinstance(msg, pynmea2.RMC):
                        if msg.latitude and msg.longitude:
                            self.latitude = float(msg.latitude)
                            self.longitude = float(msg.longitude)
                            self.speed = float(msg.spd_over_grnd) if msg.spd_over_grnd else 0.0
                            self.heading = float(msg.true_course) if msg.true_course else 0.0
                            self.last_fix_time = rospy.Time.now()
                    
                    # VTG - Track and ground speed
                    elif isinstance(msg, pynmea2.VTG):
                        if msg.spd_over_grnd_kmph:
                            self.speed = float(msg.spd_over_grnd_kmph)
                        if msg.true_track:
                            self.heading = float(msg.true_track)
                
                return True
                
        except pynmea2.ParseError as e:
            rospy.logwarn(f"NMEA parse error: {e}")
            return False
        except Exception as e:
            rospy.logwarn(f"GPS data parsing error: {e}")
            return False
        
        return False
    
    def gps_read_loop(self):
        """Main GPS reading loop"""
        while self.running and not rospy.is_shutdown():
            try:
                if not self.gps_connected:
                    rospy.logwarn("GPS not connected, attempting reconnection...")
                    self.connect_gps()
                    time.sleep(5)
                    continue
                
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    # Read line from GPS
                    line = self.serial_connection.readline().decode('ascii', errors='ignore').strip()
                    
                    if line:
                        # Publish raw NMEA data
                        self.raw_pub.publish(String(data=line))
                        
                        # Parse NMEA sentence
                        self.parse_nmea_sentence(line)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except serial.SerialException as e:
                rospy.logerr(f"GPS serial error: {e}")
                self.gps_connected = False
                time.sleep(1)
            except Exception as e:
                rospy.logerr(f"GPS read loop error: {e}")
                time.sleep(1)
    
    def publish_loop(self):
        """Publishing loop for GPS data"""
        rate = rospy.Rate(self.publish_rate)
        
        while self.running and not rospy.is_shutdown():
            try:
                with self.data_lock:
                    current_time = rospy.Time.now()
                    
                    # Check if we have recent GPS data
                    if (self.last_fix_time is not None and 
                        (current_time - self.last_fix_time).to_sec() < 5.0):
                        
                        # Publish NavSatFix message
                        self.publish_navsat_fix(current_time)
                        
                        # Publish position
                        self.publish_position(current_time)
                    
                    # Publish status
                    self.publish_status(current_time)
                
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"GPS publish loop error: {e}")
                time.sleep(1)
    
    def publish_navsat_fix(self, timestamp):
        """Publish NavSatFix message"""
        try:
            msg = NavSatFix()
            msg.header.stamp = timestamp
            msg.header.frame_id = "gps"
            
            # Status
            msg.status.status = NavSatStatus.STATUS_FIX if self.fix_quality > 0 else NavSatStatus.STATUS_NO_FIX
            msg.status.service = NavSatStatus.SERVICE_GPS
            
            # Position
            msg.latitude = self.latitude
            msg.longitude = self.longitude
            msg.altitude = self.altitude
            
            # Covariance (simplified)
            if self.hdop > 0:
                covariance = (self.hdop * 2) ** 2  # Rough estimate
                msg.position_covariance = [
                    covariance, 0, 0,
                    0, covariance, 0,
                    0, 0, covariance * 4  # Higher uncertainty in altitude
                ]
                msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
            else:
                msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
            
            self.navsat_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish NavSatFix: {e}")
    
    def publish_position(self, timestamp):
        """Publish position as PointStamped"""
        try:
            msg = PointStamped()
            msg.header.stamp = timestamp
            msg.header.frame_id = "gps"
            
            # Convert to local coordinates if needed
            # For now, just use lat/lon/alt
            msg.point.x = self.longitude
            msg.point.y = self.latitude
            msg.point.z = self.altitude
            
            self.position_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish position: {e}")
    
    def publish_status(self, timestamp):
        """Publish GPS status information"""
        try:
            status_data = {
                'connected': self.gps_connected,
                'fix_quality': self.fix_quality,
                'num_satellites': self.num_satellites,
                'hdop': self.hdop,
                'speed': self.speed,
                'heading': self.heading,
                'last_fix_age': (timestamp - self.last_fix_time).to_sec() if self.last_fix_time else -1,
                'timestamp': timestamp.to_sec()
            }
            
            status_json = json.dumps(status_data)
            self.status_pub.publish(String(data=status_json))
            
        except Exception as e:
            rospy.logerr(f"Failed to publish status: {e}")
    
    def get_current_position(self):
        """Get current GPS position (thread-safe)"""
        with self.data_lock:
            return {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'altitude': self.altitude,
                'speed': self.speed,
                'heading': self.heading,
                'fix_quality': self.fix_quality,
                'num_satellites': self.num_satellites,
                'hdop': self.hdop,
                'connected': self.gps_connected,
                'last_fix_time': self.last_fix_time.to_sec() if self.last_fix_time else None
            }
    
    def start(self):
        """Start the GPS node"""
        rospy.loginfo("Starting GPS node...")
        
        # Start threads
        self.gps_thread.start()
        self.pub_thread.start()
        
        # Keep node alive
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("GPS node shutting down...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Cleanup resources"""
        rospy.loginfo("Shutting down GPS node...")
        
        self.running = False
        
        # Wait for threads to finish
        if self.gps_thread.is_alive():
            self.gps_thread.join(timeout=2)
        
        if self.pub_thread.is_alive():
            self.pub_thread.join(timeout=2)
        
        # Close serial connection
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        
        rospy.loginfo("GPS node shutdown complete")


def main():
    """Main function"""
    try:
        gps_node = GPSNode()
        gps_node.start()
    except rospy.ROSInterruptException:
        rospy.loginfo("GPS node interrupted")
    except Exception as e:
        rospy.logerr(f"GPS node error: {e}")


if __name__ == '__main__':
    main()