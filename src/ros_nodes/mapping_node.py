#!/usr/bin/env python3
"""
ROS Mapping Node for Pothole Detection System
Subscribes to detection results and GPS data, updates pothole locations
"""

import rospy
import json
import time
import requests
import sqlite3
import os
from datetime import datetime
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import NavSatFix
import threading
from typing import Dict, List, Tuple, Optional

class PotholeMapping:
    """Class to handle pothole location mapping and storage"""
    
    def __init__(self, db_path='pothole_database.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for pothole storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create potholes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS potholes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    altitude REAL DEFAULT 0.0,
                    confidence REAL NOT NULL,
                    severity REAL DEFAULT 0.0,
                    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    detection_count INTEGER DEFAULT 1,
                    verified BOOLEAN DEFAULT FALSE,
                    notes TEXT
                )
            ''')
            
            # Create index for location queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_location 
                ON potholes (latitude, longitude)
            ''')
            
            conn.commit()
            conn.close()
            
            rospy.loginfo(f"Pothole database initialized: {self.db_path}")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize database: {e}")
    
    def add_pothole(self, lat: float, lon: float, alt: float, 
                   confidence: float, severity: float = 0.0) -> int:
        """Add new pothole or update existing one"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if pothole exists nearby (within ~10 meters)
            threshold = 0.0001  # Roughly 10 meters in decimal degrees
            cursor.execute('''
                SELECT id, detection_count FROM potholes 
                WHERE ABS(latitude - ?) < ? AND ABS(longitude - ?) < ?
                ORDER BY ((latitude - ?) * (latitude - ?) + (longitude - ?) * (longitude - ?))
                LIMIT 1
            ''', (lat, threshold, lon, threshold, lat, lat, lon, lon))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pothole
                pothole_id, count = existing
                cursor.execute('''
                    UPDATE potholes 
                    SET confidence = ?, severity = ?, last_updated = CURRENT_TIMESTAMP,
                        detection_count = detection_count + 1
                    WHERE id = ?
                ''', (confidence, severity, pothole_id))
                rospy.loginfo(f"Updated existing pothole {pothole_id}, count: {count + 1}")
            else:
                # Add new pothole
                cursor.execute('''
                    INSERT INTO potholes (latitude, longitude, altitude, confidence, severity)
                    VALUES (?, ?, ?, ?, ?)
                ''', (lat, lon, alt, confidence, severity))
                pothole_id = cursor.lastrowid
                rospy.loginfo(f"Added new pothole {pothole_id} at ({lat:.6f}, {lon:.6f})")
            
            conn.commit()
            conn.close()
            
            return pothole_id
            
        except Exception as e:
            rospy.logerr(f"Failed to add pothole: {e}")
            return -1
    
    def get_nearby_potholes(self, lat: float, lon: float, radius: float = 0.01) -> List[Dict]:
        """Get potholes near given location"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, latitude, longitude, altitude, confidence, severity,
                       detection_time, last_updated, detection_count, verified
                FROM potholes 
                WHERE ABS(latitude - ?) < ? AND ABS(longitude - ?) < ?
                ORDER BY ((latitude - ?) * (latitude - ?) + (longitude - ?) * (longitude - ?))
            ''', (lat, radius, lon, radius, lat, lat, lon, lon))
            
            results = cursor.fetchall()
            conn.close()
            
            potholes = []
            for row in results:
                potholes.append({
                    'id': row[0],
                    'latitude': row[1],
                    'longitude': row[2],
                    'altitude': row[3],
                    'confidence': row[4],
                    'severity': row[5],
                    'detection_time': row[6],
                    'last_updated': row[7],
                    'detection_count': row[8],
                    'verified': bool(row[9])
                })
            
            return potholes
            
        except Exception as e:
            rospy.logerr(f"Failed to get nearby potholes: {e}")
            return []


class GoogleMapsAPI:
    """Class to handle Google Maps API interactions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
    
    def update_pothole_marker(self, lat: float, lon: float, 
                            severity: float, description: str = "Pothole detected") -> bool:
        """Update pothole marker on Google Maps (via Places API or custom solution)"""
        try:
            # This is a placeholder - actual implementation would depend on
            # whether you're using Google My Maps, a custom web app, or other service
            
            # Example: Send to custom web service
            data = {
                'latitude': lat,
                'longitude': lon,
                'severity': severity,
                'description': description,
                'timestamp': datetime.now().isoformat()
            }
            
            # This would be your custom endpoint
            # response = self.session.post('https://your-server.com/api/potholes', json=data)
            # return response.status_code == 200
            
            rospy.loginfo(f"Would update map marker at ({lat:.6f}, {lon:.6f})")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to update map marker: {e}")
            return False


class MappingNode:
    """
    ROS node for pothole mapping and location tracking
    """
    
    def __init__(self):
        """Initialize mapping node"""
        rospy.init_node('mapping_node', anonymous=True)
        
        # Parameters
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.7)
        self.min_detections = rospy.get_param('~min_detections', 3)
        self.gps_timeout = rospy.get_param('~gps_timeout', 5.0)  # seconds
        self.maps_api_key = rospy.get_param('~maps_api_key', '')
        self.database_path = rospy.get_param('~database_path', 'pothole_database.db')
        self.update_interval = rospy.get_param('~update_interval', 1.0)  # seconds
        
        # State
        self.current_gps = None
        self.last_gps_time = None
        self.pending_detections = []
        self.data_lock = threading.Lock()
        
        # Components
        self.pothole_db = PotholeMapping(self.database_path)
        self.maps_api = GoogleMapsAPI(self.maps_api_key) if self.maps_api_key else None
        
        # Subscribers
        self.detection_sub = rospy.Subscriber('/pothole_detection/results', 
                                            String, self.detection_callback, queue_size=10)
        self.gps_sub = rospy.Subscriber('/gps/fix', 
                                      NavSatFix, self.gps_callback, queue_size=10)
        self.gps_pos_sub = rospy.Subscriber('/gps/position', 
                                          PointStamped, self.gps_position_callback, queue_size=10)
        
        # Publishers
        self.map_update_pub = rospy.Publisher('/mapping/updates', 
                                            String, queue_size=10)
        self.pothole_count_pub = rospy.Publisher('/mapping/pothole_count', 
                                               String, queue_size=10)
        
        # Start update thread
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True
        self.running = True
        
        rospy.loginfo(f"Mapping node initialized")
        rospy.loginfo(f"Confidence threshold: {self.confidence_threshold}")
        rospy.loginfo(f"Minimum detections: {self.min_detections}")
        rospy.loginfo(f"Database: {self.database_path}")
    
    def detection_callback(self, msg):
        """Callback for pothole detection results"""
        try:
            # Parse detection result
            result = json.loads(msg.data)
            
            # Check if pothole detected with sufficient confidence
            if (result.get('detected', False) and 
                result.get('confidence', 0.0) >= self.confidence_threshold):
                
                with self.data_lock:
                    # Check if we have recent GPS data
                    current_time = rospy.Time.now()
                    if (self.current_gps is not None and 
                        self.last_gps_time is not None and
                        (current_time - self.last_gps_time).to_sec() < self.gps_timeout):
                        
                        # Process detection with GPS data
                        self.process_detection_with_gps(result, self.current_gps)
                    else:
                        # Store for later processing when GPS is available
                        result['detection_time'] = current_time.to_sec()
                        self.pending_detections.append(result)
                        
                        # Keep only recent pending detections
                        cutoff_time = current_time.to_sec() - self.gps_timeout
                        self.pending_detections = [
                            d for d in self.pending_detections 
                            if d['detection_time'] > cutoff_time
                        ]
                        
                        rospy.logwarn("Pothole detected but no recent GPS data available")
            
        except Exception as e:
            rospy.logerr(f"Error processing detection: {e}")
    
    def gps_callback(self, msg):
        """Callback for GPS NavSatFix messages"""
        try:
            with self.data_lock:
                # Store GPS data if we have a valid fix
                if msg.status.status >= 0:  # Valid fix
                    self.current_gps = {
                        'latitude': msg.latitude,
                        'longitude': msg.longitude,
                        'altitude': msg.altitude,
                        'timestamp': msg.header.stamp.to_sec()
                    }
                    self.last_gps_time = rospy.Time.now()
                    
                    # Process any pending detections
                    self.process_pending_detections()
                
        except Exception as e:
            rospy.logerr(f"Error processing GPS: {e}")
    
    def gps_position_callback(self, msg):
        """Callback for GPS position messages"""
        try:
            with self.data_lock:
                self.current_gps = {
                    'latitude': msg.point.y,
                    'longitude': msg.point.x,
                    'altitude': msg.point.z,
                    'timestamp': msg.header.stamp.to_sec()
                }
                self.last_gps_time = rospy.Time.now()
                
                # Process any pending detections
                self.process_pending_detections()
                
        except Exception as e:
            rospy.logerr(f"Error processing GPS position: {e}")
    
    def process_pending_detections(self):
        """Process detections that were waiting for GPS data"""
        if not self.pending_detections or not self.current_gps:
            return
        
        processed_count = 0
        remaining_detections = []
        
        for detection in self.pending_detections:
            # Check if detection is still recent enough
            gps_time = self.current_gps['timestamp']
            detection_time = detection['detection_time']
            
            if abs(gps_time - detection_time) < self.gps_timeout:
                self.process_detection_with_gps(detection, self.current_gps)
                processed_count += 1
            else:
                # Keep for potential future processing
                remaining_detections.append(detection)
        
        self.pending_detections = remaining_detections
        
        if processed_count > 0:
            rospy.loginfo(f"Processed {processed_count} pending detections with GPS data")
    
    def process_detection_with_gps(self, detection_result, gps_data):
        """Process a detection with GPS coordinates"""
        try:
            lat = gps_data['latitude']
            lon = gps_data['longitude']
            alt = gps_data['altitude']
            confidence = detection_result['confidence']
            severity = detection_result.get('severity', 0.0)
            
            # Add to database
            pothole_id = self.pothole_db.add_pothole(lat, lon, alt, confidence, severity)
            
            if pothole_id > 0:
                # Update maps if API available
                if self.maps_api:
                    description = f"Pothole (confidence: {confidence:.2f}, severity: {severity:.2f})"
                    self.maps_api.update_pothole_marker(lat, lon, severity, description)
                
                # Publish update
                update_data = {
                    'type': 'pothole_added',
                    'id': pothole_id,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt,
                    'confidence': confidence,
                    'severity': severity,
                    'timestamp': rospy.Time.now().to_sec()
                }
                
                self.map_update_pub.publish(String(data=json.dumps(update_data)))
                
                rospy.loginfo(f"Mapped pothole {pothole_id} at ({lat:.6f}, {lon:.6f}) "
                             f"with confidence {confidence:.3f}")
            
        except Exception as e:
            rospy.logerr(f"Error processing detection with GPS: {e}")
    
    def update_loop(self):
        """Periodic update loop"""
        rate = rospy.Rate(1.0 / self.update_interval)
        
        while self.running and not rospy.is_shutdown():
            try:
                # Get pothole statistics
                if self.current_gps:
                    lat = self.current_gps['latitude']
                    lon = self.current_gps['longitude']
                    
                    # Get nearby potholes
                    nearby = self.pothole_db.get_nearby_potholes(lat, lon, radius=0.01)
                    
                    # Publish statistics
                    stats = {
                        'nearby_potholes': len(nearby),
                        'current_location': {
                            'latitude': lat,
                            'longitude': lon
                        },
                        'timestamp': rospy.Time.now().to_sec()
                    }
                    
                    self.pothole_count_pub.publish(String(data=json.dumps(stats)))
                
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"Error in update loop: {e}")
                time.sleep(1)
    
    def get_pothole_summary(self) -> Dict:
        """Get summary of mapped potholes"""
        try:
            conn = sqlite3.connect(self.pothole_db.db_path)
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM potholes')
            total_count = cursor.fetchone()[0]
            
            # Get recent count (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM potholes 
                WHERE detection_time > datetime('now', '-1 day')
            ''')
            recent_count = cursor.fetchone()[0]
            
            # Get average confidence and severity
            cursor.execute('''
                SELECT AVG(confidence), AVG(severity) FROM potholes
            ''')
            avg_conf, avg_sev = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_potholes': total_count,
                'recent_potholes': recent_count,
                'average_confidence': float(avg_conf) if avg_conf else 0.0,
                'average_severity': float(avg_sev) if avg_sev else 0.0
            }
            
        except Exception as e:
            rospy.logerr(f"Error getting pothole summary: {e}")
            return {}
    
    def start(self):
        """Start the mapping node"""
        rospy.loginfo("Starting mapping node...")
        
        # Start update thread
        self.update_thread.start()
        
        # Keep node alive
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Mapping node shutting down...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Cleanup resources"""
        rospy.loginfo("Shutting down mapping node...")
        
        self.running = False
        
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        
        # Print final summary
        summary = self.get_pothole_summary()
        rospy.loginfo(f"Final pothole summary: {summary}")
        
        rospy.loginfo("Mapping node shutdown complete")


def main():
    """Main function"""
    try:
        mapping_node = MappingNode()
        mapping_node.start()
    except rospy.ROSInterruptException:
        rospy.loginfo("Mapping node interrupted")
    except Exception as e:
        rospy.logerr(f"Mapping node error: {e}")


if __name__ == '__main__':
    main()