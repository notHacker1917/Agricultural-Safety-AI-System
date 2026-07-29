"""
Advanced Visualization for Harvester Safety Monitoring.

Shows:
- Danger zones (critical blind spot, warning FOV)
- Human detections with multi-source confidence
- Time-to-collision warnings
- Operator alerts
"""

import cv2
import numpy as np
import logging

class HarvesterSafetyVisualizer:
    """
    Visualize harvester safety zones, detections, and alerts.
    """
    
    def __init__(self):
        self.colors = {
            'critical': (0, 0, 255),    # Red
            'warning': (0, 255, 255),   # Yellow
            'safe': (0, 255, 0),        # Green
            'harvester': (128, 128, 255) # Orange
        }
        
        logging.info("Harvester Safety Visualizer initialized")

    def draw_danger_zones(self, frame, zones_data):
        """
        Draw critical and warning danger zones.
        
        Args:
            frame: Input frame (BGR)
            zones_data: Output from HarvesterSafetyEngine.get_danger_zones_visualization()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw warning zone (yellow, semi-transparent)
        warning = zones_data['warning']
        cv2.rectangle(annotated, warning['pt1'], warning['pt2'], 
                     warning['color'], 2)
        cv2.putText(annotated, warning['name'], 
                   (warning['pt1'][0], warning['pt1'][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, warning['color'], 2)
        
        # Draw critical zone (red, semi-transparent)
        critical = zones_data['critical']
        cv2.rectangle(annotated, critical['pt1'], critical['pt2'],
                     critical['color'], 3)
        cv2.putText(annotated, critical['name'],
                   (critical['pt1'][0], critical['pt1'][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, critical['color'], 2)
        
        # Draw harvester position
        hx, hy = zones_data['harvester_center']
        cv2.circle(annotated, (hx, hy), 10, self.colors['harvester'], -1)
        cv2.putText(annotated, "HARVESTER", (hx + 15, hy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['harvester'], 2)
        
        return annotated

    def draw_human_detections(self, frame, detections, risk_assessments, harvester_pos=(0.5, 0.7)):
        """
        Draw human detections with risk coloring and confidence.
        
        Args:
            frame: Input frame
            detections: [(bbox, confidence), ...]
            risk_assessments: [{risk_level, risk_score, ...}, ...]
            harvester_pos: (x_norm, y_norm)
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        for i, detection in enumerate(detections):
            # Handle enhanced detection format: (bbox, confidence, movement_data)
            if len(detection) == 3:
                bbox, conf, movement_data = detection
            else:
                # Legacy format: (bbox, confidence)
                bbox, conf = detection
                movement_data = None
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Get risk assessment
            risk = risk_assessments[i] if i < len(risk_assessments) else {'risk_level': 'SAFE', 'risk_score': 0}
            
            # Enhanced 5-tier color coding with precise thickness for proximity-based risk
            if risk['risk_level'] == 'CRITICAL':
                color = (0, 0, 255)  # Red - immediate danger (thick border)
                thickness = 5  # Thickest border for highest risk
            elif risk['risk_level'] == 'HIGH_WARNING':
                color = (0, 0, 128)  # Dark red - very high danger
                thickness = 4
            elif risk['risk_level'] == 'WARNING':
                color = (0, 165, 255)  # Orange - moderate danger
                thickness = 3
            elif risk['risk_level'] == 'LOW_WARNING':
                color = (0, 255, 255)  # Yellow - low danger
                thickness = 2
            else:  # SAFE
                color = (0, 255, 0)  # Green - no danger
                thickness = 1
            
            # Draw movement direction arrow for dynamic risk assessment
            if movement_data:
                direction = movement_data.get('direction', 'stationary')
                speed = movement_data.get('speed_category', 'moderate')

                # Draw arrow indicating movement direction
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                arrow_length = min(30, (x2 - x1) // 3)

                if direction == 'down':  # Approaching
                    arrow_tip_y = center_y + arrow_length
                    arrow_tip_x = center_x
                    cv2.arrowedLine(annotated, (center_x, center_y), (arrow_tip_x, arrow_tip_y),
                                 (0, 0, 255), 2, tipLength=0.3)  # Red arrow for approaching
                elif direction == 'up':  # Retreating
                    arrow_tip_y = center_y - arrow_length
                    arrow_tip_x = center_x
                    cv2.arrowedLine(annotated, (center_x, center_y), (arrow_tip_x, arrow_tip_y),
                                 (0, 255, 0), 2, tipLength=0.3)  # Green arrow for retreating
                elif direction == 'left':
                    arrow_tip_x = center_x - arrow_length
                    arrow_tip_y = center_y
                    cv2.arrowedLine(annotated, (center_x, center_y), (arrow_tip_x, arrow_tip_y),
                                 (255, 0, 0), 2, tipLength=0.3)
                elif direction == 'right':
                    arrow_tip_x = center_x + arrow_length
                    arrow_tip_y = center_y
                    cv2.arrowedLine(annotated, (center_x, center_y), (arrow_tip_x, arrow_tip_y),
                                 (255, 0, 0), 2, tipLength=0.3)
            
            # Draw confidence and risk
            label = f"Risk: {risk['risk_level']} ({risk['risk_score']:.2f})"
            
            # Add movement information if available
            if movement_data:
                direction = movement_data.get('direction', 'unknown')
                speed = movement_data.get('speed_category', 'unknown')
                label += f" | {direction}@{speed}"
            
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw distance if available
            if 'distance_m' in risk:
                dist_label = f"Dist: {risk['distance_m']:.1f}m"
                cv2.putText(annotated, dist_label, (x1, y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw time-to-collision if CRITICAL
            if risk['risk_level'] == 'CRITICAL' and 'time_to_collision_s' in risk:
                ttc = risk['time_to_collision_s']
                if ttc < float('inf'):
                    ttc_label = f"!" + str(int(ttc)) + "s!"
                    cv2.putText(annotated, ttc_label, (x1 + 5, y1 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        return annotated

    def draw_alerts(self, frame, detections, risk_assessments):
        """
        Draw prominent safety alerts when humans in critical zones.
        
        Args:
            frame: Input frame
            detections: Detected humans
            risk_assessments: Risk data
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Count critical alerts
        critical_count = sum(1 for ra in risk_assessments if ra.get('risk_level') == 'CRITICAL')
        warning_count = sum(1 for ra in risk_assessments if ra.get('risk_level') == 'WARNING')
        
        # Draw alert banner at top
        if critical_count > 0:
            cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 255), -1)
            cv2.putText(annotated, f"!! CRITICAL ALERT !! {critical_count} HUMAN(S) IN BLIND SPOT !!",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        elif warning_count > 0:
            cv2.rectangle(annotated, (0, 0), (w, 60), (0, 165, 255), -1)
            cv2.putText(annotated, f"WARNING: {warning_count} HUMAN(S) IN FIELD OF VIEW",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated, (0, 0), (w, 40), (0, 255, 0), -1)
            cv2.putText(annotated, "SAFE: No humans in danger zones",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Draw summary at bottom right
        summary = f"Detections: {len(detections)} | Critical: {critical_count} | Warning: {warning_count}"
        cv2.putText(annotated, summary, (w - 400, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return annotated

    def annotate_frame(self, frame, detections, risk_assessments, zones_data):
        """
        Complete annotation with all safety information.
        
        Args:
            frame: Input frame
            detections: Human detections
            risk_assessments: Risk data
            zones_data: Danger zones
            
        Returns:
            Fully annotated frame
        """
        # Draw zones first
        annotated = self.draw_danger_zones(frame, zones_data)
        
        # Draw detections
        annotated = self.draw_human_detections(annotated, detections, risk_assessments)
        
        # Draw alerts on top
        annotated = self.draw_alerts(annotated, detections, risk_assessments)
        
        return annotated
