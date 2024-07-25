import cv2
from collections import defaultdict, deque
import numpy as np
import socket
import sys

# Get the device name of the current machine
devicename = socket.gethostname()

from utilsCV import get_center_of_bbox

# This class handles the detection of objects moving in the wrong direction
class BackwardDetection:
    def __init__(self, orientations, expected_directions, line_start, line_end, fps):
        self.orientations = orientations  # Expected orientations of objects
        self.expected_directions = expected_directions  # Expected driving directions
        self.line_start = line_start  # Start point of the reference line
        self.line_end = line_end  # End point of the reference line
        self.wrong_direction_history = defaultdict(lambda: deque(maxlen=int(fps * 0.5)))  # History of wrong direction detections
        self.warning_array = {}  # Dictionary to store warnings
   
    # Method to check if a point is left of the reference line
    def is_left_of_line(self, point):
        px, py = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        return (x2 - x1) * (py - y1) > (y2 - y1) * (px - x1)
   
    # Method to draw the reference line on the frame
    def draw_line(self, frame):
        cv2.line(frame, self.line_start, self.line_end, (0, 255, 0), 2)  # Draw green line
        return frame
   
    # Method to check if the driving direction is correct
    def check_driving_direction(self, speed_tracker, obj_id, orientation):
        speed_vector = np.mean(speed_tracker.speeds[obj_id], axis=0)  # Calculate the average speed vector
        angle_rad = np.arctan2(speed_vector[1], speed_vector[0])  # Calculate the angle of the speed vector in radians
        angle_deg = np.degrees(angle_rad)  # Convert the angle to degrees
        if angle_deg < 0:
            angle_deg += 360  # Ensure the angle is within 0 to 360 degrees

        # Check if the angle of the speed vector matches the expected direction for the given orientation
        if orientation == "right":
            if angle_deg <= 80 or angle_deg >= 280:
                return True
        elif orientation == "left":
            if 100 <= angle_deg <= 260:
                return True
        elif orientation == "up":
            if 190 <= angle_deg <= 350:
                return True
        else:  # orientation == "down"
            if 10 <= angle_deg <= 170:
                return True

        return False  # Return False if the direction is not correct
   
    # Method to check if any object is moving backwards
    def check_backwards(self, results, speed_tracker):
        bboxes = results[0].boxes  # Get bounding boxes from detection results
        forklift_ids = []

        for bbox in bboxes:
            bbox_class = int(bbox.cls)  # Get the class of the detected object
            if 2 <= bbox_class <= 5:  # Check if the object class is within the range of interest
                orientation = None
                # Assign orientation based on the class
                if bbox_class == 2:
                    orientation = "down"
                elif bbox_class == 3:
                    orientation = "left"
                elif bbox_class == 4:
                    orientation = "right"
                elif bbox_class == 5:
                    orientation = "up"
                if orientation in self.orientations and bbox.id is not None:
                    obj_id = int(bbox.id)  # Get the ID of the detected object
                    if obj_id in speed_tracker.speeds:
                        # Calculate the adjusted speed vector
                        speed_vector_adjusted = np.mean(speed_tracker.speeds_adjusted[obj_id], axis=0)
                        speed_vector_adjusted_magnitude = np.linalg.norm(speed_vector_adjusted)  # Calculate the magnitude of the speed vector
                        if speed_vector_adjusted_magnitude >= 1:  # Check if the magnitude is significant
                            if self.check_driving_direction(speed_tracker, obj_id, orientation):
                                center_x, center_y = get_center_of_bbox(bbox.xyxy[0])  # Get the center of the bounding box
                                if self.is_left_of_line((center_x, center_y)):  # Check if the object is left of the line
                                    forklift_ids.append(obj_id)  # Append the object ID if it is moving in the wrong direction
        return forklift_ids  # Return the list of object IDs moving in the wrong direction
 
    # Method to update the history of backward detections
    def update_backwards_history(self, speed_tracker, forklift_ids):
        for id in speed_tracker.speeds.keys():
            if id in forklift_ids:
                self.wrong_direction_history[id].append(True)
            else:
                self.wrong_direction_history[id].append(False)
        
        # Remove history entries for objects no longer tracked
        keys_to_remove = [id for id in self.wrong_direction_history.keys() if id not in speed_tracker.speeds.keys()]
        for id in keys_to_remove:
            if id in self.wrong_direction_history:
                del self.wrong_direction_history[id]
   
    # Method to check for backwards warnings
    def backwards_warning(self, results, speed_tracker):
        forklift_ids = self.check_backwards(results, speed_tracker)  # Check for backwards detection
        
        self.update_backwards_history(speed_tracker, forklift_ids)  # Update detection history

        warning_array = []

        # Generate warnings if the history indicates repeated wrong direction detections
        for id, history in self.wrong_direction_history.items():
            if False not in history:
                warning_array.append(id)

        return warning_array  # Return the list of object IDs with warnings