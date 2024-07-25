import cv2
import numpy as np
from collections import deque, defaultdict
from utilsCV import get_center_of_bbox

# This class handles the detection of people and their zones, as well as forklifts in a video frame.
class PersonDetection:
    def __init__(self, zone1, zone2, zone3, fps):
        # Initialize the zones, detection history, and dictionaries to store locations.
        self.zones = [zone1, zone2, zone3]  # List of predefined zones
        self.person_detection_history = defaultdict(lambda: deque(maxlen=int(fps * 0.5)))  # History of detections to check if a person has been detected for the last 0.5 seconds
        self.forklift_zones = {}  # Dictionary to store forklift zones
        self.forklift_locations = {}  # Dictionary to store forklift locations
        self.people_zones = {}  # Dictionary to store people zones
        self.people_location = {}  # Dictionary to store people locations
        self.last_frame_ids = []  # List to store the IDs of the objects detected in the last frame

    # Method to detect which zone a point (x, y) belongs to
    def detect_zone(self, point):
        x, y = point
        for i, zone in enumerate(self.zones):
            if cv2.pointPolygonTest(zone, (x, y), False) >= 0:
                return i + 1  # Return the zone number (1, 2, or 3)
        return 0  # Return 0 if the point is not in any zone
   
    # Method to draw zones on the frame
    def draw_zones(self, frame, color=(255, 0, 255), thickness=2):
        for zone in self.zones:
            pts = zone.reshape((-1, 1, 2))  # Reshape points for drawing
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)  # Draw polygon lines
        return frame
   
    # Method to draw a bubble around a detected person
    def draw_bubble(self, frame, bbox, zone_number, id):
        center_x, center_y = get_center_of_bbox(bbox.xyxy[0])  # Get the center of the bounding box
        _, _, w, h = bbox.xywh[0]  # Get width and height of the bounding box
        radius = int(max(w, h) / 2)  # Calculate the radius for the bubble
        # Set bubble color based on zone number
        if zone_number == 0:
            color = (0, 255, 0)  # Green for no zone
        elif zone_number == 1:
            color = (0, 255, 255)  # Yellow for zone 1
        else:
            color = (0, 0, 255)  # Red for other zones
        # Draw circle and text for the bubble
        cv2.circle(frame, (center_x, center_y), radius, color, 2)
        cv2.putText(frame, f"Zone {zone_number}", (center_x - radius, center_y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        self.people_location[id] = (center_x, center_y)  # Store the location of the person
        return frame
   
    # Method to check the person detection in the frame
    def check_person_detection(self, results, speed_tracker, frame):
        current_ids = []
        bboxes = results[0].boxes  # Get bounding boxes from detection results
        for bbox in bboxes:
            if bbox.id is not None:
                bbox_class = int(bbox.cls)  # Get the class of the detected object
                bbox_id = int(bbox.id)  # Get the ID of the detected object
                current_ids.append(bbox_id)  # Append the current ID to the list of detected IDs
                center_x, center_y = get_center_of_bbox(bbox.xyxy[0])  # Get the center of the bounding box
                zone_number = self.detect_zone((center_x, center_y))  # Detect the zone of the object
                if bbox_class == 7:  # Class 7 is assumed to be a person
                    self.people_zones[bbox_id] = zone_number  # Store the zone of the person
                    self.draw_bubble(frame, bbox, zone_number, bbox_id)  # Draw bubble around the person
                else:  # For forklifts or other objects
                    self.forklift_zones[bbox_id] = zone_number  # Store the zone of the forklift
                    self.forklift_locations[bbox_id] = (center_x, center_y)  # Store the location of the forklift

        # Remove IDs that are not present in the current frame
        ids_to_remove = [obj_id for obj_id in self.last_frame_ids if obj_id not in current_ids]
        try:
            for id in ids_to_remove:
                if id in self.people_location and id in self.people_zones:
                    del self.people_location[id]  # Remove person's location
                    del self.people_zones[id]  # Remove person's zone
                elif id in self.forklift_locations and id in self.forklift_zones:
                    del self.forklift_locations[id]  # Remove forklift's location
                    del self.forklift_zones[id]  # Remove forklift's zone
        except:
            pass

        # Update the list of last frame IDs
        self.last_frame_ids = current_ids
        forklift_ids = []

        # Check for potential collisions between people and forklifts
        for person_id, person_zone in self.people_zones.items():
            for forklift_id, forklift_zone in self.forklift_zones.items():
                if abs(forklift_zone - person_zone) <= 1 and forklift_zone != 0 and person_zone != 0:
                    if forklift_id in speed_tracker.speeds:
                        # Calculate the average speed vector
                        speed_vector = np.mean(speed_tracker.speeds[forklift_id], axis=0)
                        # Calculate the magnitude of the speed vector
                        speed_vector_magnitude = np.sqrt(speed_vector[0]**2 + speed_vector[1]**2)

                        if 50 <= speed_vector_magnitude <= 500:  # Check if the speed is within a reasonable range
                            for person_id, location in self.people_location.items():
                                person_x, person_y = location
                                forklift_x, forklift_y = self.forklift_locations[forklift_id]
                                # Calculate the direction vector
                                direction_vector = (person_x - forklift_x, person_y - forklift_y)
                                direction_magnitude = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)

                                if direction_magnitude > 0:
                                    # Calculate the dot product
                                    dot_product = np.dot(speed_vector, direction_vector)
                                    # Calculate the cosine of the angle
                                    angle_cosine = dot_product / (speed_vector_magnitude * direction_magnitude)

                                    if angle_cosine > 0.8:  # Check if the cosine of the angle is greater than 0.8 (indicating a potential collision)
                                        forklift_ids.append(forklift_id)  # Return True if a potential collision is detected
        return forklift_ids  # Return False if no potential collision is detected
   
    # Method to update the person detection history
    def update_person_detection_history(self, forklift_ids):
        for forklift, _ in self.forklift_zones.items():
            if forklift in forklift_ids:
                self.person_detection_history[forklift].append(True)
            else:
                self.person_detection_history[forklift].append(False)

        # Remove history entries for forklifts no longer tracked
        keys_to_remove = [id for id in self.person_detection_history.keys() if id not in self.forklift_zones.keys()]
        for id in keys_to_remove:
            if id in self.person_detection_history:
                del self.person_detection_history[id]

    # Method to check for person detection warnings
    def person_detection_warning(self, results, speed_tracker, frame):
        # Check for person detection
        forklift_ids = self.check_person_detection(results, speed_tracker, frame)

        # Update detection history
        self.update_person_detection_history(forklift_ids)

        warning_array = []

        # Generate warnings if the history indicates repeated detections
        for id, history in self.person_detection_history.items():
            if False not in history:
                warning_array.append(id)
        
        return warning_array
