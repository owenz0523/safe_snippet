import cv2
import numpy as np
from utilsCV import contains_blue_ellipsoid, contains_blue_bbox
from collections import deque

# This class represents the detection and handling of ellipsoids within a video frame
class Ellipsoid:
    def __init__(self, ellipsoids):
        self.ellipsoids = ellipsoids  # List of ellipsoids to be tracked
        self.collision_history = deque(maxlen=5)  # History of collision detections, with a max length of 5
    
    # Method to draw ellipsoids on the frame
    def draw_ellipsoids(self, frame):
        for ex, ey, erx, ery, ea, et in self.ellipsoids:
            # Draw an ellipse on the frame with specified parameters
            cv2.ellipse(frame, (ex, ey), (erx, ery), ea, 0, 360, (0, 255, 255), 2)
        return frame
    
    # Method to check for collisions between ellipsoids and detected objects in the frame
    def check_collision(self, frame, results, bottom_fraction):
        # Create a blank image matching the size of the detection frame
        frame_height, frame_width = frame.shape[:2]
        blank = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        # Loop through each detection in the results
        for det in results[0].boxes:
            if det.cls <= 3:  # Only consider certain classes for collision detection
                # Extract bounding box coordinates
                x1, y1, x2, y2 = det.xyxy[0]

                # Adjust y1 to only check for the bottom part of the bounding box
                modified_y1 = y1 + (y2 - y1) * (1 - bottom_fraction)

                # Create a mask for the bounding box
                object_mask = cv2.rectangle(blank.copy(), (int(x1), int(modified_y1)), (int(x2), int(y2)), 255, -1)
                for ex, ey, erx, ery, ea, et in self.ellipsoids:
                    # Create a mask for the ellipsoid
                    ellipsoid_mask = cv2.ellipse(blank.copy(), (int(ex), int(ey)), (int(erx), int(ery)), ea, 0, 360, 255, -1)

                    # Calculate the intersection between the bounding box and ellipsoid masks
                    intersection = np.logical_and(object_mask, ellipsoid_mask)

                    # Calculate the area of the ellipsoid and the intersection
                    ellipsoid_area = np.sum(ellipsoid_mask > 0)
                    intersection_area = np.sum(intersection > 0)

                    # Check if the bounding box and ellipsoid intersect by at least 50%
                    if intersection_area / ellipsoid_area > 0.5:
                        # Extract an image of the ellipsoid and bounding box
                        ellipsoid_image = cv2.bitwise_and(frame, frame, mask=ellipsoid_mask)
                        image = frame[int(modified_y1):int(y2), int(x1):int(x2)]

                        # Check for blue color in the bounding box and ellipsoid
                        if contains_blue_bbox(image):  # Enable if trials don't work
                            if not contains_blue_ellipsoid(ellipsoid_image, et, 10):  # Last parameter is the minimum area of contour
                                return True  # Return True if a collision is detected
        return False  # Return False if no collision is detected
    
    # Method to update the collision history
    def update_collision_history(self, collision):
        self.collision_history.append(collision)  # Append the collision result to the history
        return sum(self.collision_history) >= 3  # Return True if there have been 3 or more collisions in history
    
    # Method to display a collision warning on the frame
    def collision_warning(self, frame, results, bottom_fraction=0.25):
        collision_detected = self.check_collision(frame, results, bottom_fraction)  # Check for collision
        if self.update_collision_history(collision_detected):  # Update collision history and check if warning is needed
            # Display a collision warning on the frame
            cv2.putText(frame, "Collision Warning!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        return frame  # Return the frame with the warning if applicable
