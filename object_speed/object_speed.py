from utilsCV import get_center_of_bbox
from collections import deque
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter

# This class tracks the speed of objects in the video frame using Kalman Filters and perspective transformation
class SpeedTracker:
    def __init__(self, trapezoid, view_transformer):
        self.last_positions = {}  # Dictionary to store last positions of objects
        self.last_positions_adjusted = {}  # Dictionary to store last adjusted positions of objects
        self.speeds = {}  # Dictionary to store speeds of objects
        self.speeds_adjusted = {}  # Dictionary to store adjusted speeds of objects
        self.trapezoid = trapezoid  # Trapezoid defining the region of interest
        self.view_transformer = view_transformer  # View transformer for perspective transformation
        self.bbox_to_mask = {}  # Dictionary to map bounding boxes to masks
        self.lowest_points = {}  # Dictionary to store lowest points of objects
        self.kalman_filters = {}  # Dictionary to store Kalman filters for objects
        self.speed_adjusted_magnitude_old = 0.0  # Variable to store previous adjusted speed magnitude
        self.speed_old = 0.0  # Variable to store previous speed
        self.speed_adjusted_old = 0.0  # Variable to store previous adjusted speed

    # Method to initialize a Kalman filter for a given object
    def initialize_kalman_filter(self, obj_id, initial_position):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # Create a Kalman filter with state vector of 4 and measurement vector of 2
        kf.x = np.array([initial_position[0], initial_position[1], 0, 0])  # Set the initial state
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                         [0, 1, 0, 0]])
        kf.P *= 1000.  # Covariance matrix
        kf.R = np.array([[10, 0],  # Measurement noise matrix
                         [0, 10]])
        kf.Q = np.array([[1, 0, 0, 0],  # Process noise matrix
                         [0, 1, 0, 0],
                         [0, 0, 10, 0],
                         [0, 0, 0, 10]])
        self.kalman_filters[obj_id] = kf  # Store the Kalman filter in the dictionary

    # Method to match bounding boxes to masks
    def match_bbox_to_mask(self, results):
        boxes = results[0].boxes
        masks = results[0].masks
        for box in boxes:
            if box.id is not None and box.cls <= 6:  # Check if the box class is within the range of interest
                x1, y1, x2, y2 = box.xyxy[0]
                for mask in masks:
                    point = mask.xy[0][0]
                    if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:  # Check if the point is within the bounding box
                        self.bbox_to_mask[int(box.id)] = mask.xy[0]  # Map the bounding box to the mask

    # Method to update positions and speeds of objects
    def update_positions(self, results, fps):
        current_ids = []
        bboxs = results[0].boxes
        self.match_bbox_to_mask(results)  # Match bounding boxes to masks
        for bbox in bboxs:
            if bbox.id is not None:
                obj_id = int(bbox.id)
                position = bbox.xyxy[0]
                center_x, _ = get_center_of_bbox(position)
                point = (int(center_x), int(position[3]))
                if obj_id in self.bbox_to_mask:
                    lowest_point = max(self.bbox_to_mask[obj_id], key=lambda p: p[1])
                    if cv2.pointPolygonTest(self.trapezoid, lowest_point, False) >= 0:
                        current_ids.append(obj_id)
                        if cv2.pointPolygonTest(self.trapezoid, point, False) >= 0:
                            if obj_id not in self.kalman_filters:  # Initialize Kalman filter if necessary
                                self.initialize_kalman_filter(obj_id, point)
                            kf = self.kalman_filters[obj_id]
                            kf.predict()  # Prediction step
                            kf.update(point)  # Update step
                            filtered_point = kf.x[:2]  # Get filtered position from Kalman filter
                            adjusted_position = self.view_transformer.transform_point(np.array(filtered_point))  # Transform the point
                            if obj_id in self.last_positions_adjusted and obj_id in self.last_positions:
                                try:
                                    dx = filtered_point[0] - self.last_positions[obj_id][0]
                                    dy = filtered_point[1] - self.last_positions[obj_id][1]
                                    dx_adjusted = adjusted_position[0] - self.last_positions_adjusted[obj_id][0]
                                    dy_adjusted = adjusted_position[1] - self.last_positions_adjusted[obj_id][1]
                                    speed = (dx * fps, dy * fps)
                                    speed_adjusted = (dx_adjusted * fps, dy_adjusted * fps)
                                    speed_adjusted_magnitude = np.linalg.norm(speed_adjusted)
                                    self.speed_adjusted_magnitude_old = speed_adjusted_magnitude
                                    self.speed_old = speed
                                    self.speed_adjusted_old = speed_adjusted
                                except:
                                    print('object_speed2.py line 91-96 error')
                                    speed_adjusted_magnitude = self.speed_adjusted_magnitude_old
                                    speed = self.speed_old
                                    speed_adjusted = self.speed_adjusted_old
                                if speed_adjusted_magnitude > 3:  # Maintain last 5 speeds
                                    speed_adjusted = (speed_adjusted / speed_adjusted_magnitude) * 2
                                if obj_id not in self.speeds and obj_id not in self.speeds_adjusted:
                                    self.speeds[obj_id] = deque(maxlen=5)
                                    self.speeds_adjusted[obj_id] = deque(maxlen=5)
                                self.speeds[obj_id].append(speed)
                                self.speeds_adjusted[obj_id].append(speed_adjusted)
                            self.last_positions_adjusted[obj_id] = adjusted_position
                            self.last_positions[obj_id] = filtered_point
                            break
        ids_to_remove = [obj_id for obj_id in self.last_positions if obj_id not in current_ids]
        try:
            for id in ids_to_remove:
                if id in self.speeds and id in self.speeds_adjusted:
                    del self.speeds[id]
                    del self.speeds_adjusted[id]
                if id in self.kalman_filters:
                    del self.kalman_filters[id]
        except:
            pass

    # Method to annotate speeds on the frame
    def annotate_speeds(self, frame, vector_scale=10):
        for obj_id, speed in self.speeds_adjusted.items():
            avg_speed = np.mean(speed, axis=0)  # Calculate the average speed
            pos = self.last_positions[obj_id]
            if obj_id in self.lowest_points:
                point = self.lowest_points[obj_id]
                x = int(point[0])
                y = int(point[1])
                end_pos = (int(x + avg_speed[0] * vector_scale), int(y + avg_speed[1] * vector_scale))
                cv2.arrowedLine(frame, (x, y), end_pos, (255, 255, 255), 2, tipLength=0.3)  # Draw an arrow indicating the speed direction
                cv2.putText(frame, f"Speed-x: {avg_speed[0]:.2f} m/s | Speed-y: {avg_speed[1]:.2f} m/s  | Speed: {np.sqrt(avg_speed[0]**2 + avg_speed[1]**2):.2f} m/s  | Angle: {np.tan(np.arctan2(avg_speed[1], avg_speed[0])):.2f} degrees",  (int(pos[0]), int(pos[1] - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        return frame
