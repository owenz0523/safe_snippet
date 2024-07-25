from collections import deque, defaultdict
import numpy as np
import cv2

# This class detects collisions and near misses between objects (e.g., forklifts) and columns
class CollisionDetector:
    def __init__(self):
        # Deques to store the history of near misses and collisions
        self.near_miss_history = defaultdict(lambda: deque(maxlen=5))
        self.collision_history = defaultdict(lambda: deque(maxlen=5))

    # Method to get the lowest point of an object based on its speed vector
    def get_lowest_point(self, speed_tracker, obj_id):
        if obj_id in speed_tracker.speeds and obj_id in speed_tracker.speeds_adjusted:
            # Calculate the average speed vector of the object
            speed_vector = np.mean(speed_tracker.speeds[obj_id], axis=0)
            # Calculate the angle of the speed vector in radians and convert to degrees
            angle_rad = np.arctan2(speed_vector[1], speed_vector[0])
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360  # Ensure the angle is within 0 to 360 degrees

            # Get the mask of the bounding box
            mask = speed_tracker.bbox_to_mask[obj_id]
            # Extract x-values from the mask
            x_values = [point[0] for point in mask]

            point = None
            orientation = None
            # Determine the lowest point and orientation based on the angle
            if angle_deg < 70 or angle_deg >= 290:  # Moving right
                threshold = np.percentile(x_values, 85)
                points = [p for p in mask if p[0] >= threshold]
                if points:
                    point = max(points, key=lambda item: item[1])
                    orientation = "right"
            elif 70 <= angle_deg < 110:  # Moving up
                points = np.array(mask)
                center_x = np.mean(points[:, 0])
                center_y = np.mean(points[:, 1])
                point = (center_x, center_y)
                orientation = "up"
            elif 110 <= angle_deg < 250:  # Moving left
                threshold = np.percentile(x_values, 15)
                points = [p for p in mask if p[0] <= threshold]
                if points:
                    point = max(points, key=lambda item: item[1])
                    orientation = "left"
            elif 250 <= angle_deg < 290:  # Moving down
                point = max(mask, key=lambda item: item[1])
                orientation = "down"
            # Store the lowest point in the speed tracker
            speed_tracker.lowest_points[obj_id] = point
            return point, orientation
        return None, None

    # Method to create a line segment representing the width of an object
    def create_line_segment(self, speed_tracker, point, bbox_class, obj_id, orientation, isBackwards, forklift_width=1.32, forklift_fr_width=2.62):
        # Calculate the adjusted speed vector and transform the point
        speed_vector_adjusted = np.mean(speed_tracker.speeds_adjusted[obj_id], axis=0)
        adjusted_point = speed_tracker.view_transformer.transform_point(np.array(point))
        # Calculate the magnitude of the speed vector
        speed_vector_magnitude = np.linalg.norm(speed_vector_adjusted)

        if speed_vector_magnitude > 0:
            # Calculate the perpendicular vector and normalize it
            perp_vector = np.array([speed_vector_adjusted[1], -speed_vector_adjusted[0]])
            perp_vector_normalized = perp_vector / np.linalg.norm(perp_vector)
            if isBackwards:
                forklift_width = 2.62  # Adjust width if moving backwards

            # Calculate the width vector
            width_vector = perp_vector_normalized * forklift_width

            if width_vector is not None:
                extra_width_vector = 0
                if not isBackwards:
                    forklift_extra_width = (forklift_fr_width - forklift_width)
                    extra_width_vector = perp_vector_normalized * forklift_extra_width / 2  # Calculate extra width for front

                line_start = np.array(adjusted_point)  # Start point of the line segment
                line_end = np.array(adjusted_point)  # End point of the line segment

                # Adjust line end based on orientation
                try:
                    if orientation == "right":
                        line_end += width_vector
                    elif orientation == "left":
                        line_end -= width_vector
                    else:  # "up" or "down"
                        line_start -= width_vector / 2
                        line_end += width_vector / 2

                    if 2 <= bbox_class <= 5:  # Adjust for forklifts
                        line_start += extra_width_vector
                        line_end -= extra_width_vector

                    return line_start, line_end
                except Exception as e:
                    print(e)
                    return None, None
        return None, None

    # Method to get the nearest point on a line segment to a column
    def get_nearest_point_to_column(self, line_start, line_end, column, obj_id, speed_tracker):
        # Calculate the line segment vector
        line_segment = line_end - line_start

        # Convert column to numpy array
        column_point = np.array(column)
        # Calculate the vector from line start to column
        point_vector = column_point - line_start

        # Calculate the length of the line segment squared
        segment_length_squared = np.dot(line_segment, line_segment)
        if segment_length_squared != 0:
            # Calculate the projection length and clamp it to the segment
            projection = np.dot(point_vector, line_segment) / segment_length_squared
            projection = max(0, min(1, projection))
            # Calculate the closest point on the line segment
            closest_point = line_start + projection * line_segment
        else:
            closest_point = line_start  # If segment length is zero, closest point is line start

        return closest_point

    # Method to check for near misses between objects and columns
    def check_near_miss(self, results, columns, speed_tracker, isBackwards, forklift_width=1.32, column_radius=0.4572):
        # Get bounding boxes from detection results
        bboxes = results[0].boxes
        forklift_ids = []

        for bbox in bboxes:
            # Get the class of the detected object
            bbox_class = int(bbox.cls)
            if bbox_class <= 6 and bbox.id is not None:  # Check if the object class is within the range of interest
                if 2 <= bbox_class <= 5:  # Adjust width for forklifts
                    forklift_width = 2.62
                # Get the ID of the detected object
                obj_id = int(bbox.id)
                # Get the lowest point and orientation
                point, orientation = self.get_lowest_point(speed_tracker, obj_id)
                if point is not None and orientation is not None:
                    # Check if the point is within the trapezoid
                    if cv2.pointPolygonTest(speed_tracker.trapezoid, point, False) >= 0:
                        # Calculate the adjusted speed vector and its magnitude
                        speed_vector_adjusted = np.mean(speed_tracker.speeds_adjusted[obj_id], axis=0)
                        speed_vector_magnitude = np.linalg.norm(speed_vector_adjusted)
                        # Create the line segment
                        line_start, line_end = self.create_line_segment(speed_tracker, point, bbox_class, obj_id, orientation, isBackwards)
                        if line_start is not None and line_end is not None:
                            for column in columns:  # Check each column
                                # Calculate the unit vector of the speed
                                speed_unit_vector = speed_vector_adjusted / speed_vector_magnitude
                                # Get the closest point to the column
                                closest_point = self.get_nearest_point_to_column(line_start, line_end, column, obj_id, speed_tracker)
                                # Calculate the vector from the column to the closest point
                                column_vector = (column[0] - closest_point[0], column[1] - closest_point[1])
                                # Calculate the magnitude of the column vector
                                column_vector_magnitude = np.linalg.norm(column_vector)

                                # Calculate the projection length and the projection vector
                                projection_length = np.dot(speed_unit_vector, column_vector)
                                projection_vector = projection_length * speed_unit_vector

                                # Calculate the perpendicular vector and its magnitude
                                perpendicular_vector = column_vector - projection_vector
                                perpendicular_vector_magnitude = np.linalg.norm(perpendicular_vector)
                                
                                forklift_width_dial = 0.5
                                speed_dial = 1.5

                                # Check if the near miss criteria are met
                                if (perpendicular_vector_magnitude < (forklift_width * forklift_width_dial + column_radius) and 
                                    projection_length > 0 and 
                                    column_vector_magnitude < speed_vector_magnitude * speed_dial):
                                    print(perpendicular_vector_magnitude, column_vector, speed_vector_adjusted)
                                    forklift_ids.append(obj_id)
        return forklift_ids

    # Method to check for collisions between objects and columns
    def check_collision(self, results, columns, speed_tracker, isBackwards, column_radius=0.4572):
        # Get bounding boxes from detection results
        bboxes = results[0].boxes
        forklift_ids = []
        for bbox in bboxes:
            # Get the class of the detected object
            bbox_class = int(bbox.cls)
            if bbox_class <= 6 and bbox.id is not None:  # Check if the object class is within the range of interest
                # Get the ID of the detected object
                obj_id = int(bbox.id)
                # Get the lowest point and orientation
                point, orientation = self.get_lowest_point(speed_tracker, obj_id)

                if point is not None and orientation is not None:
                    # Check if the point is within the trapezoid
                    if cv2.pointPolygonTest(speed_tracker.trapezoid, point, False) >= 0:
                        # Create the line segment
                        line_start, line_end = self.create_line_segment(speed_tracker, point, bbox_class, obj_id, orientation, isBackwards)
                        if line_start is not None and line_end is not None:
                            for column in columns:  # Check each column
                                # Get the closest point to the column
                                closest_point = self.get_nearest_point_to_column(line_start, line_end, column, obj_id, speed_tracker)
                                # Calculate the vector from the column to the closest point
                                column_vector = np.array(column) - np.array(closest_point)
                                # Calculate the magnitude of the column vector
                                column_vector_magnitude = np.linalg.norm(column_vector)
                                # Check if the collision criteria are met
                                if column_vector_magnitude < column_radius:
                                    forklift_ids.append(obj_id)
        return forklift_ids

    # Method to update the near miss history
    def update_near_miss_history(self, speed_tracker, forklift_ids):
        # Update the history of near misses for each object
        for id in speed_tracker.speeds.keys():
            if id in forklift_ids:
                self.near_miss_history[id].append(True)
            else:
                self.near_miss_history[id].append(False)

        # Remove history entries for objects no longer tracked
        keys_to_remove = [id for id in self.near_miss_history.keys() if id not in speed_tracker.speeds.keys()]
        for id in keys_to_remove:
            if id in self.near_miss_history:
                del self.near_miss_history[id]

    # Method to check for near miss warnings
    def near_miss_warning(self, results, columns, speed_tracker, isBackwards):
        # Check for near misses
        forklift_ids = self.check_near_miss(results, columns, speed_tracker, isBackwards)

        # Update detection history
        self.update_near_miss_history(speed_tracker, forklift_ids)

        warning_array = []
        # Generate warnings if the history indicates repeated near misses
        for id, history in self.near_miss_history.items():
            if sum(history) == 4:
                warning_array.append(id)

        return warning_array

    # Method to update the collision history
    def update_collision_history(self, speed_tracker, forklift_ids):
        # Update the history of collisions for each object
        for id in speed_tracker.speeds.keys():
            if id in forklift_ids:
                self.collision_history[id].append(True)
            else:
                self.collision_history[id].append(False)

        # Remove history entries for objects no longer tracked
        keys_to_remove = [id for id in self.collision_history.keys() if id not in speed_tracker.speeds.keys()]
        for id in keys_to_remove:
            if id in self.collision_history:
                del self.collision_history[id]

    # Method to check for collision warnings
    def collision_warning(self, results, columns, speed_tracker, isBackwards):
        # Check for collisions
        forklift_ids = self.check_collision(results, columns, speed_tracker, isBackwards)

        # Update collision history
        self.update_collision_history(speed_tracker, forklift_ids)

        warning_array = []
        # Generate warnings if the history indicates repeated collisions
        for id, history in self.collision_history.items():
            if sum(history) == 4:
                warning_array.append(id)

        return warning_array
