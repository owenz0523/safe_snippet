import numpy as np
import cv2

# This class is responsible for transforming points from one view (perspective) to another
class ViewTransformer():
    def __init__(self, space_width, space_length, pixel_vertices):
        self.space_width = space_width  # The width of the target space
        self.space_length = space_length  # The length of the target space

        self.pixel_vertices = pixel_vertices  # The vertices in the original image (source perspective)

        # Define the vertices in the target space (destination perspective)
        self.target_vertices = np.array([
            [0, self.space_length],  # Bottom-left corner
            [0, 0],  # Top-left corner
            [self.space_width, 0],  # Top-right corner
            [self.space_width, self.space_length]  # Bottom-right corner
        ])
    
        # Convert the vertices to float32 for perspective transformation
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Get the perspective transformation matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    # Method to transform a point from the source perspective to the target perspective
    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))  # Convert point coordinates to integer
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0  # Check if the point is inside the polygon defined by pixel_vertices
        if not is_inside:
            return None  # Return None if the point is outside the polygon
        
        # Reshape the point for perspective transformation
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        # Apply the perspective transformation to the point
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        # Return the transformed point, which is the first element in the resulting array
        return transform_point[0][0]
