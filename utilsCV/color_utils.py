import cv2
import numpy as np

# Function to check if an image contains a blue ellipsoid based on the specified criteria
def contains_blue_ellipsoid(image, et, min_area=10):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the blue color in HSV space
    lower_blue = np.array([85, 105, 85])
    upper_blue = np.array([135, 165, 185])

    # Create a mask that captures only the blue areas of the image
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize visibility flags for the left and right sides of the image
    left_visible, right_visible = False, False

    # Determine the position of blue ellipsoids
    for contour in contours:
        x, _, w, _ = cv2.boundingRect(contour)  # Get the bounding rectangle for the contour
        area = cv2.contourArea(contour)  # Calculate the area of the contour

        if area > min_area:  # Check if the contour area is greater than the minimum area
            if x + w // 2 < image.shape[1] // 2:  # Check if the contour is in the left half of the image
                left_visible = True
            else:  # Otherwise, the contour is in the right half of the image
                right_visible = True
    
    # Determine the result based on visibility and perspective (et value)
    if et == 0 and not right_visible:  # Case when et is 0 and right side is not visible
        return False
    elif et == 1 and right_visible:  # Case when et is 1 and right side is visible
        return False
    elif et == 2 and left_visible:  # Case when et is 2 and left side is visible
        return False
    elif et == 3 and not left_visible:  # Case when et is 3 and left side is not visible
        return False
    return True  # Return True if none of the above conditions are met

# Function to check if an image contains any blue bounding box
def contains_blue_bbox(image):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the blue color in HSV space
    lower_blue = np.array([85, 105, 85])
    upper_blue = np.array([135, 165, 185])

    # Create a mask that captures only the blue areas of the image
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Return True if any blue area is found in the mask, otherwise return False
    return np.any(mask > 0)