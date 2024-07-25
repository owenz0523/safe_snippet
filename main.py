from utilsCV import read_video, show_video, release_video, draw_bbox, click_event, draw_trapezoid
from model_setup import model_config
from cuda_setup import cudaCheck
from object_speed import SpeedTracker
from view_transformer import ViewTransformer
from collision_detection import CollisionDetector
from backward_detection import BackwardDetection
from person_detection import PersonDetection

import numpy as np
import cv2

def main():
    # Read Video
    source = 'input_videos/test.mp4'
    cap = read_video(source)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
    print(fps)

    # Initialize Model
    dev = cudaCheck()  # Check for CUDA device
    model = model_config('models/240621_seg3.pt', dev, 0.4, 0.5, True, 100, False)  # Load YOLOv8 model with specified parameters
    print(model.names)

    # Initialize View Transformer
    trapezoid = np.array([[270, 830], [20, 150], [235, 65], [1050, 265]])  # Define trapezoid area for perspective transformation
    # trapezoid = np.array([[430, 1250], [20, 240], [340, 105], [1560, 395]])
    view_transformer = ViewTransformer(9.6012, 27.2415, trapezoid)  # Initialize view transformer with specified dimensions

    # Initialize Speed Tracker and Collision Detector
    speed_tracker = SpeedTracker(trapezoid, view_transformer)  # Create SpeedTracker instance
    collision_detector = CollisionDetector()  # Create CollisionDetector instance

    # Initialize Backward Detection
    orientation = {
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, 1)
    }
    expected_direction = {
        "down": "up",
        "left": "right",
        "right": "left",
        "up": "down"
    }
    backwards_detection = BackwardDetection(orientation, expected_direction, (200, 75), (1920, 640), fps)  # Create BackwardDetection instance

    # Initialize Person Detection
    zone1 = np.array([[18, 180], [400, 55], [590, 106], [37, 329]], dtype=np.int32)  # Define zone 1
    zone2 = np.array([[37, 329], [590, 106], [1104, 170], [125, 630]], dtype=np.int32)  # Define zone 2
    zone3 = np.array([[125, 630], [1104, 170], [1916, 376], [334, 1074]], dtype=np.int32)  # Define zone 3
    person_detection = PersonDetection(zone1, zone2, zone3, fps)  # Create PersonDetection instance

    # Set up window for mouse callback
    cv2.namedWindow("YOLOv8 Inference")  # Create a named window for displaying the video
    cv2.setMouseCallback("YOLOv8 Inference", click_event, {'frame': None})  # Set up mouse callback for "YOLOv8 Inference" window

    # Read Video Frames
    paused = False  # Variable to control pausing the video
    frame_count = 0  # Counter to keep track of frames

    # Column points for collision detection
    columns = [(8.2804, 7.8994), (8.2804, 19.3421), (5.6100, 19.3421)]  # Define column points for collision detection

    # Track collision frames
    near_miss_frame_number = 1
    collision_frame_number = 1
    backward_frame_number = 1
    person_frame_number = 1

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()  # Read a frame from the video
            frame_count += 1  # Increment the frame count

        if ret:
            # Skip every 5th frame and reset the frame count
            if frame_count % 5 != 0:
                continue
            frame_count = 0

            # Update frame in callback parameters
            cv2.setMouseCallback("YOLOv8 Inference", click_event, {'frame': frame})

            # Run YOLOv8 inference on the frame
            results = model.track(frame, conf=0.4, iou=0.5, persist=True, tracker="bytetrack.yaml")

            # Visualize the results on the frame
            annotated_frame = draw_bbox(results)

            # Annotate with ellipsoids + trapezoid
            annotated_frame = draw_trapezoid(annotated_frame, trapezoid, color=(255, 255, 255), thickness=2)
            
            # Keep track of speed
            speed_tracker.update_positions(results, fps)  # Update positions and speeds of objects
            annotated_frame = speed_tracker.annotate_speeds(annotated_frame, 30)  # Annotate speeds on the frame

            # Draw zones for person detection
            person_detection.draw_zones(annotated_frame)

            # Check for near misses and collisions
            near_miss_ids = collision_detector.near_miss_warning(results, columns, speed_tracker)
            collision_ids = collision_detector.collision_warning(results, columns, speed_tracker)

            # Check for backwards driving
            backwards_ids = backwards_detection.backwards_warning(results, speed_tracker)

            # Check for pedestrians
            person_ids = person_detection.person_detection_warning(results, speed_tracker, annotated_frame)

            # Save frames and print messages for detected events
            if near_miss_ids:
                frame_filename = f"images/near_miss_{near_miss_frame_number}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                near_miss_frame_number += 1
                print("near miss detected", near_miss_ids)

            if collision_ids:
                frame_filename = f"images/collision_{collision_frame_number}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                collision_frame_number += 1
                print("collision detected", collision_ids)

            if backwards_ids:
                frame_filename = f"images/backwards_{backward_frame_number}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                backward_frame_number += 1
                print("backwards detected", backwards_ids)

            if person_ids:
                frame_filename = f"images/person_{person_frame_number}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                person_frame_number += 1
                print("person detected", person_ids)

            # Show Video
            show_video(annotated_frame)

            # Break
            key = cv2.waitKey(1) & 0xFF  # Wait for a key press
            if key == ord('q'):  # Break loop if 'q' key is pressed
                break
            elif key == 32:  # Pause or resume video if space bar is pressed
                paused = not paused
        else:
            break
    
    # Release the video capture object and close the display window
    release_video(cap)

if __name__ == '__main__':
    main()  # Call the main function