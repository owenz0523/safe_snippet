import cv2
 
def file_streaming(video_path, frame_interval=5):
    print(f"Attempting to open video file at: {video_path}")
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print("Error: Could not open video.")
            return None
    except Exception as e:
        print(f'file_streaming error: {e}')
        return None
 
    frame_counter = 0
 
    while True:
        ret, frame = capture.read()  # Read the frame
        if not ret:
            print("Reached the end of the video or failed to read the video stream.")
            break  # Exit the loop if no more frames or error
 
        # Process every 5th frame
        if frame_counter % frame_interval == 0:
            # Do your processing here
            print(f"Processing frame {frame_counter}")
            # Display the frame
            cv2.imshow('Frame', frame)
 
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            print(f"Skipping frame {frame_counter}")
 
        frame_counter += 1
 
    # Release the video capture object and close all windows
    capture.release()
    cv2.destroyAllWindows()
 
    return capture
 
 
# RTSP URL
# rtsp_url = "rtsp://rtsp_stream"
# file_streaming(rtsp_url)
