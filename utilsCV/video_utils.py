import cv2

# read video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

# display video
def show_video(frame):
    imS = cv2.resize(frame, (1440, 810))
    cv2.imshow("YOLOv8 Inference", imS)

# end video
def release_video(cap):
    cap.release()
    cv2.destroyAllWindows()