import cv2

# get the x,y coordinates of the mouse click on the
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)