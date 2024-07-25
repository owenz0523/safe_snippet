import cv2

# draw trapezoid on frame
def draw_trapezoid(frame, trapezoid, color=(255, 0, 255), thickness=2):
    pts = trapezoid.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    return frame