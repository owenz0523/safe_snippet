# get the center of a bounding box
def get_center_of_bbox(bbox):
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    return center_x, center_y

# draw bounding boxes on the frame
def draw_bbox(results):
    return results[0].plot()