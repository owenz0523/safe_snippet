from ultralytics import YOLO

# configurations for model setup
def model_config(path_file, device, conf, iou, agnostic_nms, max_det, verbose):
	try:
		model = YOLO(path_file)
		model.to(device)
		model.overrides['conf'] = conf  # NMS confidence threshold
		model.overrides['iou'] = iou # NMS IoU threshold
		model.overrides['agnostic_nms'] = agnostic_nms  # NMS class-agnostic
		model.overrides['max_det'] = max_det  # maximum number of detections per image
		model.overrides['verbose'] = verbose
	except:
		print ('model_config error')    	
	return model