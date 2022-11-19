from pathlib import Path
import numpy as np
import tempfile
import json
import sys
sys.path.insert(0, 'Mask2Former')
import cv2

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode as CN
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg

from skimage.measure import label, regionprops, find_contours
import torch

# import Mask2Former project
from mask2former import add_maskformer2_config

from flask import Flask, Response, request


app = Flask(__name__)


''' Convert a mask to border image '''
def mask_to_border(mask):
	h, w = mask.shape
	border = np.zeros((h, w))

	contours = find_contours(mask, 0.5)
	for contour in contours:
		for c in contour:
			x = int(c[0])
			y = int(c[1])
			border[x][y] = 255

	return border

''' Mask to bounding boxes '''
def mask_to_bbox(mask):
	bboxes = []

	mask = mask_to_border(mask)
	lbl = label(mask)
	props = regionprops(lbl)
	for prop in props:
		x1 = prop.bbox[1]
		y1 = prop.bbox[0]

		x2 = prop.bbox[3]
		y2 = prop.bbox[2]

		bboxes.append([x1, y1, x2, y2])

	return bboxes

def detect_bounding_boxes(maskImage, minSize):
	''' Detecting bounding boxes '''
	bboxes = mask_to_bbox(maskImage)
	# filter out bboxes that have width or height smaller than minSize
	bboxes = [bbox for bbox in bboxes if (bbox[2] - bbox[0]) > minSize and (bbox[3] - bbox[1]) > minSize]
	return bboxes

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
	if (request.method == 'OPTIONS'):
		print('got options 1')
		response = Response()
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		print('got options 2')
		return response

	body = request.get_data()
	image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)

	threshold = float(request.args.get('threshold'))
	if (threshold == None):
		threshold = 0.5

	# parse boosts, which is a list of floats teh same length as class_names
	# boosts_arg = request.args.get('boosts')
	# boosts = [float(boost) for boost in boosts_arg.split(',')]
	# print(f'boosts arg {boosts_arg}')
	# if (len(boosts) == 1 and boosts[0] == ''):
	# 	print('defaulting boosts')
	# 	boosts = [1.0] * len(class_names)

	# predictions, _ = run_on_image(im)
	predictions = predictor(image)
	visualizer = Visualizer(image[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	# outputs = predictor(im)
	# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	# panoptic_result = v.draw_panoptic_seg(outputs['panoptic_seg'][0].to('cpu'), outputs['panoptic_seg'][1]).get_image()
	# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	# instance_result = v.draw_instance_predictions(outputs['instances'].to('cpu')).get_image()
	# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	# semantic_result = visualizer.draw_sem_seg(predictions['sem_seg'].argmax(0).to('cpu')).get_image()

	# img = cv2.imencode('.png', semantic_result)[1].tobytes()
	# cv2.imwrite('out.png', semantic_result)

	semantic_result, bboxes = visualizer.draw_sem_seg(predictions['sem_seg'].argmax(0).to('cpu'))

	# encode the segment mask into a png, the rgb values storing the class index out of 255
	segment_mask_img = cv2.imencode('.png', semantic_result.get_image())[1].tobytes()

	response = Response(segment_mask_img)
	response.headers['Content-Type'] = 'image/png'
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Headers'] = '*'
	response.headers['Access-Control-Allow-Methods'] = '*'
	response.headers['Access-Control-Expose-Headers'] = '*'
	response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
	response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
	response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
	response.headers['X-Bounding-Boxes'] = json.dumps(bboxes)

	return response


if __name__ == '__main__':
	cfg = get_cfg()
	add_deeplab_config(cfg)
	add_maskformer2_config(cfg)
	cfg.merge_from_file('configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml')
	cfg.MODEL.WEIGHTS = 'model_final_f07440.pkl'
	cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
	cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
	cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
	predictor = DefaultPredictor(cfg)
	coco_metadata = MetadataCatalog.get('coco_2017_val_panoptic')

	app.run(host='0.0.0.0', port=8080, threaded=True)
