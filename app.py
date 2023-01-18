import numpy as np
import torch
import json
import cv2
import os

import detectron2

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo

from predictor import AsyncPredictor

coco_metadata = MetadataCatalog.get('coco_2017_val_panoptic')

# import Mask2Former project
from mask2former import add_maskformer2_config

from PIL import Image
import imutils

###

from flask import Flask, Response, request
import flask

import sys
import time

###

import imutils
from skimage.measure import label, regionprops, find_contours

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

###

app = Flask(__name__)

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file('configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml')
cfg.MODEL.WEIGHTS = 'model_final_f07440.pkl'
#cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
#cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
# predictor = DefaultPredictor(cfg)
num_gpu = torch.cuda.device_count()
async_predictor = AsyncPredictor(cfg, num_gpus=num_gpu)

default_class_names = ['person']


@app.route('/predict', methods=['POST', 'OPTIONS'])
def inference():
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

	# im = cv2.imread(str('desert.png'))
	body = request.get_data()
	im = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
	im = imutils.resize(im, width=512)

	ts1 = time.time()
	outputs = async_predictor(im)
	ts2 = time.time()
	print('segmentation time: ', ts2 - ts1, file=sys.stderr)

	num_masks = outputs['sem_seg'].shape[0]
	print(f'Found {num_masks} masks, {coco_metadata.stuff_classes}', file=sys.stderr)

	predictions = outputs

	# numMasks = predictions['sem_seg'].shape[0]

	# class names array from the query string, split it by ','
	classes_arg = request.args.get('classes')
	class_names = classes_arg.split(',')
	print(f'classes arg {classes_arg}')
	if (len(class_names) == 1 and class_names[0] == ''):
		print('defaulting classes')
		class_names = default_class_names
	# parse boosts, which is a list of floats the same length as class_names
	boosts_arg = request.args.get('boosts')
	boosts = [float(boost) for boost in boosts_arg.split(',')]
	print(f'boosts arg {boosts_arg}')
	if len(boosts) == 1 and boosts[0] == '':
		print('defaulting boosts')
		boosts = [1.0] * len(class_names)
	# parse the threshold query string
	threshold = float(request.args.get('threshold'))
	if (threshold == None):
		threshold = 0.5

	# predictions['sem_seg'] is a Tensor
	r = predictions['sem_seg']

	# boost the predictions for each class
	for i in range(num_masks):
		r[i] = r[i] * 0.5

	# copy of r
	r2 = r.clone()

	# zero out elements where the mask is below the threshold
	# r2[r2 < threshold] = 0
	# clear out zero values
	blank_area = (r2[0] == 0)
	pred_mask = r2.argmax(dim=0).to('cpu')
	pred_mask[blank_area] = 255

	# encode the segment mask into a png, the rgb values storing the class index out of 255
	segment_mask_img = cv2.imencode('.png', pred_mask.numpy())[1].tobytes()

	sem_seg = pred_mask

	bounding_boxes = []
	if isinstance(sem_seg, torch.Tensor):
		sem_seg = sem_seg.numpy()
	labels, areas = np.unique(sem_seg, return_counts=True)
	sorted_idxs = np.argsort(-areas).tolist()
	labels = labels[sorted_idxs]
	for label in filter(lambda l: l < len(coco_metadata.stuff_classes), labels):
		binary_mask = (sem_seg == label).astype(np.uint8)
		text = coco_metadata.stuff_classes[label]

		# TODO sometimes drawn on wrong objects. the heuristics here can improve.
		_num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
		if stats[1:, -1].size == 0:
			continue

		# draw text on the largest component, as well as other very large components.
		for cid in range(1, _num_cc):
			left = stats[cid, cv2.CC_STAT_LEFT]
			top = stats[cid, cv2.CC_STAT_TOP]
			width = stats[cid, cv2.CC_STAT_WIDTH]
			height = stats[cid, cv2.CC_STAT_HEIGHT]

			left = int(left)
			top = int(top)
			width = int(width)
			height = int(height)

			index = int(label)

			bounding_boxes.append({
				'index': index,
				'label': text,
				'bbox': [left, top, left + width, top + height],
			})

	# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	# semantic_result, bboxes = v.draw_sem_seg(outputs['sem_seg'].argmax(0).to('cpu'))

	# sem_seg = Image.fromarray(np.uint8(semantic_result.get_image())).convert('RGB')
	# sem_seg.save('r3.png')

	# segment_mask_img = cv2.imencode('.png', semantic_result.get_image())[1].tobytes()

	response = Response(segment_mask_img)
	response.headers['Content-Type'] = 'image/png'
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Headers'] = '*'
	response.headers['Access-Control-Allow-Methods'] = '*'
	response.headers['Access-Control-Expose-Headers'] = '*'
	response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
	response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
	response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
	response.headers['X-Bounding-Boxes'] = json.dumps(bounding_boxes)
	return response

# serve api routes
@app.route("/pointe", methods=["POST", "OPTIONS"])
def pointe():
    if (flask.request.method == "OPTIONS"):
        # print("got options 1")
        response = flask.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        # print("got options 2")
        return response

    # get body bytes
    body = flask.request.get_data()

    proxyRequest = requests.post("http://127.0.0.1:8000/pointe", data=body)
    
    # proxy the response content back to the client
    response = flask.Response(proxyRequest.content)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, threaded=False)
