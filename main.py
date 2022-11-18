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

# import Mask2Former project
from mask2former import add_maskformer2_config

from flask import Flask, Response, request


app = Flask(__name__)


def run_on_image(image):
	predictions = predictor(image)
	# visualizer = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
	visualizer = Visualizer(image[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	if 'sem_seg' in predictions:
		r = predictions['sem_seg']
		blank_area = (r[0] == 0)
		pred_mask = r.argmax(dim=0).to('cpu')
		pred_mask[blank_area] = 255
		pred_mask = np.array(pred_mask, dtype=np.int)

		vis_output = visualizer.draw_sem_seg(
			pred_mask
		)
	else:
		raise NotImplementedError
	return predictions, vis_output

""" Convert a mask to border image """
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

def detect_bounding_boxes(maskImage, minSize):
	""" Detecting bounding boxes """
	bboxes = mask_to_bbox(maskImage)
	# filter out bboxes that have width or height smaller than minSize
	bboxes = [bbox for bbox in bboxes if (bbox[2] - bbox[0]) > minSize and (bbox[3] - bbox[1]) > minSize]
	return bboxes

""" Mask to bounding boxes """
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

@app.route('/predict', methods=['POST'])
def predict():
	body = request.get_data()
	im = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)

	threshold = float(request.args.get("threshold"))
	if (threshold == None):
		threshold = 0.5

	# parse boosts, which is a list of floats teh same length as class_names
	# boosts_arg = request.args.get('boosts')
	# boosts = [float(boost) for boost in boosts_arg.split(',')]
	# print(f'boosts arg {boosts_arg}')
	# if (len(boosts) == 1 and boosts[0] == ''):
	# 	print('defaulting boosts')
	# 	boosts = [1.0] * len(class_names)

	predictions, visualized_output = run_on_image(im)
	# outputs = predictor(im)
	# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	# panoptic_result = v.draw_panoptic_seg(outputs['panoptic_seg'][0].to('cpu'), outputs['panoptic_seg'][1]).get_image()
	# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	# instance_result = v.draw_instance_predictions(outputs['instances'].to('cpu')).get_image()
	# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
	# semantic_result = v.draw_sem_seg(outputs['sem_seg'].argmax(0).to('cpu')).get_image()

	# img = cv2.imencode('.png', semantic_result)[1].tobytes()
	# cv2.imwrite('out.png', semantic_result)

	num_masks = predictions['sem_seg'].shape[0]
	bounding_boxes = []

	r = predictions['sem_seg']

	# boost the predictions for each class
	# for i in range(num_masks):
	# 	r[i] = r[i] * boosts[i]

	# copy of r
	r2 = r.clone()

	# zero out elements where the mask is below the threshold
	r2[r2 < threshold] = 0
	# clear out zero values
	blank_area = (r2[0] == 0)
	pred_mask = r2.argmax(dim=0).to('cpu')
	pred_mask[blank_area] = 255

	# encode the segment mask into a png, the rgb values storing the class index out of 255
	segment_mask_img = cv2.imencode('.png', pred_mask.numpy())[1].tobytes()

	# compute bounding boxes
	for i in range(num_masks):
		# get the mask for this class (i)
		# to do this, filter to include only the pixels where this class is the argmax of mask prediction set
		# the data is a tensor()
		# we want to set the result in the mask to 1 if the class was i, and 0 otherwise
		mask = (pred_mask == i).float()
		# print('got mask')
		# pprint(mask)
		# pprint(mask.shape)
		# convert to numpy
		mask = mask.numpy().astype(np.uint8)
		bboxes = detect_bounding_boxes(mask, 64)
		# print(f'got bounding boxes: {i} {len(bboxes)}')
		bounding_boxes.append(bboxes)

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

	app.run(host='0.0.0.0', port=80, threaded=True)
