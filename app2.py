import numpy as np
import torch
import json
import cv2
import os
import sys

import detectron2

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo

coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# import Mask2Former project
from mask2former import add_maskformer2_config

from PIL import Image
import imutils

import time
from flask import Flask, Response, request

# from werkzeug.middleware.profiler import ProfilerMiddleware

app = Flask(__name__)
# app.wsgi_app = ProfilerMiddleware(app.wsgi_app)

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)

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
    outputs = predictor(im)
    ts2 = time.time()
    print('inference time: ', ts2 - ts1)

    numMasks = outputs["sem_seg"].shape[0]
    print(f"Found {numMasks} masks", file=sys.stderr)

    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    semantic_result, bboxes = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu"))

    sem_seg = Image.fromarray(np.uint8(semantic_result.get_image())).convert('RGB')
    sem_seg.save('r3.png')

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
    app.run(host='0.0.0.0', port=8080, threaded=True)
