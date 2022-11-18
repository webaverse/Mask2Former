from pathlib import Path
import numpy as np
import tempfile
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

# import Mask2Former project
from mask2former import add_maskformer2_config

from flask import Flask, Response, request


app = Flask(__name__)


@app.route('/predict', methods=['GET', 'OPTIONS'])
def predict():
    body = request.get_data()
    im = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    panoptic_result = v.draw_panoptic_seg(outputs['panoptic_seg'][0].to('cpu'),
                                          outputs['panoptic_seg'][1]).get_image()
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    instance_result = v.draw_instance_predictions(outputs['instances'].to('cpu')).get_image()
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    semantic_result = v.draw_sem_seg(outputs['sem_seg'].argmax(0).to('cpu')).get_image()
    result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]
    print(type(instance_result))
    img = cv2.imencode('.png', result)[1].tobytes()
    # return out_path
    response = Response(img)
    response.headers['Content-Type'] = 'image/png'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Expose-Headers'] = '*'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    # response.headers['X-Panoptic-Seg'] = panoptic_result.tolist()
    # response.headers['X-Instance-Pred'] = instance_result.tolist()
    # response.headers['X-Semantic-Seg'] = semantic_result.tolist()
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