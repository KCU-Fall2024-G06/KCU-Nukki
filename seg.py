import torch
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import cv2
import os
import logging
import sys

# def show_mask(mask, ax):
#     color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def segImage(image) -> bool:
    print(f"Image size: {image.size}")
    logger.debug(f"Image size: {image.size}")
    onnx_model_path = "./weights/sam_mobile.onnx"

    
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    sam.to(device='cpu')
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    image_embedding.shape

    return os.path.isfile(onnx_model_path)