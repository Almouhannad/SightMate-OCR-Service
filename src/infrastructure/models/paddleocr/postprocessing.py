import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List
from shapely.geometry import Polygon
from src.infrastructure.models.paddleocr.config import paddle_ocr_settings
from src.infrastructure.models.paddleocr.helpers import unclip_polygon, warp_crop

def post_process(det_map: np.ndarray, orig_pil: Image.Image) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Post-process detection map to get boxes and crops."""
    h, w = det_map.shape
    bin_map = (cv2.GaussianBlur(det_map, (5,5), 0) > paddle_ocr_settings.box_threshold).astype(np.uint8)*255
    cnts, _ = cv2.findContours(bin_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    orig = np.array(orig_pil)
    scale_x, scale_y = orig_pil.width / w, orig_pil.height / h

    boxes, crops = [], []
    for c in cnts:
        if cv2.contourArea(c) < paddle_ocr_settings.min_area:
            continue
        eps = paddle_ocr_settings.poly_approx_eps * cv2.arcLength(c, True)
        pp = cv2.approxPolyDP(c, eps, True).reshape(-1,2).astype(np.float32)
        pp = unclip_polygon(pp, paddle_ocr_settings.unclip_ratio)
        if pp.shape[0] < 4:
            continue
        
        # Scale back to original
        pp[:,0] *= scale_x
        pp[:,1] *= scale_y
        
        # Get minAreaRect and boxPoints
        rect = cv2.minAreaRect(pp)
        box4 = cv2.boxPoints(rect).astype(np.float32)
        
        # Clip coordinates
        box4[:,0] = np.clip(box4[:,0], 0, orig_pil.width-1)
        box4[:,1] = np.clip(box4[:,1], 0, orig_pil.height-1)
        if Polygon(box4).area < paddle_ocr_settings.min_area:
            continue

        crop = warp_crop(orig, box4, paddle_ocr_settings.rec_height)
        boxes.append(box4)
        crops.append(crop)

    return boxes, crops 