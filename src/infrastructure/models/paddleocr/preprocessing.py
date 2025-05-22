import io
import numpy as np
import cv2
from PIL import Image
from typing import Tuple
from src.infrastructure.models.paddleocr.config import paddle_ocr_settings

def preprocess_for_det(im_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
    """Preprocess image for detection model."""
    img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
    img = img.resize(paddle_ocr_settings.target_size, Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32)/255.0 - paddle_ocr_settings.det_norm_mean) / paddle_ocr_settings.det_norm_std
    return arr.transpose(2,0,1)[None,...].astype(np.float32), img

def preprocess_recognize(crop: np.ndarray) -> np.ndarray:
    """Preprocess crop for recognition model."""
    im = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    im = (im - paddle_ocr_settings.rec_norm_mean)/paddle_ocr_settings.rec_norm_std
    return im.transpose(2,0,1)[None,...].astype(np.float32)
