import numpy as np
import cv2
from PIL import Image
import io
from src.infrastructure.models.paddleocr.config import paddle_ocr_settings

def _normalize_and_transpose(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Shared preprocessing function for normalization and transpose operations.
    
    Args:
        img: Input image array
        mean: Normalization mean values
        std: Normalization standard deviation values
    
    Returns:
        Normalized and transposed image array
    """
    normalized = (img - mean) / std
    return normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)

def preprocess_for_det(im_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    """Preprocess image for detection."""
    img = Image.open(io.BytesIO(im_bytes)).convert("RGB")
    img = img.resize(paddle_ocr_settings.target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return _normalize_and_transpose(arr, paddle_ocr_settings.det_norm_mean, paddle_ocr_settings.det_norm_std), img

def preprocess_recognize(crop: np.ndarray) -> np.ndarray:
    """Preprocess crop for recognition."""
    im = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return _normalize_and_transpose(im, paddle_ocr_settings.rec_norm_mean, paddle_ocr_settings.rec_norm_std)
