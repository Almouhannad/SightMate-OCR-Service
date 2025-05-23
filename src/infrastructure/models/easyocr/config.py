import os
import yaml
from pydantic import BaseModel
from typing import List, Optional

class EasyOCRSettings(BaseModel):
    # Model configuration
    lang_list: List[str]
    gpu: bool
    model_storage_directory: str
    user_network_directory: Optional[str]
    detect_network: str
    recog_network: str
    download_enabled: bool
    detector: bool
    recognizer: bool
    verbose: bool
    quantize: bool
    cudnn_benchmark: bool

    # OCR parameters
    decoder: str
    beamWidth: int
    batch_size: int
    workers: int
    allowlist: Optional[str]
    blocklist: Optional[str]
    detail: int
    paragraph: bool
    min_size: int
    rotation_info: Optional[List[int]]

    # Contrast parameters
    contrast_ths: float
    adjust_contrast: float
    filter_ths: float

    # Text detection parameters
    text_threshold: float
    low_text: float
    link_threshold: float
    canvas_size: int
    mag_ratio: float

    # Bounding box parameters
    slope_ths: float
    ycenter_ths: float
    height_ths: float
    width_ths: float
    add_margin: float
    x_ths: float
    y_ths: float

    # Additional filtering
    threshold: float
    bbox_min_score: float
    bbox_min_size: int
    max_candidates: int
    output_format: str

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EasyOCRSettings":
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            return cls(**config)

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

easy_ocr_settings = EasyOCRSettings.from_yaml(config_path) 