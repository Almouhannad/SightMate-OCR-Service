import os
import yaml
import numpy as np
from pydantic import BaseModel
from typing import Tuple, List

class PaddleOCRSettings(BaseModel):
    # ONNX model paths
    det_model_path: str
    rec_model_path: str
    char_dict_path: str

    # ONNX runtime configuration
    providers: List[str]

    # Detection parameters
    box_threshold: float
    min_area: int
    unclip_ratio: float
    poly_approx_eps: float

    # Recognition parameters
    rec_height: int
    target_size: Tuple[int, int]

    # Normalization parameters
    det_norm_mean: List[float]
    det_norm_std: List[float]
    rec_norm_mean: List[float]
    rec_norm_std: List[float]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PaddleOCRSettings":
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            config['target_size'] = tuple(config['target_size'])
            # Convert normalization parameters to numpy arrays
            for key in ['det_norm_mean', 'det_norm_std', 'rec_norm_mean', 'rec_norm_std']:
                config[key] = np.array(config[key], dtype=np.float32)
            return cls(**config)

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

paddle_ocr_settings = PaddleOCRSettings.from_yaml(config_path)
