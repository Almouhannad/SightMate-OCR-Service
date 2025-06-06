import os
import yaml
from typing import Dict
from pydantic import BaseModel

class GemmaSettings(BaseModel):
    # API Configuration
    lms_api_base_url: str
    chat_endpoint: str

    # Model Configuration
    model_name: str
    prompt_path: str

    # Request Configuration
    headers: Dict[str, str]

    # Generation Parameters
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repeat_penalty: float

    # Response Processing
    strip_json_markers: bool

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GemmaSettings":
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            return cls(**config)

    def get_full_api_url(self, base_url_override: str | None = None) -> str:
        """Get the full API URL, optionally overriding the base URL"""
        base = base_url_override or self.lms_api_base_url
        return f"{base.rstrip('/')}{self.chat_endpoint}"

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

gemma_settings = GemmaSettings.from_yaml(config_path) 