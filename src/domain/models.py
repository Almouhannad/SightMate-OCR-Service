from pydantic import BaseModel
from typing import List, Tuple

class OCRInput(BaseModel):
    """
    Input for OCR: raw image bytes.
    """
    image_bytes: bytes

class TextBlock(BaseModel):
    """
    Single block of recognized text with its bounding box.
    """
    text: str
    # (x_min, y_min, x_max, y_max)
    bbox: Tuple[int, int, int, int]

class OCROutput(BaseModel):
    """
    OCR result: list of text blocks.
    """
    blocks: List[TextBlock]
