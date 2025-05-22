from pydantic import BaseModel
from typing import List, Tuple, Optional

class OCRInput(BaseModel):
    """
    Input for OCR: raw image bytes.
    """
    image_bytes: bytes

class TextBlock(BaseModel):
    """
    Single block of recognized text with its region coordinates.
    """
    text: str
    # Region points in (x,y) format defining the text area
    text_region: Optional[List[Tuple[float, float]]] = None

class OCROutput(BaseModel):
    """
    OCR result: list of text blocks.
    """
    blocks: List[TextBlock]
