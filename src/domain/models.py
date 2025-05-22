from pydantic import BaseModel
from typing import List, Optional

class OCRInput(BaseModel):
    """
    Input for OCR: raw image bytes.
    """
    image_bytes: bytes

class TextBlock(BaseModel):
    """
    Single block of recognized text.
    """
    text: str

class OCROutput(BaseModel):
    """
    OCR result: list of text blocks and annotated image.
    """
    blocks: List[TextBlock]
    annotated_image: Optional[bytes] = None
