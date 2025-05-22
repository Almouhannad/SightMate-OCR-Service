from pydantic import BaseModel

from src.domain.models import OCROutput


class HealthResponse(BaseModel):
    status: str

class OCRResponse(BaseModel):
    result: OCROutput
