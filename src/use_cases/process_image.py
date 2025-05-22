from src.api.schemas import OCRResponse
from src.domain.ports import OCRPort
from src.domain.models import OCRInput

class ProcessImageUseCase:
    """
    Use-case for processing an image via OCR.
    """
    def __init__(self, ocr_port: OCRPort):
        self._ocr_port = ocr_port

    def execute(self, image_bytes: bytes) -> OCRResponse:
        """
        Wrap raw bytes into OCRInput, delegate to OCRPort, return result.
        """
        ocr_input = OCRInput(image_bytes=image_bytes)
        result = self._ocr_port.predict(ocr_input)
        return OCRResponse(result=result)
