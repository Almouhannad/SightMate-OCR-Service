from src.domain.models import OcrInput, OcrOutput
from src.domain.ports import OcrPort


class ProcessImageUseCase:
    """
    Use-case for processing an image via OCR.
    """
    def __init__(self, ocr_port: OcrPort):
        self._ocr_port = ocr_port

    def execute(self, ocr_input: OcrInput) -> OcrOutput:
        """
        Delegate a Pydantic OcrInput to the OCRPort and return OCRResponse.
        """
        result = self._ocr_port.predict(ocr_input)
        return result