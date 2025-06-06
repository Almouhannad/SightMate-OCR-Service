from abc import ABC, abstractmethod

from src.domain.models import OcrInput, OcrOutput

class OcrPort(ABC):
    """
    Port/interface for OCR implementations.
    """
    @abstractmethod
    def predict(self, ocrInput: OcrInput) -> OcrOutput:
        """
        Perform OCR on the given input data.
        """
        pass
