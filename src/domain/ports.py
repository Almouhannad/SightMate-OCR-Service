from abc import ABC, abstractmethod
from src.domain.models import OCRInput, OCROutput

class OCRPort(ABC):
    """
    Port/interface for OCR implementations.
    """
    @abstractmethod
    def predict(self, data: OCRInput) -> OCROutput:
        """
        Perform OCR on the given input data.
        """
        pass
