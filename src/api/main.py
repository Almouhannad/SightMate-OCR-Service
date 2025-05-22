from fastapi import FastAPI
from src.core.config import settings
from src.infrastructure.models.registry import get_adapter
from src.domain.ports import OCRPort
from .schemas import HealthResponse

app = FastAPI(title="OCR Service")

def ocr_dependency() -> OCRPort:
    """Dependency that provides the configured OCR adapter instance."""
    AdapterCls = get_adapter(settings.ocr_adapter)
    return AdapterCls()

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")
