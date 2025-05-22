from fastapi import FastAPI, File, UploadFile, Depends
from src.core.config import settings
from src.infrastructure.models.registry import get_adapter
from src.domain.ports import OCRPort
from src.domain.models import OCRInput, OCROutput
from src.use_cases.process_image import ProcessImageUseCase
from .schemas import HealthResponse, OCRResponse

app = FastAPI(title="OCR Service")

def get_ocr_port() -> OCRPort:
    """Dependency that provides the configured OCR adapter instance."""
    AdapterCls = get_adapter(settings.ocr_adapter)
    return AdapterCls()

app.dependency_overrides[OCRPort] = get_ocr_port

def get_process_use_case(
    ocr_port: OCRPort = Depends(get_ocr_port)
) -> ProcessImageUseCase:
    return ProcessImageUseCase(ocr_port)

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post(
    "/ocr/predict",
    response_model=OCRResponse,
    summary="Run OCR on an uploaded image",
    description="Accepts an image file, performs OCR, and returns detected text blocks."
)
async def predict(
    file: UploadFile = File(...),
    use_case: ProcessImageUseCase = Depends(get_process_use_case)
) -> OCROutput:
    """
    Reads uploaded file bytes, invokes the ProcessImageUseCase, and returns OCR results.
    """
    image_bytes = await file.read()
    return use_case.execute(image_bytes)
