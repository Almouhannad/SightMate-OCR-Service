import time
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Depends, Request
from src.core.config import settings
from src.infrastructure.models.registry import get_adapter
from src.use_cases.process_image import ProcessImageUseCase
from .schemas import HealthResponse, OCRResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: instantiate adapter & use case once
    AdapterCls = get_adapter(settings.ocr_adapter)
    app.state.ocr_port = AdapterCls()
    app.state.process_use_case = ProcessImageUseCase(app.state.ocr_port)
    yield

app = FastAPI(
    title="OCR Service",
    lifespan=lifespan,
)

def get_process_use_case(request: Request) -> ProcessImageUseCase:
    return request.app.state.process_use_case

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
    use_case: ProcessImageUseCase = Depends(get_process_use_case),
) -> OCRResponse:
    image_bytes = await file.read()
    st = time.perf_counter()
    response = use_case.execute(image_bytes)
    elapsed_ms = (time.perf_counter() - st) * 1000
    print(f"Inference time = {elapsed_ms:.2f} ms")
    response.result.annotated_image = base64.b64encode(
        response.result.annotated_image
    )
    return response
