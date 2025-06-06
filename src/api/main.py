from contextlib import asynccontextmanager
import time
from fastapi import Depends, FastAPI, Request

from src.api.schemas import HealthResponse
from src.core.config import CONFIG
from src.domain.models import OcrInput, OcrOutput
from src.domain.use_cases.process_image import ProcessImageUseCase
from src.infrastructure.models.registry import get_adapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: instantiate adapter & use case once
    AdapterCls = get_adapter(CONFIG.ocr_adapter)
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
    return HealthResponse(status=CONFIG.lms_api)

@app.post(
    "/ocr/predict",
    response_model=OcrOutput,
    summary="Run OCR on an uploaded image",
    description="Accepts an image file, performs OCR, and returns detected text blocks."
)
async def predict(
    ocr_input: OcrInput,
    use_case: ProcessImageUseCase = Depends(get_process_use_case),
) -> OcrOutput:
    st = time.perf_counter()
    response = use_case.execute(ocr_input)
    elapsed_ms = (time.perf_counter() - st) * 1000
    print(f"Inference time = {elapsed_ms:.2f} ms")
    return response
