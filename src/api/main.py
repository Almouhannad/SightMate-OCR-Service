from contextlib import asynccontextmanager
import time
from fastapi import Depends, FastAPI, Request

from src.api.dependencies.authentication import authenticate_api_key
from src.api.schemas import HealthResponse
from src.core.config import CONFIG
from src.domain.authentication.api_key import ApiKey
from src.domain.models import OcrInput, OcrOutput
from src.domain.use_cases.process_image import ProcessImageUseCase
from src.infrastructure.authentication.api_key_repositories.registry import get_api_key_repository
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
    return HealthResponse()

@app.get("/create_key", response_model=ApiKey)
async def create_api_key() -> ApiKey:
    api_key_repo = get_api_key_repository(CONFIG.api_key_repository)()
    return await api_key_repo.create()


@app.post(
    "/ocr/predict",
    response_model=OcrOutput,
    summary="Run OCR on an uploaded image",
    description="Accepts an image file, performs OCR, and returns detected text blocks."
)
async def predict(
    ocr_input: OcrInput,
    use_case: ProcessImageUseCase = Depends(get_process_use_case),
    _ = Depends(authenticate_api_key),
) -> OcrOutput:
    st = time.perf_counter()
    response = use_case.execute(ocr_input)
    elapsed_ms = (time.perf_counter() - st) * 1000
    print(f"Inference time = {elapsed_ms:.2f} ms")
    return response
