from fastapi import FastAPI

from .schemas import HealthResponse

app = FastAPI(title="OCR Service")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")
