import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """Application settings."""

    # Use environment variable if available, else fallback to hardcoded defaults
    ocr_adapter: str = os.getenv("OCR_ADAPTER", "easyocr")
    lms_api: str = os.getenv("LMS_API", "")

    model_config = SettingsConfigDict(
        # Use env file (for host), otherwise env variables will be used (for container)
        env_file=".env" if Path(".env").exists() else None,
        env_file_encoding="utf-8"
    )

CONFIG = AppConfig()
