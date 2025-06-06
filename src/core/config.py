from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    """Application settings."""
    
    # OCR adapter settings
    ocr_adapter: str = "paddleocr"  # Default to paddleocr
    lms_api: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

CONFIG = AppConfig()
