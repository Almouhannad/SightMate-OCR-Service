from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # OCR adapter settings
    ocr_adapter: str = "paddleocr"  # Default to paddleocr
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
