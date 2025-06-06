from pydantic import BaseModel
from typing import List, Optional, Dict


class Rect(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


class OcrLang(BaseModel):
    lang: str


class OcrLangs:
    EN: OcrLang = OcrLang(lang="en")
    AR: OcrLang = OcrLang(lang="ar")
    LANGS: List[str] = ["en", "ar"]


class OcrOptions(BaseModel):
    lang: OcrLang = OcrLangs.EN


class OcrInput(BaseModel):
    bytes: List[int]
    metadata: Optional[Dict[str, object]] = None
    options: OcrOptions = OcrOptions()


class OcrResult(BaseModel):
    text: str
    confidence: Optional[float] = None
    box: Rect

class OcrOutput(BaseModel):
    texts: list[OcrResult]
    description: Optional[Dict[str, object]] = []
