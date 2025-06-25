from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class ApiKey(BaseModel):
    """
    Core business entity. No database or BSON concerns here.
    """
    id: Optional[str] = None
    hashed_key: str
    initialized_in: datetime
    last_use_in: Optional[datetime] = None
    number_of_requests: int = 0