from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from httpx import head

from src.core.config import CONFIG
from src.domain.authentication.api_key import ApiKey
from src.infrastructure.authentication.api_key_repositories.registry import get_api_key_repository

_api_key_repository = get_api_key_repository(CONFIG.api_key_repository)()
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)
def get_unauthorized_error (detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "API key"}
    )

async def authenticate_api_key(api_key: str = Depends(api_key_header)) -> ApiKey:
    """
    FastAPI dependency to authenticate via X-API-Key header.
    """
    if not api_key:
        raise get_unauthorized_error("Missing API Key")
    
    matching = await _api_key_repository.get_by_key(api_key)

    if not matching:
        raise get_unauthorized_error("Invalid API Key")
    
    await _api_key_repository.update_usage(matching)

    return matching
