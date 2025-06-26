from typing import Type

from src.domain.authentication.api_key_repository import ApiKeyRepository

_API_KEY_REPOSITORIES: dict[str, Type[ApiKeyRepository]] = {}

def register_api_key_repository(name: str):
    def decorator(cls: Type[ApiKeyRepository]):
        _API_KEY_REPOSITORIES[name] = cls
        return cls
    return decorator

def get_api_key_repository(name: str) -> Type[ApiKeyRepository]:
    try:
        return _API_KEY_REPOSITORIES[name]
    except KeyError:
        raise ValueError(f"No API key repository registered under name {name!r}")

def list_available_repositories() -> list[str]:
    return list(_API_KEY_REPOSITORIES.keys()) 