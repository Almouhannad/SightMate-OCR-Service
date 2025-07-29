from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from src.domain.authentication.api_key import ApiKey


class ApiKeyRepository(ABC):
    @abstractmethod
    async def get_by_key(self, key: str) -> Optional[ApiKey]:
        """
        Retrieve an ApiKey by its plain-text key. Return None if not found.
        """
        pass

    @abstractmethod
    async def create(
        self,
        key: Optional[str] = None
    ) -> ApiKey:
        """
        Store a fresh ApiKey record in persistence.
        Should return the stored entity (with id populated).
        """
        pass

    @abstractmethod
    async def update_usage(
        self,
        entity: ApiKey,
        last_use_in: Optional[datetime] = None,
        increment: int = 1
    ) -> ApiKey:
        """
        Update the last_use timestamp and increment number_of_requests.
        """
        pass