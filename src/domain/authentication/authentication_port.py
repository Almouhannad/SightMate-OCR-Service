from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from src.domain.authentication.api_key import ApiKey


class AuthenticationPort(ABC):
    @abstractmethod
    async def get_by_hashed(self, hashed_key: str) -> Optional[ApiKey]:
        """
        Retrieve an ApiKey by its hashed_key. Return None if not found.
        """
        pass

    @abstractmethod
    async def create(
        self,
        entity: ApiKey
    ) -> ApiKey:
        """
        Store a fresh ApiKey record in persistence.
        Should return the stored entity (with id populated).
        """
        pass

    @abstractmethod
    async def update_usage(
        self,
        api_key_id: str,
        last_use_in: datetime,
        increment: int = 1
    ) -> None:
        """
        Update the last_use timestamp and increment number_of_requests.
        """
        pass