from datetime import datetime
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import CONFIG
from src.domain.authentication.api_key import ApiKey
from src.domain.authentication.api_key_repository import ApiKeyRepository
from src.infrastructure.authentication.mongo_db.api_key_dto import ApiKeyDTO


class MongoDbAuthenticationAdapter(ApiKeyRepository):
    def __init__(self):
        # Initialize Motor client & select database/collection
        self._client = AsyncIOMotorClient(CONFIG.mongodb_uri)
        self._db = self._client[CONFIG.mongodb_database]
        self._collection = self._db["api_keys"]

    async def get_by_hashed(self, hashed_key: str) -> Optional[ApiKey]:
        """
        Find an API-key record by its hash and return the domain model.
        """
        doc = await self._collection.find_one({"hashed_key": hashed_key})
        if not doc:
            return None
        dto = ApiKeyDTO(**doc)
        return dto.to_domain()

    async def get_by_key(self, key: str) -> Optional[ApiKey]:
        """
        Find an API-key record by its hash and return the domain model.
        """
        doc = await self._collection.find_one({"hashed_key": key})
        if not doc:
            return None
        dto = ApiKeyDTO(**doc)
        return dto.to_domain()

    async def create(self, key: str) -> ApiKey:
        """
        Persist a new ApiKey record; return the saved domain entity with its new ID.
        """
        pass

    async def update_usage(
        self, api_key_id: str, last_use_in: datetime, increment: int = 1
    ) -> None:
        """
        Update the last‐use timestamp and increment the request counter.
        """
        await self._collection.update_one(
            {"_id": ObjectId(api_key_id)},
            {
                "$set": {"last_use_in": last_use_in},
                "$inc": {"number_of_requests": increment},
            },
        )