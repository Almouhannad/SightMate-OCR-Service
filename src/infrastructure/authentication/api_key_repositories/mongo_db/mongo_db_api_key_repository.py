from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import CONFIG
from src.domain.authentication.api_key import ApiKey
from src.domain.authentication.api_key_repository import ApiKeyRepository
from src.infrastructure.authentication.mongo_db.api_key_dto import ApiKeyDTO
from src.infrastructure.authentication.utils.hash_provider import HashProvider

API_KEYS_COLLECTION = "api_keys"
class MongoDbApiKeyRepository(ApiKeyRepository):
    def __init__(self):
        # Initialize Motor client & select database/collection
        self._client = AsyncIOMotorClient(CONFIG.mongodb_uri)
        self._db = self._client[CONFIG.mongodb_database]
        self._collection = self._db[API_KEYS_COLLECTION]
        self._hash_provider = HashProvider()

    async def get_by_key(self, key: str) -> Optional[ApiKey]:
        """
        Retrieve an ApiKey by its plain-text key. Return None if not found.
        """
        cursor = self._collection.find({})
        matching = None
        async for doc in cursor:
            if self._hash_provider.verify_api_key(key, doc["hashed_key"]):
                matching = doc
                break
        return ApiKeyDTO(**matching).to_domain()

    async def create(
        self,
        key: str
    ) -> ApiKey:
        """
        Store a fresh ApiKey record in persistence.
        Should return the stored entity (with id populated).
        """
        dto = ApiKeyDTO.from_domain(ApiKey(hashed_key=self._hash_provider.hash_api_key(key)))
        result = await self._collection.insert_one(dto.model_dump(by_alias=True))
        # Inject generated ObjectId back into DTO → domain
        dto.id = str(result.inserted_id)
        return dto.to_domain()

    async def update_usage(
        self,
        entity: ApiKey,
        last_use_in: Optional[datetime] = None,
        increment: int = 1
    ) -> ApiKey:
        """
        Update the last_use timestamp and increment number_of_requests.
        """
        entity.update_usage(last_use_in, increment)
        dto = ApiKeyDTO.from_domain(entity)
        print(dto.last_use_in)
        await self._collection.update_one(
            {"_id": dto.id},
            {
                "$set": {
                    "last_use_in": dto.last_use_in,
                    "number_of_requests": dto.number_of_requests,
                }
            },
        )
        return entity
        