from datetime import datetime
import secrets
import string
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient

from src.core.config import CONFIG
from src.domain.authentication.api_key import ApiKey
from src.domain.authentication.api_key_repository import ApiKeyRepository
from src.infrastructure.authentication.api_key_repositories.registry import register_api_key_repository
from src.infrastructure.authentication.api_key_repositories.mongo_db.api_key_dao import ApiKeyDAO
from src.infrastructure.authentication.utils.hash_provider import HashProvider

API_KEYS_COLLECTION = "api_keys"
KEY_PREFIX_SIZE = 10
KEY_SIZE = 48
@register_api_key_repository("mongo_db")
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
        key_prefix = key[:KEY_PREFIX_SIZE]
        cursor = self._collection.find({"key_prefix": key_prefix}) # This will be indexed search
        matching = None
        async for doc in cursor:
            if self._hash_provider.verify_api_key(key, doc["hashed_key"]):
                matching = doc
                break
        if matching is not None:
            return ApiKeyDAO(**matching).to_domain()
        return None

    async def create(
        self,
        key: Optional[str] = None
    ) -> ApiKey:
        """
        Store a fresh ApiKey record in persistence.
        Should return the stored entity (with id populated).
        """
        if key is None:
            characters = string.ascii_letters + string.digits
            random_part = ''.join(secrets.choice(characters) for _ in range(KEY_SIZE))
            key = f"sk-{random_part}"            
        key_prefix = key[:KEY_PREFIX_SIZE]
        dao = ApiKeyDAO.from_domain(
            ApiKey(
                hashed_key=self._hash_provider.hash_api_key(key),
                key_prefix=key_prefix
            )
        )
        result = await self._collection.insert_one(dao.model_dump(by_alias=True))
        # Inject generated ObjectId back into DAO → domain
        dao.id = str(result.inserted_id)
        return dao.to_domain()

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
        dao = ApiKeyDAO.from_domain(entity)
        print(dao.last_use_in)
        await self._collection.update_one(
            {"_id": dao.id},
            {
                "$set": {
                    "last_use_in": dao.last_use_in,
                    "number_of_requests": dao.number_of_requests,
                }
            },
        )
        return entity
        