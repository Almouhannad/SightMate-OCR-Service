from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId

from src.domain.authentication.api_key import ApiKey

class PyObjectId(ObjectId):
    """Teach Pydantic how to handle ObjectId."""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, _):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

class ApiKeyDTO(BaseModel):
    """
    Persistence model for Mongo. Knows about ObjectId, _id aliasing,
    JSON encoding, and ID validation.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    hashed_key: str
    key_prefix: str
    initialized_in: datetime
    last_use_in: Optional[datetime] = None
    number_of_requests: int = 0

    class Config:
        validate_by_name = True
        json_encoders = {ObjectId: str}

    def to_domain(self) -> ApiKey:
        """Convert DTO → pure domain model (ID becomes string)."""
        return ApiKey(
            id=str(self.id),
            hashed_key=self.hashed_key,
            key_prefix=self.key_prefix,
            initialized_in=self.initialized_in,
            last_use_in=self.last_use_in,
            number_of_requests=self.number_of_requests,
        )

    @classmethod
    def from_domain(cls, entity: ApiKey) -> "ApiKeyDTO":
        """Convert domain model → DTO (string ID → ObjectId)."""
        data = entity.model_dump()
        # if no id, let default_factory generate one:
        if data.get("id") is not None:
            data["_id"] = ObjectId(data["id"])
        if "id" in data:
            del data["id"]            
        return cls(**data)
