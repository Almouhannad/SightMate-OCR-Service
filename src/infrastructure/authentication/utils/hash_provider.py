import threading
from passlib.context import CryptContext
from typing import Any


class _HashProviderMeta(type):
    """Metaclass that ensures a single instance of HashProvider."""
    _instance: Any = None
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking to ensure thread safety
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class HashProvider(metaclass=_HashProviderMeta):
    """
    Singleton class providing API key hashing and verification,
    with a single CryptContext created on first instantiation.
    """

    def __init__(self) -> None:
        # Initialize CryptContext only once per singleton instance
        if not hasattr(self, "_pwd_context"):
            self._pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_api_key(self, plain_key: str) -> str:
        """
        Hashes an API key for storage.
        """
        return self._pwd_context.hash(plain_key)

    def verify_api_key(self, plain_key: str, hashed_key: str) -> bool:
        """
        Verifies a plain API key against the stored bcrypt hash.
        """
        return self._pwd_context.verify(plain_key, hashed_key)
