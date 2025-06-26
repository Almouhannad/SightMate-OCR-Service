import os
from enum import Enum
from typing import Dict
from dotenv import dotenv_values

# Load .env but allow override by real environment
_local_vars: Dict[str, str | None] = dotenv_values()


class ConfigField(Enum):
    RUNNING_ON                     = "RUNNING_ON"
    OCR_ADAPTER                    = "OCR_ADAPTER"
    LMS_API_BASE_URI_FOR_HOST      = "LMS_API_BASE_URI_FOR_HOST"
    LMS_API_BASE_URI_FOR_CONTAINER = "LMS_API_BASE_URI_FOR_CONTAINER"
    MONGODB_URI                    = "MONGODB_URI"
    MONGO_DATABASE                 = "MONGO_DATABASE"
    API_KEY_REPOSITORY             = "API_KEY_REPOSITORY"


class AppConfig:
    """Application settings, loaded from .env or environment."""

    @staticmethod
    def _get(field: ConfigField, default: str) -> str:
        """
        Retrieve a setting by:
          1. checking .env values (FOR HOST),
          2. then os.environ (with `default`) (FOR CONTAINER),
        """
        # 1) Try the parsed .env file
        val = _local_vars.get(field.value)
        if val is not None:
            return val

        # 2) Fallback to real environment, supplying a str default
        return os.getenv(field.value, default)

    @property
    def running_on(self) -> str:
        return self._get(ConfigField.RUNNING_ON, "container")

    @property
    def ocr_adapter(self) -> str:
        return self._get(ConfigField.OCR_ADAPTER, "easyocr")

    @property
    def lms_api(self) -> str:
        # choose host or container URI based on running_on
        if self.running_on == "host":
            return self._get(ConfigField.LMS_API_BASE_URI_FOR_HOST, "")
        return self._get(ConfigField.LMS_API_BASE_URI_FOR_CONTAINER, "")

    @property
    def mongodb_uri(self) -> str:
        return self._get(ConfigField.MONGODB_URI, "")

    @property
    def mongodb_database(self) -> str:
        return self._get(ConfigField.MONGO_DATABASE, "")
    
    @property
    def api_key_repository(self) -> str:
        return self._get(ConfigField.API_KEY_REPOSITORY, "")    


# Single, module‐level instance
CONFIG = AppConfig()
