from typing import Optional
from urllib.parse import quote_plus
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise AI/ML FastAPI Platform"
    API_V1_STR: str = "/api/v1"

    # Pydantic v2 Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
