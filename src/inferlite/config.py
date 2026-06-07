from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERLITE_", extra="ignore")

    default_model: str = Field(default="gpt2")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    sqlite_path: str = Field(default="inferlite.db")
    admin_bootstrap_secret: str = Field(default="inferlite-admin")


settings = Settings()
