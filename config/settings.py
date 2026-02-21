"""
config/settings.py
──────────────────
Centralised settings loaded from .env via pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"

    # Data
    data_dir: Path = Path("./data")
    excel_file: str = "EXIM_DatasetAlgo_Hackathon.xlsx"

    # Algorithm
    top_k_matches: int = 10
    similarity_threshold: float = 0.25

    # OpenAI (optional)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("./logs/app.log")

    @property
    def excel_path(self) -> Path:
        return self.data_dir / self.excel_file


@lru_cache
def get_settings() -> Settings:
    return Settings()
