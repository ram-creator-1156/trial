"""
Application configuration loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # General
    APP_ENV: str = "development"
    APP_DEBUG: bool = True

    # Backend server
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    # OpenAI / LangChain
    OPENAI_API_KEY: str = ""

    # Data paths
    DATA_FILE: str = "data/EXIM_DatasetAlgo_Hackathon.xlsx"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
