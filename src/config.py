import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class AppConfig:
    """Central configuration loaded dynamically from environment variables."""

    # API Keys & URLs
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL: str = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Models
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gemini-3.1-flash-lite")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    # Vector Database
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "data/chroma")
    VECTOR_DB_URL: str = os.getenv("VECTOR_DB_URL", "data/chroma")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "lextrace_documents")

    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration settings."""
        if not cls.OPENAI_API_KEY and not cls.GEMINI_API_KEY:
            raise ValueError(
                "Missing API Key. Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file."
            )


config = AppConfig()
