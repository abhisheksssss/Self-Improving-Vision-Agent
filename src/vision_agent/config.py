from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Provider Options: gemini | openai_compatible | groq | ollama
    VISION_PROVIDER: str = "gemini"
    VISION_MODEL: str = "gemini-2.5-flash"

    REASONING_PROVIDER: str = "gemini"
    REASONING_MODEL: str = "gemini-2.5-flash"

    # API Credentials
    GEMINI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    CEBREAS_API_KEY: str = ""

    # Generic OpenAI-Compatible Settings
    OPENAI_COMPATIBLE_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENAI_COMPATIBLE_API_KEY: str = ""

    # Safety & Execution Controls
    SAFETY_MAX_STEPS: int = 25
    EMERGENCY_HOTKEY: str = "ctrl+alt+esc"
    DATA_DIR: str = "./data"
    LOG_LEVEL: str = "INFO"

    # Memory & Storage Settings
    SQLITE_DB_PATH: str = "./data/sivac.db"
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    @property
    def data_path(self) -> Path:
        p = Path(self.DATA_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = Path(self.CHROMA_PERSIST_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sqlite_db_url(self) -> str:
        # Ensure parent directory exists
        db_file = Path(self.SQLITE_DB_PATH).resolve()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        # SQLite URL with forward slashes
        return f"sqlite:///{db_file.as_posix()}"


settings = Settings()
