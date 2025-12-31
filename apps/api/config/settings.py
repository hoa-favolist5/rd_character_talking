from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Character AI API"
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]

    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/dev_warehouse"

    # AWS
    aws_region: str = "ap-northeast-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # S3
    s3_bucket_audio: str = "character-audio"
    s3_bucket_static: str = "character-static"

    # Anthropic API
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_fast_model: str = "claude-3-haiku-20240307"  # Fast model for conversation
    # Models: claude-sonnet-4-20250514, claude-opus-4-20250514, claude-3-5-sonnet-20241022
    # Fast models: claude-3-haiku-20240307 (fastest, ~100ms)

    # Polly
    polly_voice_id: str = "Takumi"  # Japanese male voice
    polly_engine: str = "neural"

    # Transcribe
    transcribe_language_code: str = "ja-JP"

    # Character defaults
    default_character_name: str = "Ai"
    default_character_personality: str = "A kind and knowledgeable AI assistant. Committed to polite communication and clear explanations."


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

