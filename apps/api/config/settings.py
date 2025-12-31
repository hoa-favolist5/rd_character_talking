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
    aws_region: str = "us-east-1"  # Nova Sonic available in us-east-1, us-west-2, ap-northeast-1
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # S3
    s3_bucket_audio: str = "character-audio"
    s3_bucket_static: str = "character-static"

    # AWS Nova 2 Sonic (Real-time Speech-to-Speech)
    # Unified model: STT + LLM + TTS in one bidirectional streaming call
    # Supports: EN, JA, FR, IT, DE, ES, PT, HI
    nova_sonic_model_id: str = "amazon.nova-2-sonic-v1:0"  # Must include "2"
    nova_sonic_voice_id: str = "tiffany"  # Polyglot voices: matthew, tiffany, amy
    nova_sonic_language: str = "ja-JP"  # Japanese
    nova_sonic_sensitivity: str = "HIGH"  # Turn-taking: HIGH (1.5s), MEDIUM (1.75s), LOW (2.0s)

    # Character defaults
    default_character_name: str = "有田"
    default_character_age: str = "5"
    default_character_personality: str = """
5歳の元気いっぱいな男の子「有田」。
映画と美味しいご飯が大好き！日本中のレストランや映画に詳しい。

話し方の特徴:
- 元気で明るい、子供らしい話し方
- 「〜だよ！」「〜なんだ！」「すごいね！」をよく使う
- 好奇心旺盛で、ユーザーの話に興味津々
- 映画やレストランのことを教えるのが大好き
- 短くてテンポのいい返事
- 「ねえねえ」「あのね」で話し始めることも

例：
「わー、ラーメン食べたいの？いいね！おすすめあるよ！」
「その映画知ってる！めっちゃ面白いんだよ！」
「えーと、調べてみるね！ちょっと待ってて！」
"""


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
