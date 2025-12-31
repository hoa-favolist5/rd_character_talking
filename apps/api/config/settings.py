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
    anthropic_model: str = "claude-3-5-sonnet-20241022"  # Best for natural conversation
    anthropic_fast_model: str = "claude-3-5-haiku-20241022"  # Smarter + still fast
    # Model comparison:
    #   claude-3-5-sonnet: Most natural, human-like, smart (~300ms)
    #   claude-3-5-haiku:  Fast + smart, good for quick replies (~100ms)
    #   claude-sonnet-4:   Latest but can be slower
    #   claude-3-haiku:    Fastest but less natural

    # Google Cloud Text-to-Speech (high-quality, natural Japanese voices)
    # Set GOOGLE_APPLICATION_CREDENTIALS env var to service account JSON path
    google_tts_voice: str = "ja-JP-Neural2-D"  # Young male, natural
    google_tts_language: str = "ja-JP"
    # 
    # ===== JAPANESE VOICES (Most Natural) =====
    # Neural2 voices (most natural, recommended):
    #   ja-JP-Neural2-B: Female, warm and natural
    #   ja-JP-Neural2-C: Male, deep and calm
    #   ja-JP-Neural2-D: Male, young and energetic ← RECOMMENDED for young character
    #
    # Wavenet voices (very natural):
    #   ja-JP-Wavenet-A: Female, warm
    #   ja-JP-Wavenet-B: Female, cheerful
    #   ja-JP-Wavenet-C: Male, mature
    #   ja-JP-Wavenet-D: Male, calm
    #
    # Standard voices (faster, less natural):
    #   ja-JP-Standard-A through D
    #
    # Journey voices (conversational, expressive):
    #   ja-JP-Journey-D: Male, conversational
    #   ja-JP-Journey-F: Female, conversational

    # Transcribe
    transcribe_language_code: str = "ja-JP"

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
