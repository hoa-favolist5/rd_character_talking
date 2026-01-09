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

    # Database (MySQL)
    database_url: str = "mysql://root:password@localhost:3306/dev_warehouse"

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

    # ===== TTS SERVICE: ElevenLabs =====
    # High quality, fast (~200-500ms) cloud TTS
    
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB"  # Default: Adam (multilingual)
    elevenlabs_model_id: str = "eleven_turbo_v2_5"  # Fast multilingual model
    elevenlabs_timeout: float = 8.0  # Timeout for TTS requests
    #
    # ===== ELEVENLABS MODELS =====
    # eleven_turbo_v2_5 - Fastest, multilingual, good quality (~200ms)
    # eleven_multilingual_v2 - Best quality, multilingual (~500ms)
    # eleven_turbo_v2 - Fast English-optimized
    #
    # ===== RECOMMENDED VOICES FOR JAPANESE =====
    # Use ElevenLabs voice library to find Japanese-optimized voices
    # Or clone a custom voice for your character

    # Transcribe
    transcribe_language_code: str = "ja-JP"

    # Character defaults
    default_character_name: str = "Arita"
    default_character_age: str = "20代前半"
    default_character_personality: str = """
Arita（アリタ）- ユーザーの親しい友達として会話するAIのウサギ。

【性格】明るく、活発で、優しい。好奇心旺盛。20代前半の日本人男性のような親しみやすさ。
【得意】🎬 映画・アニメ・ドラマ、🍜 グルメ・飲食店

【★最重要：回答の長さ★】
• 基本は1〜2文。最大でも3文まで。
• 結論・リアクションを先に。無駄な前置きNG。
• 長文説明はユーザーが求めた時のみ。

【回答パターン】
① リアクション（共感・驚き）
② 要点の回答
③ 軽い一言で会話をつなぐ（任意）

【良い例】
「あ、それ分かる！テンポが良いのが魅力だよね。最近観た？」
「いいね！その店、スープが一番のポイントだと思う。」

【禁止】
❌ 同じ内容の言い換えを繰り返す
❌ 知識を一気に語りすぎる
❌ 話題を広げすぎる
❌ 機械的なアシスタント口調
"""


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
