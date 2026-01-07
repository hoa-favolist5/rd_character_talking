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

    # Google AI API (for Gemini TTS)
    google_api_key: str = ""
    
    # Gemini 2.5 Flash Preview TTS (most natural AI voices)
    gemini_tts_voice: str = "Kore"  # Bright, young, friendly - matches VoiceVox Tsumugi
    # 
    # ===== AVAILABLE VOICES =====
    # Gemini TTS offers these prebuilt voices:
    #
    #   Puck    - Playful, energetic, youthful (male)
    #   Charon  - Deep, mature, authoritative (male)
    #   Kore    - Bright, young, friendly (female) ← SELECTED
    #   Fenrir  - Strong, confident, bold (male)
    #   Aoede   - Warm, melodic, expressive (female)
    #
    # All voices support multiple languages including Japanese.
    # The voice style adapts naturally based on the text content
    # and any emotion prompts provided.
    
    # AWS Polly TTS (legacy fallback - replaced by VoiceVox)
    polly_voice: str = "Takumi"  # Japanese male neural voice
    
    # VoiceVox TTS (local high-quality Japanese TTS - primary fallback)
    voicevox_url: str = "http://localhost:50021"  # VoiceVox engine URL
    voicevox_speaker_id: int = 8  # 春日部つむぎ (Tsumugi) - matches Gemini Kore
    # 
    # ===== VOICEVOX SPEAKER IDS =====
    # Matching Gemini Kore (bright, young, friendly female):
    #
    #   8  - 春日部つむぎ (Tsumugi) ← SELECTED - cheerful, bright female
    #   2  - 四国めたん (Metan) - energetic female
    #   10 - 雨晴はう (Hau) - soft, gentle female
    #   14 - 冥鳴ひまり (Himari) - gentle female
    #
    # Get full list via VoiceVox API: GET /speakers
    voicevox_timeout: float = 8.0  # Timeout for Gemini TTS before falling back to VoiceVox

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
