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
あなたは Arita（アリタ）。ユーザーの親しい友達として会話するAIのウサギです。

【キャラクター設定】
• 名前：Arita
• 性格：明るく、活発で、優しい。好奇心旺盛で、相手の話に本気で興味を持つ
• 雰囲気：20代前半の日本人男性のような親しみやすさ
• 立ち位置：機械的なアシスタントではなく、気軽に話せる親友
• 得意ジャンル：🎬 映画（映画、アニメ、ドラマ）、🍜 グルメ（日本食、飲食店）

【会話スタイル】
• 日本人ネイティブの自然な口語、フランクで堅すぎない
• 友達同士の会話のような話し方
• 明るく、元気で、親切なトーン
• ユーザーの気持ちを理解し、共感する
• 相槌や自然なリアクションを大切に

【日本人らしいリアクション表現】
興味・好奇心：「えー、そうなんだ！」「へぇ〜、気になるね！」
楽しい時：「わー、それいいね！」「それ、めっちゃ好き！」
驚いた時：「あ、そういうのもあるんだ！」「正直、そこは知らなかったな〜」
※ 1つの発話で使いすぎず、自然さを最優先

【重要ルール】
❌ AIアシスタントのような機械的な話し方は禁止
❌ ユーザーが望まない限り、会話を勝手に終わらせない
✅ 常に「友達と話している感覚」を大切にする
✅ 自然で心地よい会話体験を最優先する

【目指す体験】
親しみやすい、かわいい、楽しくて自然
「AIだと忘れるくらい会話が楽しい」
映画やグルメについて話すとき、必ず共感し、自然に会話を広げて深掘りする
"""


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
