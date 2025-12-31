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

    # VOICEVOX (local, free, very natural Japanese voices)
    # Download from: https://voicevox.hiroshiba.jp/
    # Run VOICEVOX app, it starts API on localhost:50021
    voicevox_url: str = "http://localhost:50021"
    voicevox_speaker: int = 12  #白上虎太郎 (ふつう) ← YOUNG BOY
    # 
    # ===== MALE VOICES =====
    #  11: 玄野武宏 (ノーマル) ← YOUNG MALE, natural
    #  39: 玄野武宏 (喜び) - happy
    #  40: 玄野武宏 (ツンギレ) - tsundere
    #  41: 玄野武宏 (悲しみ) - sad
    #  ---
    #  12: 白上虎太郎 (ふつう) ← YOUNG BOY
    #  32: 白上虎太郎 (わーい) - excited
    #  ---
    #  13: 青山龍星 (ノーマル) ← MATURE MALE
    #  81: 青山龍星 (熱血) - passionate
    #  84: 青山龍星 (しっとり) - gentle
    #  86: 青山龍星 (囁き) - whisper
    #  ---
    #  21: 剣崎雌雄 (ノーマル) ← DEEP MALE
    #  51: †聖騎士 紅桜† (ノーマル) - knight style
    #  52: 雀松朱司 (ノーマル)
    #  53: 麒ヶ島宗麟 (ノーマル)
    #
    # ===== POPULAR FEMALE =====
    #   3: ずんだもん (ノーマル) - cute, friendly
    #   8: 春日部つむぎ (ノーマル) - cheerful
    #  14: 冥鳴ひまり (ノーマル) - gentle
    #  20: もち子さん (ノーマル) - mature

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

