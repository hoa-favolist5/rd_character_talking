"""Voice configuration for dynamic voice selection based on content type."""

from dataclasses import dataclass
from enum import Enum


class ContentType(str, Enum):
    """Content types that affect voice selection."""
    
    # Film/Show genres
    COMEDY = "comedy"
    HORROR = "horror"
    THRILLER = "thriller"
    ROMANCE = "romance"
    DRAMA = "drama"
    ACTION = "action"
    CHILDREN = "children"
    ANIMATION = "animation"
    DOCUMENTARY = "documentary"
    SCIFI = "scifi"
    FANTASY = "fantasy"
    MYSTERY = "mystery"
    
    # General categories
    NEUTRAL = "neutral"
    CHEERFUL = "cheerful"
    SERIOUS = "serious"
    CUTE = "cute"


@dataclass
class VoiceConfig:
    """Configuration for a specific voice."""
    
    voice_id: str
    engine: str = "neural"
    rate: str = "100%"  # Speech rate
    pitch: str = "0%"   # Pitch adjustment
    description: str = ""


# AWS Polly Japanese voices
# - Takumi (Male, Neural) - default, professional
# - Mizuki (Female, Standard) - mature female
# - Kazuha (Female, Neural) - young adult female
# - Tomoko (Female, Neural) - warm female

# Voice configurations for different content types
# Note: AWS Polly Neural voices do NOT support 'pitch' in SSML prosody tag.
# Only 'rate' adjustments are supported, so we use different voices and rates
# to create variety.
VOICE_CONFIGS: dict[ContentType, VoiceConfig] = {
    # Comedy - cheerful, slightly faster (female voice for warmth)
    ContentType.COMEDY: VoiceConfig(
        voice_id="Kazuha",
        rate="110%",
        pitch="0%",  # Not used for neural
        description="Cheerful and upbeat for comedy content",
    ),
    
    # Horror/Thriller - slower (male voice for gravitas)
    ContentType.HORROR: VoiceConfig(
        voice_id="Takumi",
        rate="85%",
        pitch="0%",
        description="Slower voice for suspense",
    ),
    ContentType.THRILLER: VoiceConfig(
        voice_id="Takumi",
        rate="90%",
        pitch="0%",
        description="Tense, measured delivery",
    ),
    
    # Romance/Drama - soft, emotional female voice (slower for emotion)
    ContentType.ROMANCE: VoiceConfig(
        voice_id="Kazuha",
        rate="92%",
        pitch="0%",
        description="Soft, romantic female voice",
    ),
    ContentType.DRAMA: VoiceConfig(
        voice_id="Kazuha",
        rate="95%",
        pitch="0%",
        description="Emotional, expressive voice",
    ),
    
    # Children/Animation - bright, faster (female voice for friendliness)
    ContentType.CHILDREN: VoiceConfig(
        voice_id="Kazuha",
        rate="115%",
        pitch="0%",
        description="Bright, child-friendly voice",
    ),
    ContentType.ANIMATION: VoiceConfig(
        voice_id="Kazuha",
        rate="108%",
        pitch="0%",
        description="Animated, energetic voice",
    ),
    
    # Action - strong, confident male voice
    ContentType.ACTION: VoiceConfig(
        voice_id="Takumi",
        rate="105%",
        pitch="0%",
        description="Strong, confident voice for action",
    ),
    
    # Sci-Fi/Fantasy
    ContentType.SCIFI: VoiceConfig(
        voice_id="Takumi",
        rate="95%",
        pitch="0%",
        description="Slightly futuristic tone",
    ),
    ContentType.FANTASY: VoiceConfig(
        voice_id="Kazuha",
        rate="100%",
        pitch="0%",
        description="Magical, whimsical tone",
    ),
    
    # Documentary - clear, professional (slower for clarity)
    ContentType.DOCUMENTARY: VoiceConfig(
        voice_id="Takumi",
        rate="90%",
        pitch="0%",
        description="Clear, professional narration",
    ),
    
    # Mystery - intriguing, measured
    ContentType.MYSTERY: VoiceConfig(
        voice_id="Takumi",
        rate="88%",
        pitch="0%",
        description="Mysterious, intriguing tone",
    ),
    
    # General categories
    ContentType.CHEERFUL: VoiceConfig(
        voice_id="Kazuha",
        rate="110%",
        pitch="0%",
        description="Happy, upbeat voice",
    ),
    ContentType.CUTE: VoiceConfig(
        voice_id="Kazuha",
        rate="105%",
        pitch="0%",
        description="Cute, kawaii voice",
    ),
    ContentType.SERIOUS: VoiceConfig(
        voice_id="Takumi",
        rate="90%",
        pitch="0%",
        description="Serious, professional tone",
    ),
    ContentType.NEUTRAL: VoiceConfig(
        voice_id="Takumi",
        rate="100%",
        pitch="0%",
        description="Default neutral voice",
    ),
}

# Keywords to detect content type from conversation
CONTENT_TYPE_KEYWORDS: dict[ContentType, list[str]] = {
    ContentType.COMEDY: [
        "comedy", "funny", "laugh", "humor", "hilarious", "コメディ", "お笑い", "面白い", "ギャグ",
        "comic", "sitcom", "parody", "comedic",
    ],
    ContentType.HORROR: [
        "horror", "scary", "ghost", "zombie", "fear", "ホラー", "怖い", "恐怖", "幽霊", "お化け",
        "creepy", "terror", "nightmare", "haunted", "monster",
    ],
    ContentType.THRILLER: [
        "thriller", "suspense", "tension", "crime", "サスペンス", "スリラー", "緊張", "犯罪",
        "detective", "mystery", "dangerous",
    ],
    ContentType.ROMANCE: [
        "romance", "love", "romantic", "couple", "恋愛", "ロマンス", "ラブ", "恋", "愛",
        "dating", "relationship", "wedding", "kiss", "カップル",
    ],
    ContentType.DRAMA: [
        "drama", "emotional", "tear", "touching", "ドラマ", "感動", "泣ける", "感情",
        "moving", "heartfelt", "深い",
    ],
    ContentType.CHILDREN: [
        "children", "kids", "child", "family", "子供", "キッズ", "ファミリー", "子ども", "児童",
        "animated", "cartoon", "disney", "pixar", "ghibli", "ジブリ", "アニメ映画",
        "educational", "cute animal",
    ],
    ContentType.ANIMATION: [
        "animation", "anime", "animated", "アニメ", "アニメーション", "cartoon",
        "manga", "漫画", "ジャパニメーション",
    ],
    ContentType.ACTION: [
        "action", "fight", "battle", "explosion", "アクション", "戦い", "バトル", "爆発",
        "martial arts", "war", "combat", "hero", "superhero", "ヒーロー",
    ],
    ContentType.SCIFI: [
        "sci-fi", "science fiction", "space", "future", "robot", "SF", "宇宙", "未来", "ロボット",
        "alien", "technology", "cyberpunk", "dystopia",
    ],
    ContentType.FANTASY: [
        "fantasy", "magic", "wizard", "dragon", "ファンタジー", "魔法", "魔女", "ドラゴン",
        "mythical", "fairy", "enchanted", "kingdom", "エルフ", "妖精",
    ],
    ContentType.DOCUMENTARY: [
        "documentary", "documentary", "history", "nature", "ドキュメンタリー", "歴史", "自然",
        "real story", "true story", "実話", "記録",
    ],
    ContentType.MYSTERY: [
        "mystery", "detective", "puzzle", "clue", "ミステリー", "探偵", "謎", "推理",
        "whodunit", "secret", "秘密",
    ],
}


def detect_content_type(text: str) -> ContentType:
    """
    Detect content type from text using keyword matching.
    
    Args:
        text: Text to analyze (user message or conversation context)
        
    Returns:
        Detected ContentType, defaults to NEUTRAL
    """
    text_lower = text.lower()
    
    # Count keyword matches for each content type
    scores: dict[ContentType, int] = {}
    
    for content_type, keywords in CONTENT_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[content_type] = score
    
    if not scores:
        return ContentType.NEUTRAL
    
    # Return content type with highest score
    return max(scores, key=lambda ct: scores[ct])


def get_voice_config(content_type: ContentType) -> VoiceConfig:
    """
    Get voice configuration for a content type.
    
    Args:
        content_type: The detected content type
        
    Returns:
        VoiceConfig for the content type
    """
    return VOICE_CONFIGS.get(content_type, VOICE_CONFIGS[ContentType.NEUTRAL])

