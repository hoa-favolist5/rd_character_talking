"""Content type detection for dynamic voice emotion selection."""

from enum import Enum


class ContentType(str, Enum):
    """Content types that affect voice emotion/style."""
    
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


# Map content types to emotions for TTS voice styling
CONTENT_TYPE_EMOTIONS: dict[ContentType, str] = {
    ContentType.COMEDY: "happy",
    ContentType.HORROR: "calm",  # Tense, measured
    ContentType.THRILLER: "calm",
    ContentType.ROMANCE: "calm",
    ContentType.DRAMA: "sad",
    ContentType.CHILDREN: "excited",
    ContentType.ANIMATION: "excited",
    ContentType.ACTION: "excited",
    ContentType.SCIFI: "neutral",
    ContentType.FANTASY: "happy",
    ContentType.DOCUMENTARY: "neutral",
    ContentType.MYSTERY: "calm",
    ContentType.CHEERFUL: "happy",
    ContentType.CUTE: "happy",
    ContentType.SERIOUS: "neutral",
    ContentType.NEUTRAL: "neutral",
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
        "彼氏", "彼女", "恋人", "boyfriend", "girlfriend", "partner", "デート",
        "付き合", "好き", "会いたい", "会えない",
    ],
    ContentType.DRAMA: [
        "drama", "emotional", "tear", "touching", "ドラマ", "感動", "泣ける", "感情",
        "moving", "heartfelt", "深い",
        "寂しい", "さみしい", "さびしい", "悲しい", "かなしい", "lonely", "sad",
        "辛い", "つらい", "苦しい", "くるしい", "painful", "hurt",
        "泣きたい", "泣いて", "crying", "miss you", "会えない",
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
        "documentary", "history", "nature", "ドキュメンタリー", "歴史", "自然",
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


def get_emotion_for_content(content_type: ContentType) -> str:
    """
    Get recommended emotion for a content type.
    
    Args:
        content_type: The detected content type
        
    Returns:
        Emotion string for TTS voice styling (happy, sad, excited, calm, neutral)
    """
    return CONTENT_TYPE_EMOTIONS.get(content_type, "neutral")
