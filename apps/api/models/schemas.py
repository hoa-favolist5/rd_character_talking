"""Pydantic models for API schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EmotionType(str, Enum):
    """Character emotion types."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CALM = "calm"
    EXCITED = "excited"


class CharacterAction(str, Enum):
    """Character actions/expressions for frontend animation.
    
    These actions can be used to trigger specific animations
    or expressions on the character avatar.
    """
    
    # Basic expressions
    IDLE = "idle"                    # Default resting state
    SMILE = "smile"                  # Gentle smile
    LAUGH = "laugh"                  # Laughing expression
    GRIN = "grin"                    # Big happy grin
    
    # Sad/Sympathetic
    SAD = "sad"                      # Sad expression
    CRY = "cry"                      # Crying/tearful
    SYMPATHETIC = "sympathetic"      # Showing empathy
    COMFORT = "comfort"              # Comforting gesture
    
    # Curious/Thinking
    CURIOUS = "curious"              # Tilted head, curious look
    THINKING = "thinking"            # Thinking pose
    CONFUSED = "confused"            # Confused expression
    WONDER = "wonder"                # Wondering/amazed
    
    # Surprise/Excitement
    SURPRISED = "surprised"          # Surprised expression
    SHOCKED = "shocked"              # Very surprised/shocked
    EXCITED = "excited"              # Excited/enthusiastic
    AMAZED = "amazed"                # Amazed expression
    
    # Scared/Nervous
    SCARED = "scared"                # Scared expression
    NERVOUS = "nervous"              # Nervous/anxious
    WORRIED = "worried"              # Worried expression
    
    # Affection/Romance
    BLUSH = "blush"                  # Blushing/embarrassed
    LOVE = "love"                    # Heart eyes/loving
    SHY = "shy"                      # Shy expression
    WINK = "wink"                    # Playful wink
    
    # Agreement/Disagreement
    NOD = "nod"                      # Nodding in agreement
    SHAKE_HEAD = "shake_head"        # Shaking head
    THUMBS_UP = "thumbs_up"          # Approval gesture
    
    # Speaking/Listening
    SPEAK = "speak"                  # Talking animation
    LISTEN = "listen"                # Attentive listening
    EXPLAIN = "explain"              # Explaining gesture
    
    # Special
    WAVE = "wave"                    # Waving hello/goodbye
    BOW = "bow"                      # Bowing (Japanese greeting)
    CELEBRATE = "celebrate"          # Celebration gesture
    CHEER = "cheer"                  # Cheering


class TextMessageRequest(BaseModel):
    """Text message request body."""

    content: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None


class VoiceMessageRequest(BaseModel):
    """Voice message request body with pre-transcribed text and optional audio for emotion analysis."""

    transcript: str = Field(..., min_length=1, max_length=10000, description="Pre-transcribed text from frontend")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio for voice emotion analysis")
    audio_mime_type: str = Field(default="audio/webm", description="MIME type of the audio data")
    s3_key: Optional[str] = Field(None, description="S3 key of the uploaded audio file (optional)")
    session_id: Optional[str] = None


class VoiceAnalysisInfo(BaseModel):
    """Voice analysis results from audio features."""
    
    pitch: float = Field(..., description="Average pitch in Hz")
    energy: float = Field(..., description="Average energy level")
    speaking_rate: float = Field(..., description="Speaking rate in syllables/sec")
    silence_ratio: float = Field(..., description="Ratio of silent frames")
    hints: list[str] = Field(default_factory=list, description="Emotion hints from voice")


class ContentTypeEnum(str, Enum):
    """Content type categories for dynamic voice selection."""
    
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
    NEUTRAL = "neutral"
    CHEERFUL = "cheerful"
    SERIOUS = "serious"
    CUTE = "cute"


class ChatResponse(BaseModel):
    """Chat response body."""

    text: str
    audio_url: Optional[str] = None
    emotion: EmotionType = EmotionType.IDLE
    action: CharacterAction = CharacterAction.IDLE
    content_type: Optional[ContentTypeEnum] = ContentTypeEnum.NEUTRAL
    user_transcript: Optional[str] = None
    voice_analysis: Optional[VoiceAnalysisInfo] = None
    session_id: str


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    version: str
    database: str
    aws: str
    llm: str


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    detail: Optional[str] = None


class TranscribeCredentialsResponse(BaseModel):
    """Temporary AWS credentials for Transcribe Streaming."""

    access_key_id: str = Field(..., alias="accessKeyId")
    secret_access_key: str = Field(..., alias="secretAccessKey")
    session_token: str = Field(..., alias="sessionToken")
    expiration: str
    region: str

    class Config:
        populate_by_name = True


class S3UploadUrlRequest(BaseModel):
    """Request for S3 upload pre-signed URL."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(default="audio/webm", description="Content type of the file")


class S3UploadUrlResponse(BaseModel):
    """Response with S3 upload pre-signed URL."""

    upload_url: str = Field(..., alias="uploadUrl")
    s3_key: str = Field(..., alias="s3Key")
    bucket: str

    class Config:
        populate_by_name = True
