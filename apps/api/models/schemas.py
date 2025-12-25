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


class TextMessageRequest(BaseModel):
    """Text message request body."""

    content: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None


class VoiceMessageRequest(BaseModel):
    """Voice message request body with pre-transcribed text."""

    transcript: str = Field(..., min_length=1, max_length=10000, description="Pre-transcribed text from frontend")
    s3_key: Optional[str] = Field(None, description="S3 key of the uploaded audio file (optional)")
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response body."""

    text: str
    audio_url: Optional[str] = None
    emotion: EmotionType = EmotionType.IDLE
    user_transcript: Optional[str] = None
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
