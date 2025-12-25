"""FastAPI main application entry point."""

import os

from config.settings import get_settings as _get_settings

# Set Anthropic API key for CrewAI
_settings = _get_settings()
if _settings.anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = _settings.anthropic_api_key

# Set a dummy OpenAI key to prevent CrewAI from complaining
os.environ.setdefault("OPENAI_API_KEY", "not-needed-using-anthropic")

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import socketio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from crews.character_crew import create_character_crew
from db.connection import db
from models.schemas import (
    ChatResponse,
    EmotionType,
    ErrorResponse,
    HealthCheck,
    S3UploadUrlRequest,
    S3UploadUrlResponse,
    TextMessageRequest,
    TranscribeCredentialsResponse,
    VoiceMessageRequest,
)
from services.credentials import credentials_service


settings = get_settings()

# Socket.IO setup
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.cors_origins,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    import asyncio
    
    # Startup
    print("Starting up...")
    await db.connect()
    print("Database connected")
    
    # Warm up connection pool with parallel queries
    print("Warming up connection pool...")
    try:
        await asyncio.gather(*[db.fetchval("SELECT 1") for _ in range(5)])
        print("Connection pool warmed up")
    except Exception as e:
        print(f"Pool warmup warning: {e}")

    yield

    # Shutdown
    print("Shutting down...")
    await db.disconnect()
    print("Database disconnected")


# FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI Character Voice Interaction API",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Socket.IO
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Character crew instance (lazy loaded)
_crew = None


def get_crew():
    """Get or create character crew instance."""
    global _crew
    if _crew is None:
        _crew = create_character_crew(
            character_name=settings.default_character_name,
            personality=settings.default_character_personality,
            voice_id=settings.polly_voice_id,
        )
    return _crew


# REST API Endpoints


@app.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """Health check endpoint."""
    try:
        await db.fetchval("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return HealthCheck(
        status="healthy",
        version="1.0.0",
        database=db_status,
        aws="configured" if settings.aws_access_key_id else "not_configured",
        llm="anthropic" if settings.anthropic_api_key else "not_configured",
    )


@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(request: TextMessageRequest) -> ChatResponse:
    """
    Handle text-based chat messages.

    Args:
        request: Text message request

    Returns:
        Chat response with text, audio URL, and emotion
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        crew = get_crew()
        result = await crew.process_message(
            user_message=request.content,
            session_id=session_id,
        )

        return ChatResponse(
            text=result["text"],
            audio_url=result.get("audio_url"),
            emotion=EmotionType(result.get("emotion", "idle")),
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/voice", response_model=ChatResponse)
async def chat_voice(request: VoiceMessageRequest) -> ChatResponse:
    """
    Handle voice-based chat messages with pre-transcribed text.

    The frontend performs real-time transcription using AWS Transcribe Streaming
    and sends the transcript directly. Audio is optionally uploaded to S3.

    Args:
        request: Voice message with transcript and optional S3 key

    Returns:
        Chat response with text, audio URL, and emotion
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Process with crew (transcript is already available from frontend)
        crew = get_crew()
        result = await crew.process_message(
            user_message=request.transcript,
            session_id=session_id,
        )

        return ChatResponse(
            text=result["text"],
            audio_url=result.get("audio_url"),
            emotion=EmotionType(result.get("emotion", "idle")),
            user_transcript=request.transcript,
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/credentials/transcribe", response_model=TranscribeCredentialsResponse)
async def get_transcribe_credentials() -> dict:
    """
    Get temporary AWS credentials for Transcribe Streaming.
    
    These credentials allow the frontend to use AWS Transcribe Streaming
    for real-time speech-to-text transcription.
    
    Returns:
        Temporary AWS credentials with limited permissions
    """
    try:
        return credentials_service.get_transcribe_credentials()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/credentials/s3-upload", response_model=S3UploadUrlResponse)
async def get_s3_upload_url(request: S3UploadUrlRequest) -> dict:
    """
    Get a pre-signed URL for S3 audio upload.
    
    Args:
        request: Upload request with filename and content type
        
    Returns:
        Pre-signed URL and S3 key for upload
    """
    try:
        s3_key = f"voice/{uuid.uuid4().hex}/{request.filename}"
        return credentials_service.get_s3_upload_url(s3_key, request.content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Socket.IO Event Handlers


@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    print(f"Client disconnected: {sid}")


@sio.event
async def message(sid, data):
    """
    Handle incoming messages via WebSocket.

    Expected data format:
    {
        "type": "text" | "voice",
        "content": "message text",
        "transcript": "transcribed text" (for voice, from frontend STT),
        "s3Key": "optional/s3/key" (for voice),
        "sessionId": "session-id"
    }
    
    Note: Voice messages now send pre-transcribed text from frontend.
    The frontend uses AWS Transcribe Streaming for real-time STT.
    """
    try:
        msg_type = data.get("type", "text")
        session_id = data.get("sessionId") or sid

        # Get the text content - for voice, use transcript from frontend STT
        if msg_type == "voice":
            content = data.get("transcript", "")
            s3_key = data.get("s3Key")  # Optional: S3 key for audio backup
            if not content:
                await sio.emit("error", {"message": "No transcript provided"}, room=sid)
                return
        else:
            content = data.get("content", "")

        if not content.strip():
            await sio.emit("error", {"message": "Empty message"}, room=sid)
            return

        # Process message
        crew = get_crew()
        print(f"[WS DEBUG] Processing message for {sid}: {content[:50]}...")
        
        result = await crew.process_message(
            user_message=content,
            session_id=session_id,
        )
        
        print(f"[WS DEBUG] Got result, sending to {sid}")
        print(f"[WS DEBUG] Response text: {result.get('text', 'NO TEXT')[:100]}")

        # Send response
        await sio.emit(
            "response",
            {
                "text": result["text"],
                "audioUrl": result.get("audio_url"),
                "emotion": result.get("emotion", "idle"),
                "userTranscript": content if msg_type == "voice" else None,
            },
            room=sid,
        )
        print(f"[WS DEBUG] Emit completed for {sid}")

    except Exception as e:
        print(f"[WS ERROR] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit("error", {"message": str(e)}, room=sid)


# Export the ASGI app for uvicorn
app = socket_app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )

