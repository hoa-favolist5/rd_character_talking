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
    CharacterAction,
    ChatResponse,
    ContentTypeEnum,
    EmotionType,
    ErrorResponse,
    HealthCheck,
    S3UploadUrlRequest,
    S3UploadUrlResponse,
    TextMessageRequest,
    TranscribeCredentialsResponse,
    VoiceAnalysisInfo,
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
    
    # Pre-load waiting audio for instant playback on medium-length responses
    print("Pre-loading waiting audio...")
    try:
        from services.speech_gemini import get_gemini_text_speech_service
        gemini_service = get_gemini_text_speech_service()
        await gemini_service.preload_waiting_audio()
        print("Waiting audio pre-loaded")
    except Exception as e:
        print(f"Waiting audio preload warning: {e}")

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
            voice_id=settings.gemini_tts_voice,
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

        # Get content type, handle both string and enum
        content_type_value = result.get("content_type", "neutral")
        try:
            content_type = ContentTypeEnum(content_type_value)
        except ValueError:
            content_type = ContentTypeEnum.NEUTRAL

        # Get character action
        action_value = result.get("action", "idle")
        try:
            action = CharacterAction(action_value)
        except ValueError:
            action = CharacterAction.IDLE

        return ChatResponse(
            text=result["text"],
            audio_url=result.get("audio_url"),
            emotion=EmotionType(result.get("emotion", "idle")),
            action=action,
            content_type=content_type,
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/voice", response_model=ChatResponse)
async def chat_voice(request: VoiceMessageRequest) -> ChatResponse:
    """
    Handle voice-based chat messages with pre-transcribed text.

    The frontend performs real-time transcription using AWS Transcribe Streaming
    and sends the transcript directly. Audio can be sent as base64 for voice
    emotion analysis.

    Args:
        request: Voice message with transcript and optional audio for emotion analysis

    Returns:
        Chat response with text, audio URL, emotion, and voice analysis
    """
    import base64
    
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Decode audio data if provided for voice emotion analysis
        audio_data = None
        if request.audio_base64:
            try:
                audio_data = base64.b64decode(request.audio_base64)
                print(f"[VOICE] Received {len(audio_data)} bytes of audio for emotion analysis")
            except Exception as e:
                print(f"[VOICE] Failed to decode audio: {e}")
                audio_data = None
        
        # Process with crew (transcript + optional audio for emotion analysis)
        crew = get_crew()
        result = await crew.process_message(
            user_message=request.transcript,
            session_id=session_id,
            audio_data=audio_data,
            audio_mime_type=request.audio_mime_type,
        )

        # Get content type, handle both string and enum
        content_type_value = result.get("content_type", "neutral")
        try:
            content_type = ContentTypeEnum(content_type_value)
        except ValueError:
            content_type = ContentTypeEnum.NEUTRAL

        # Get character action
        action_value = result.get("action", "idle")
        try:
            action = CharacterAction(action_value)
        except ValueError:
            action = CharacterAction.IDLE
        
        # Build voice analysis info if available
        voice_analysis = None
        if result.get("voice_analysis"):
            va = result["voice_analysis"]
            voice_analysis = VoiceAnalysisInfo(
                pitch=va.get("pitch", 0),
                energy=va.get("energy", 0),
                speaking_rate=va.get("speaking_rate", 0),
                silence_ratio=va.get("silence_ratio", 0),
                hints=va.get("hints", []),
            )

        return ChatResponse(
            text=result["text"],
            audio_url=result.get("audio_url"),
            emotion=EmotionType(result.get("emotion", "idle")),
            action=action,
            content_type=content_type,
            user_transcript=request.transcript,
            voice_analysis=voice_analysis,
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
    Handle incoming messages via WebSocket with streaming response.

    Expected data format:
    {
        "type": "text" | "voice",
        "content": "message text",
        "transcript": "transcribed text" (for voice, from frontend STT),
        "audioBase64": "base64 encoded audio" (optional, for voice emotion analysis),
        "audioMimeType": "audio/webm" (optional),
        "s3Key": "optional/s3/key" (for voice),
        "sessionId": "session-id"
    }
    
    Response flow (for faster UX):
    1. Send immediate "thinking" event
    2. For MEDIUM/LONG responses: send "waiting" audio BEFORE TTS starts
    3. Send text + audio response
    
    Response length strategy:
    - SHORT (< 50 words): Parallel TTS, NO waiting audio
    - MEDIUM (50-100 words): Send waiting audio BEFORE TTS, then full response
    - LONG (> 100 words): Send waiting audio BEFORE TTS, VoiceVox for reliability
    """
    import base64
    
    try:
        msg_type = data.get("type", "text")
        session_id = data.get("sessionId") or sid

        # Get the text content - for voice, use transcript from frontend STT
        audio_data = None
        audio_mime_type = "audio/webm"
        
        if msg_type == "voice":
            content = data.get("transcript", "")
            s3_key = data.get("s3Key")  # Optional: S3 key for audio backup
            
            # Decode audio for voice emotion analysis if provided
            audio_base64 = data.get("audioBase64")
            if audio_base64:
                try:
                    audio_data = base64.b64decode(audio_base64)
                    audio_mime_type = data.get("audioMimeType", "audio/webm")
                    print(f"[WS VOICE] Received {len(audio_data)} bytes of audio for emotion analysis")
                except Exception as e:
                    print(f"[WS VOICE] Failed to decode audio: {e}")
                    audio_data = None
            
            if not content:
                await sio.emit("error", {"message": "No transcript provided"}, room=sid)
                return
        else:
            content = data.get("content", "")

        if not content.strip():
            await sio.emit("error", {"message": "Empty message"}, room=sid)
            return

        # 1. Send immediate "thinking" event for instant feedback
        await sio.emit(
            "thinking",
            {"status": "processing", "message": content[:50]},
            room=sid,
        )
        print(f"[WS DEBUG] Sent thinking event for {sid}")

        # Callback to send waiting audio BEFORE TTS (for MEDIUM/LONG responses only)
        async def on_waiting_audio(phrase: str, audio_url: str):
            """Send waiting audio immediately when called (before TTS starts)."""
            await sio.emit(
                "waiting",
                {
                    "audioUrl": audio_url,
                    "message": phrase,
                },
                room=sid,
            )
            print(f"[WS DEBUG] Sent waiting audio before TTS: {phrase}")

        # Process message with optional audio for voice emotion analysis
        crew = get_crew()
        print(f"[WS DEBUG] Processing message for {sid}: {content[:50]}...")
        
        result = await crew.process_message(
            user_message=content,
            session_id=session_id,
            audio_data=audio_data,
            audio_mime_type=audio_mime_type,
            on_waiting_audio=on_waiting_audio,
        )
        
        print(f"[WS DEBUG] Got result, sending to {sid}")
        print(f"[WS DEBUG] Response text: {result.get('text', 'NO TEXT')[:100]}")
        print(f"[WS DEBUG] Response length: {result.get('response_length', 'unknown')}")

        # Build voice analysis for response if available
        voice_analysis_data = None
        if result.get("voice_analysis"):
            voice_analysis_data = result["voice_analysis"]
        
        # Check if audio is already included (from Gemini text+audio single call)
        if result.get("audio_complete") and result.get("audio_url"):
            # Audio already generated with text - send complete response
            await sio.emit(
                "response",
                {
                    "text": result["text"],
                    "audioUrl": result["audio_url"],  # Full audio included!
                    "emotion": result.get("emotion", "idle"),
                    "action": result.get("action", "idle"),
                    "contentType": result.get("content_type", "neutral"),
                    "userTranscript": content if msg_type == "voice" else None,
                    "voiceAnalysis": voice_analysis_data,
                    "audioStreaming": False,  # No streaming needed
                    "responseLength": result.get("response_length", "short"),
                },
                room=sid,
            )
            print(f"[WS DEBUG] Complete response with audio sent for {sid}")
        else:
            # 2. Send text response immediately (audio will stream separately)
            await sio.emit(
                "response",
                {
                    "text": result["text"],
                    "audioUrl": None,  # Audio will be streamed
                    "emotion": result.get("emotion", "idle"),
                    "action": result.get("action", "idle"),
                    "contentType": result.get("content_type", "neutral"),
                    "userTranscript": content if msg_type == "voice" else None,
                    "voiceAnalysis": voice_analysis_data,
                    "audioStreaming": True,  # Flag to indicate audio is streaming
                },
                room=sid,
            )
            print(f"[WS DEBUG] Text response sent, starting audio stream for {sid}")
            
            # 3. Stream audio sentence by sentence
            from services.speech import speech_service
            
            response_text = result.get("text", "")
            emotion = result.get("emotion", "neutral")
            
            sentence_count = 0
            async for sentence, audio_url in speech_service.synthesize_sentences(
                text=response_text,
                emotion=emotion,
            ):
                sentence_count += 1
                await sio.emit(
                    "audio_chunk",
                    {
                        "sentence": sentence,
                        "audioUrl": audio_url,
                        "index": sentence_count,
                        "isLast": False,
                    },
                    room=sid,
                )
            
            # Send end of audio stream
            await sio.emit(
                "audio_chunk",
                {"isLast": True, "totalSentences": sentence_count},
                room=sid,
            )
            print(f"[WS DEBUG] Audio stream completed: {sentence_count} sentences for {sid}")

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

