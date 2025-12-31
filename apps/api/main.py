"""FastAPI main application entry point - Nova 2 Sonic Demo."""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import socketio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from db.connection import db
from models.schemas import (
    CharacterAction,
    ChatResponse,
    ContentTypeEnum,
    EmotionType,
    HealthCheck,
    TextMessageRequest,
)
from services.nova_sonic import get_nova_sonic_service

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
    
    print("Starting up... (Nova 2 Sonic Demo)")
    await db.connect()
    print("Database connected")
    
    # Warm up connection pool
    try:
        await asyncio.gather(*[db.fetchval("SELECT 1") for _ in range(5)])
        print("Connection pool warmed up")
    except Exception as e:
        print(f"Pool warmup warning: {e}")

    yield

    print("Shutting down...")
    await db.disconnect()
    print("Database disconnected")


# FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI Character Voice Interaction API - Nova 2 Sonic",
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
        llm="nova_sonic",
    )


@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(request: TextMessageRequest) -> ChatResponse:
    """
    Handle text-based chat messages using Nova 2 Sonic.
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        nova = get_nova_sonic_service()
        
        # Get text response and audio
        response_text = ""
        audio_chunks = []
        
        async for event in nova.process_text_to_speech(
            text=request.content,
            system_prompt=settings.default_character_personality,
        ):
            if event["type"] == "response_text":
                response_text = event["text"]
            elif event["type"] == "audio":
                audio_chunks.append(event["data"])
        
        # Combine audio chunks
        audio_url = None
        if audio_chunks:
            import base64
            combined = base64.b64decode("".join(audio_chunks))
            audio_url = f"data:audio/pcm;base64,{base64.b64encode(combined).decode()}"

        return ChatResponse(
            text=response_text,
            audio_url=audio_url,
            emotion=EmotionType.IDLE,
            action=CharacterAction.IDLE,
            content_type=ContentTypeEnum.NEUTRAL,
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Socket.IO Event Handlers


@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    print(f"[NOVA SONIC] Client connected: {sid}")


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    print(f"[NOVA SONIC] Client disconnected: {sid}")


@sio.event
async def message(sid, data):
    """
    Handle incoming messages via WebSocket with Nova 2 Sonic streaming.

    Expected data format:
    {
        "type": "text" | "voice",
        "content": "message text",
        "audioBase64": "base64 encoded audio" (for voice),
        "sessionId": "session-id"
    }
    """
    try:
        msg_type = data.get("type", "text")
        session_id = data.get("sessionId") or sid
        content = data.get("content", "") or data.get("transcript", "")

        if not content.strip():
            await sio.emit("error", {"message": "Empty message"}, room=sid)
            return

        # Send thinking event
        await sio.emit(
            "thinking",
            {"status": "processing", "message": content[:50]},
            room=sid,
        )
        print(f"[NOVA SONIC] Processing: {content[:50]}...")

        nova = get_nova_sonic_service()
        
        # Stream TTS sentence by sentence
        response_text = ""
        sentence_count = 0
        
        async for sentence, audio_url in nova.synthesize_sentences(
            text=content,
            system_prompt=settings.default_character_personality,
        ):
            sentence_count += 1
            response_text += sentence
            
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
        
        # Send text response
        await sio.emit(
            "response",
            {
                "text": response_text,
                "emotion": "idle",
                "action": "idle",
                "contentType": "neutral",
            },
            room=sid,
        )
        
        # End of stream
        await sio.emit(
            "audio_chunk",
            {"isLast": True, "totalSentences": sentence_count},
            room=sid,
        )
        print(f"[NOVA SONIC] Completed: {sentence_count} sentences")

    except Exception as e:
        print(f"[NOVA SONIC ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit("error", {"message": str(e)}, room=sid)


# Export the ASGI app
app = socket_app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
