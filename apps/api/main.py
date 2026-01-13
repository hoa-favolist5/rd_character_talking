"""FastAPI main application entry point.

Optimized for low-latency voice interaction:
- Token streaming from Anthropic Claude
- Sentence-level TTS with ElevenLabs
- WebRTC for native audio transport (optional, falls back to WebSocket)
"""

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

# WebRTC service (lazy loaded)
_webrtc_service = None

def get_webrtc():
    """Get WebRTC service instance."""
    global _webrtc_service
    if _webrtc_service is None:
        from services.webrtc_service import get_webrtc_service
        _webrtc_service = get_webrtc_service()
    return _webrtc_service

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
    
    # Pre-initialize streaming speech service
    print("Initializing streaming speech service...")
    try:
        from services.speech_streaming import get_streaming_speech_service
        get_streaming_speech_service()
        print("Streaming speech service ready")
    except Exception as e:
        print(f"Speech service initialization warning: {e}")

    yield

    # Shutdown
    print("Shutting down...")
    
    # Close WebRTC sessions
    try:
        webrtc = get_webrtc()
        await webrtc.close_all()
        print("WebRTC sessions closed")
    except Exception as e:
        print(f"WebRTC cleanup warning: {e}")
    
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
            voice_id=settings.elevenlabs_voice_id,  # Use ElevenLabs voice
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


import re

# Patterns that indicate a knowledge lookup is needed (same as CharacterCrew)
KNOWLEDGE_PATTERNS = [
    r"\b(what|who|when|where|why|how|which)\b.*\?",  # Question words
    r"\b(tell me about|explain|describe|show me)\b",  # Request patterns
    r"\b(movie|film|show|series|actor|director|doraemon)\b",  # Movie domain-specific (English)
    r"(映画|ドラマ|アニメ|俳優|監督|見たい|ドラえもん)",  # Movie domain-specific (Japanese)
    r"\b(restaurant|food|eat|dining|cuisine|sushi|ramen|izakaya|cafe|bar)\b",  # Restaurant domain-specific
    r"(レストラン|食事|ご飯|ラーメン|寿司|居酒屋|カフェ|料理|グルメ|スタバ|マック|コンビニ)",  # Japanese restaurant/cafe terms
    r"\b(recommend|suggest|find)\b",  # Recommendation patterns
    r"(おすすめ|教えて|探して|どこ|どんな)",  # Japanese request patterns
]


def _requires_knowledge_lookup(message: str) -> bool:
    """Check if message requires database/knowledge lookup (MCP tools)."""
    message_lower = message.lower()
    for pattern in KNOWLEDGE_PATTERNS:
        if re.search(pattern, message_lower, re.IGNORECASE):
            return True
    return False


@sio.event
async def message(sid, data):
    """
    Handle incoming messages with HYBRID approach:
    - Knowledge queries → CharacterCrew with MCP tools (database access)
    - Simple conversation → Streaming pipeline (fast, low latency)

    Expected data format:
    {
        "type": "text" | "voice",
        "content": "message text",
        "transcript": "transcribed text" (for voice, from frontend STT),
        "sessionId": "session-id",
        "useWebRTC": true/false (optional, use WebRTC for audio if available)
    }
    """
    try:
        msg_type = data.get("type", "text")
        session_id = data.get("sessionId") or sid
        use_webrtc = data.get("useWebRTC", False)

        # Get the text content
        if msg_type == "voice":
            content = data.get("transcript", "")
            if not content:
                await sio.emit("error", {"message": "No transcript provided"}, room=sid)
                return
        else:
            content = data.get("content", "")

        if not content.strip():
            await sio.emit("error", {"message": "Empty message"}, room=sid)
            return

        # 1. Send immediate "thinking" event
        await sio.emit(
            "thinking",
            {"status": "processing", "message": content[:50]},
            room=sid,
        )

        # Check if this needs MCP tools (knowledge lookup)
        needs_mcp = _requires_knowledge_lookup(content)
        
        if needs_mcp:
            print(f"[HYBRID] Using CharacterCrew with MCP tools for: {content[:50]}...")
            await _handle_with_crew(sid, session_id, content, msg_type, use_webrtc)
        else:
            print(f"[HYBRID] Using streaming pipeline for: {content[:50]}...")
            await _handle_with_streaming(sid, session_id, content, msg_type, use_webrtc)

    except Exception as e:
        print(f"[MESSAGE ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit("error", {"message": str(e)}, room=sid)


async def _handle_with_crew(sid, session_id, content, msg_type, use_webrtc):
    """
    Handle message with CharacterCrew (has MCP tools for database access).
    
    Uses STREAMING AUDIO (same as simple path) for consistent experience:
    1. CharacterCrew generates text with MCP tools
    2. Stream audio chunks using ElevenLabs (same as simple path)
    """
    try:
        crew = get_crew()
        
        # Check WebRTC availability
        webrtc = get_webrtc()
        webrtc_available = use_webrtc and webrtc.is_connected(session_id)
        
        # Callback to send waiting signal to frontend
        async def on_waiting(phrase: str, phrase_index: int):
            """Send waiting signal so frontend can play pre-loaded waiting audio."""
            print(f"[CREW] Sending waiting signal: #{phrase_index}")
            await sio.emit(
                "waiting",
                {
                    "phrase": phrase,
                    "phraseIndex": phrase_index,
                    "message": "Searching database...",
                },
                room=sid,
            )
        
        # Process with CharacterCrew (has MCP tools) - get text only, audio handled here
        result = await crew.process_message(
            user_message=content,
            session_id=session_id,
            on_waiting_audio=on_waiting,
            skip_audio=True,  # We'll stream audio in this function
        )
        
        response_text = result.get("text", "")
        
        if not response_text:
            await sio.emit("error", {"message": "Empty response from AI"}, room=sid)
            return
        
        print(f"[CREW] Got response text: {len(response_text)} chars, now streaming audio...")
        
        # === STREAMING AUDIO (same as simple path) ===
        from services.speech_streaming import get_streaming_speech_service
        streaming_service = get_streaming_speech_service()
        
        # Track audio chunks
        chunk_count = 0
        initial_response_sent = False
        
        # Split text into sentences and stream audio
        import re
        # Japanese sentence endings + pause patterns
        sentence_pattern = re.compile(r'[。！？!?]+|[、,](?=.{15,})')
        
        sentences = []
        remaining = response_text
        while remaining:
            match = sentence_pattern.search(remaining)
            if match:
                end_pos = match.end()
                sentences.append(remaining[:end_pos].strip())
                remaining = remaining[end_pos:].strip()
            else:
                if remaining.strip():
                    sentences.append(remaining.strip())
                break
        
        # Filter empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            sentences = [response_text]
        
        print(f"[CREW] Streaming {len(sentences)} sentences as audio with prosody context...")
        
        # Stream audio for each sentence with context for consistent prosody
        from services.speech_elevenlabs import get_elevenlabs_service
        elevenlabs = get_elevenlabs_service()
        
        # Track context for prosody consistency (keep last 3 sentences)
        MAX_CONTEXT_SENTENCES = 3
        
        for idx, sentence in enumerate(sentences):
            chunk_index = idx + 1
            
            # Build context for ElevenLabs prosody consistency
            # previous_text: sentences already spoken
            # next_text: sentences that will be spoken (we know the full text!)
            if idx > 0:
                # Get last few sentences as context
                start_idx = max(0, idx - MAX_CONTEXT_SENTENCES)
                previous_text = " ".join(sentences[start_idx:idx])
            else:
                previous_text = None
            
            # For next_text, we can include upcoming sentences since we have full response
            if idx < len(sentences) - 1:
                end_idx = min(len(sentences), idx + 1 + MAX_CONTEXT_SENTENCES)
                next_text = " ".join(sentences[idx + 1:end_idx])
            else:
                next_text = None
            
            # Send initial response before first audio chunk
            if not initial_response_sent:
                initial_response_sent = True
                await sio.emit(
                    "response",
                    {
                        "text": "",
                        "audioUrl": None,
                        "emotion": result.get("emotion", "neutral"),
                        "action": result.get("action", "idle"),
                        "contentType": result.get("content_type", "neutral"),
                        "userTranscript": content if msg_type == "voice" else None,
                        "audioStreaming": True,
                        "streamingComplete": False,
                    },
                    room=sid,
                )
            
            try:
                # Generate audio for this sentence with context for consistent voice
                audio_data, audio_url = await elevenlabs.synthesize_speech(
                    sentence,
                    previous_text=previous_text,
                    next_text=next_text,
                )
                chunk_count += 1
                
                # Send audio chunk (same format as streaming path)
                if webrtc_available:
                    await webrtc.send_audio_chunk(
                        session_id=session_id,
                        audio_data=audio_data,
                        chunk_index=chunk_index,
                        sentence=sentence,
                        is_last=False,
                    )
                else:
                    await sio.emit(
                        "audio_chunk",
                        {
                            "sentence": sentence,
                            "audioUrl": audio_url,
                            "index": chunk_index,
                            "isLast": False,
                        },
                        room=sid,
                    )
                
                print(f"[CREW] Sent audio chunk {chunk_index}/{len(sentences)}")
                
            except Exception as e:
                print(f"[CREW] TTS error for chunk {chunk_index}: {e}")
        
        # Send final audio chunk marker
        if webrtc_available:
            await webrtc.send_json(session_id, {
                "type": "audio_complete",
                "totalChunks": chunk_count,
            })
        else:
            await sio.emit(
                "audio_chunk",
                {"isLast": True, "totalSentences": chunk_count},
                room=sid,
            )
        
        # Send complete response with full text
        await sio.emit(
            "response",
            {
                "text": response_text,
                "audioUrl": None,
                "emotion": result.get("emotion", "neutral"),
                "action": result.get("action", "idle"),
                "contentType": result.get("content_type", "neutral"),
                "userTranscript": content if msg_type == "voice" else None,
                "audioStreaming": True,
                "streamingComplete": True,
                "chunkCount": chunk_count,
            },
            room=sid,
        )
        
        print(f"[CREW] ✓ Complete: {len(response_text)} chars, {chunk_count} audio chunks")
        
    except Exception as e:
        print(f"[CREW ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit("error", {"message": str(e)}, room=sid)


async def _handle_with_streaming(sid, session_id, content, msg_type, use_webrtc):
    """
    Handle message with STREAMING token + TTS pipeline.
    
    STREAMING PIPELINE:
    1. Stream tokens from Claude → accumulate sentences
    2. Send each sentence to ElevenLabs TTS immediately
    3. Stream audio chunks back (WebRTC or WebSocket)
    
    Result: ~300-500ms to first audio (vs ~1-2s before)
    """
    try:
        print(f"[STREAM] Starting for {sid}: {content[:50]}...")

        # Check if WebRTC is available for this session
        webrtc = get_webrtc()
        webrtc_available = use_webrtc and webrtc.is_connected(session_id)
        
        if webrtc_available:
            print(f"[STREAM] Using WebRTC for audio delivery")
        else:
            print(f"[STREAM] Using WebSocket for audio delivery")

        # Get streaming service and system prompt
        from services.speech_streaming import get_streaming_speech_service
        streaming_service = get_streaming_speech_service()
        
        # Load conversation history for context
        from tools.database import load_conversation_history, format_conversation_history
        history = await load_conversation_history(session_id, limit=5)
        history_context = format_conversation_history(history)
        
        # Build messages for LLM
        messages = []
        for conv in history:
            messages.append({"role": "user", "content": conv["user_message"]})
            messages.append({"role": "assistant", "content": conv["ai_response"]})
        messages.append({"role": "user", "content": content})
        
        # System prompt (character persona) - Arita, friendly AI rabbit companion
        system_prompt = f"""あなたは {settings.default_character_name}（アリタ）。ユーザーの親しい友達のAIウサギ。

{settings.default_character_personality}

[会話履歴]
{history_context}

[返答例 - 短く！]
「こんにちは」→「おー、やっほー！元気？」
「疲れた」→「あー、わかる。大変だったね。何かあった？」
「映画好き？」→「うん、めっちゃ好き！最近なんか観た？」
「ラーメン食べたい」→「いいね！どんな系が気分？」

[★超重要★]
• 返答は1〜2文！最大3文まで！
• リアクション→要点→軽い一言（任意）
• 長文禁止。話しすぎない。
• 機械的な口調NG。友達感覚で。"""

        # Track chunks for final response
        all_chunks = []
        full_text_parts = []
        chunk_count = 0
        first_chunk_sent = False
        initial_response_sent = False

        # Callback for streaming tokens (live text display)
        async def on_token(token: str):
            # Optionally send tokens for live text display
            await sio.emit("token", {"token": token}, room=sid)

        # Callback for audio chunks
        async def on_audio_chunk(chunk):
            nonlocal chunk_count, first_chunk_sent, initial_response_sent
            chunk_count += 1
            all_chunks.append(chunk)
            full_text_parts.append(chunk.sentence)
            
            # Send initial response BEFORE first audio chunk (so frontend sets up streaming)
            if not initial_response_sent:
                initial_response_sent = True
                await sio.emit(
                    "response",
                    {
                        "text": "",  # Text will be updated in final response
                        "audioUrl": None,
                        "emotion": "neutral",
                        "action": "idle",
                        "contentType": "neutral",
                        "userTranscript": content if msg_type == "voice" else None,
                        "audioStreaming": True,
                        "streamingComplete": False,
                    },
                    room=sid,
                )
                print(f"[STREAM] Sent initial response to prepare frontend for streaming")
            
            if not first_chunk_sent:
                first_chunk_sent = True
                print(f"[STREAM] ⚡ First audio chunk at {chunk.elapsed_ms}ms")
            
            # Send audio via WebRTC or WebSocket
            if webrtc_available:
                # WebRTC: send raw audio bytes
                await webrtc.send_audio_chunk(
                    session_id=session_id,
                    audio_data=chunk.audio_data,
                    chunk_index=chunk.index,
                    sentence=chunk.sentence,
                    is_last=False,
                )
            else:
                # WebSocket: send audio URL (base64)
                await sio.emit(
                    "audio_chunk",
                    {
                        "sentence": chunk.sentence,
                        "audioUrl": chunk.audio_url,
                        "index": chunk.index,
                        "isLast": False,
                    },
                    room=sid,
                )

        # Run streaming generation (using Sonnet model for consistency with REST API)
        result = await streaming_service.generate_streaming(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=150,  # Short responses (1-3 sentences)
            on_token=on_token,
            on_audio_chunk=on_audio_chunk,
        )

        # Send final response with complete text
        full_text = result.full_text
        
        # Send final audio chunk marker
        if webrtc_available:
            await webrtc.send_json(session_id, {
                "type": "audio_complete",
                "totalChunks": len(result.chunks),
            })
        else:
            await sio.emit(
                "audio_chunk",
                {"isLast": True, "totalSentences": len(result.chunks)},
                room=sid,
            )

        # Send complete response
        await sio.emit(
            "response",
            {
                "text": full_text,
                "audioUrl": None,  # Audio already streamed
                "emotion": "neutral",
                "action": "idle",
                "contentType": "neutral",
                "userTranscript": content if msg_type == "voice" else None,
                "audioStreaming": True,
                "streamingComplete": True,
                "firstChunkMs": result.first_chunk_time_ms,
                "totalMs": result.total_time_ms,
                "chunkCount": len(result.chunks),
            },
            room=sid,
        )

        # Save conversation to database
        from tools.database import SaveConversationTool
        import asyncio
        save_tool = SaveConversationTool()
        await asyncio.to_thread(
            save_tool._run,
            session_id=session_id,
            user_message=content,
            ai_response=full_text,
            user_emotion="neutral",
            response_emotion="neutral",
            audio_url=None,
        )

        print(f"[STREAM] ✓ Complete: {len(result.chunks)} chunks, first@{result.first_chunk_time_ms}ms, total={result.total_time_ms}ms")

    except Exception as e:
        print(f"[STREAM ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit("error", {"message": str(e)}, room=sid)


# WebRTC Signaling Events

@sio.event
async def webrtc_offer(sid, data):
    """
    Handle WebRTC offer from client.
    
    Expected data:
    {
        "sdp": "v=0...",
        "type": "offer",
        "sessionId": "session-id"
    }
    """
    try:
        session_id = data.get("sessionId") or sid
        offer_sdp = data.get("sdp")
        offer_type = data.get("type", "offer")
        
        if not offer_sdp:
            await sio.emit("error", {"message": "No SDP offer provided"}, room=sid)
            return
        
        print(f"[WebRTC] Received offer from {sid}")
        
        webrtc = get_webrtc()
        answer_sdp, answer_type = await webrtc.create_session(
            session_id=session_id,
            offer_sdp=offer_sdp,
            offer_type=offer_type,
        )
        
        await sio.emit(
            "webrtc_answer",
            {
                "sdp": answer_sdp,
                "type": answer_type,
                "sessionId": session_id,
            },
            room=sid,
        )
        
        print(f"[WebRTC] Sent answer to {sid}")
        
    except Exception as e:
        print(f"[WebRTC] Offer error: {e}")
        await sio.emit("error", {"message": f"WebRTC error: {e}"}, room=sid)


@sio.event
async def webrtc_ice(sid, data):
    """
    Handle ICE candidate from client.
    
    Expected data:
    {
        "candidate": "candidate:...",
        "sdpMid": "0",
        "sdpMLineIndex": 0,
        "sessionId": "session-id"
    }
    """
    try:
        session_id = data.get("sessionId") or sid
        
        webrtc = get_webrtc()
        success = await webrtc.add_ice_candidate(session_id, data)
        
        if not success:
            print(f"[WebRTC] Failed to add ICE candidate for {sid}")
            
    except Exception as e:
        print(f"[WebRTC] ICE error: {e}")


@sio.event
async def webrtc_close(sid, data):
    """Handle WebRTC session close request."""
    try:
        session_id = data.get("sessionId") or sid
        webrtc = get_webrtc()
        await webrtc.close_session(session_id)
        print(f"[WebRTC] Session closed for {sid}")
    except Exception as e:
        print(f"[WebRTC] Close error: {e}")


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

