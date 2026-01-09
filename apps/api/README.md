# Character AI API

Python FastAPI backend with streaming voice AI for ultra-low latency interaction.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NEW STREAMING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Input ──► Token Streaming (Claude Haiku) ──► Sentence Buffer  │
│                        │                                │            │
│                        │ ~100ms                         │            │
│                        ▼                                ▼            │
│               Live Text Display              ElevenLabs TTS          │
│                                                    │                 │
│                                                    │ ~200-500ms      │
│                                                    ▼                 │
│                        WebRTC (low latency) ◄─── Audio Chunk        │
│                              OR                                      │
│                        WebSocket (fallback)                          │
│                                                                      │
│  Result: ~300-500ms to first audio (vs ~1-2s before)                │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **Token Streaming** - Stream LLM tokens, TTS as sentences complete
- **ElevenLabs TTS** - High-quality, fast (~200-500ms) synthesis
- **WebRTC Audio** - Low-latency native audio transport (optional)
- **WebSocket Fallback** - Base64 audio when WebRTC unavailable
- **CrewAI Multi-Agent** - Brain + Emotion agents for complex queries
- **Fast Path** - Simple messages bypass agents (~300ms total)
- **MySQL + Connection Pooling** - Conversation history persistence

## Tech Stack

- **LLM**: Anthropic Claude (Haiku for speed, Sonnet for quality)
- **TTS**: ElevenLabs (eleven_turbo_v2_5)
- **STT**: AWS Transcribe (frontend)
- **Transport**: WebRTC + Socket.IO
- **Database**: MySQL with aiomysql

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Run migrations
python -m db.migrations.run

# Start server
uvicorn main:app --reload
```

## Environment Variables

```env
# Anthropic (required)
ANTHROPIC_API_KEY=sk-ant-...

# ElevenLabs TTS (required)
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=pNInz6obpgDQGcFmaJgB
ELEVENLABS_MODEL_ID=eleven_turbo_v2_5

# Database
DATABASE_URL=mysql://user:pass@localhost:3306/character

# AWS (for STT)
AWS_REGION=ap-northeast-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

## API Endpoints

### REST

- `GET /health` - Health check
- `POST /chat/text` - Text chat (non-streaming)
- `POST /chat/voice` - Voice chat (non-streaming)
- `GET /credentials/transcribe` - AWS Transcribe credentials

### Socket.IO Events

**Client → Server:**
- `message` - Send text/voice message
- `webrtc_offer` - WebRTC SDP offer
- `webrtc_ice` - ICE candidate
- `webrtc_close` - Close WebRTC session

**Server → Client:**
- `thinking` - Processing started
- `token` - Streaming token (live text)
- `audio_chunk` - Audio chunk (sentence)
- `response` - Complete response
- `webrtc_answer` - WebRTC SDP answer
- `webrtc_ice` - ICE candidate
- `error` - Error message

## Performance

| Metric | Before | After |
|--------|--------|-------|
| First Audio | ~1-2s | ~300-500ms |
| Full Response | ~2-4s | ~1-2s |
| Audio Transport | Base64 WS | WebRTC |

## Services

- `speech_streaming.py` - Token streaming + sentence TTS
- `speech_elevenlabs.py` - ElevenLabs TTS client
- `webrtc_service.py` - WebRTC session management
- `llm.py` - Anthropic Claude client
