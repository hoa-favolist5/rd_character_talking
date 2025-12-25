# Character AI API

Python FastAPI backend with CrewAI multi-agent orchestration for AI character voice interaction.

## Features

- FastAPI with async support
- CrewAI multi-agent system (Brain, Knowledge, Emotion agents)
- AWS Bedrock (Claude 3) integration
- AWS Transcribe (Japanese STT)
- AWS Polly (Japanese TTS)
- PostgreSQL with pgvector for semantic search
- Socket.IO for real-time communication

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

- `DATABASE_URL` - PostgreSQL connection string
- `AWS_REGION` - AWS region (default: ap-northeast-1)
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `BEDROCK_MODEL_ID` - Bedrock model ID
- `S3_BUCKET_AUDIO` - S3 bucket for audio files

## API Endpoints

- `GET /health` - Health check
- `POST /chat/text` - Text chat
- `POST /chat/voice` - Voice chat
- `WS /socket.io/` - Real-time WebSocket

