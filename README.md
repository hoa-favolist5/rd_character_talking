# AI Character Voice Interaction System

An interactive AI character that communicates with users via voice in Japanese, powered by CrewAI agents and AWS services.

## Architecture

- **Frontend**: NuxtJS 3 + Vue 3 with animated avatar
- **Backend**: Python FastAPI + CrewAI multi-agent system
- **AI Services**: AWS Bedrock (Claude), Transcribe, Polly
- **Database**: PostgreSQL with pgvector for semantic search

```
┌────────────────────────────────────────────────────────────┐
│                    Browser (NuxtJS)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Voice/Text│  │  Avatar  │  │   Chat   │                  │
│  │  Input   │  │ Display  │  │  Window  │                  │
│  └────┬─────┘  └────▲─────┘  └────▲─────┘                  │
└───────┼─────────────┼─────────────┼────────────────────────┘
        │ WebSocket   │             │
        ▼             │             │
┌───────────────────────────────────────────────────────────┐
│                   FastAPI + Socket.IO                     │
├───────────────────────────────────────────────────────────┤
│                      CrewAI Crew                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│  │  Emotion  │  │ Knowledge │  │   Brain   │              │
│  │   Agent   │  │   Agent   │  │   Agent   │              │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘              │
└────────┼──────────────┼──────────────┼────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Bedrock    │  │ PostgreSQL  │  │   Polly     │
│  (Claude)   │  │ (pgvector)  │  │   (TTS)     │
└─────────────┘  └─────────────┘  └─────────────┘
```

## Documentation

- **[Sequence Diagrams](docs/SEQUENCE_DIAGRAM.md)** - Detailed process flow documentation

## Project Structure

```
character/
├── apps/
│   ├── web/          # NuxtJS Frontend
│   └── api/          # Python FastAPI Backend
├── docs/             # Documents
├── docker-compose.yml
└── Makefile
```

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+
- Docker & Docker Compose
- AWS CLI configured

### Development Setup

1. Start local PostgreSQL:
```bash
docker-compose up -d postgres
```

2. Start the API:
```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn main:app --reload
```

3. Start the frontend:
```bash
cd apps/web
npm install
npm run dev
```

## Environment Variables

### API (.env)
```
DATABASE_URL=postgresql://character:character@localhost:5432/character
AWS_REGION=ap-northeast-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

### Web (.env)
```
NUXT_PUBLIC_API_URL=http://localhost:8000
NUXT_PUBLIC_WS_URL=ws://localhost:8000
```

## License

MIT

