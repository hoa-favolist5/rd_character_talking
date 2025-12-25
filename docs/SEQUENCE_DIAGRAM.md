# AI Character System - Process Sequence Documentation

This document describes the sequence of processes in the AI Character Voice Interaction System.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Text Chat Flow](#text-chat-flow)
3. [Voice Chat Flow](#voice-chat-flow)
4. [CrewAI Agent Orchestration](#crewai-agent-orchestration)
5. [Database Operations](#database-operations)
6. [WebSocket Real-time Flow](#websocket-real-time-flow)

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Client [Browser - NuxtJS]
        UI[Chat UI]
        VR[Voice Recorder]
        WS[WebSocket Client]
        Avatar[Character Avatar]
    end

    subgraph Server [FastAPI Server]
        API[REST API]
        SIO[Socket.IO]
        Crew[CrewAI Crew]
    end

    subgraph Agents [CrewAI Agents]
        Brain[Brain Agent]
        Knowledge[Knowledge Agent]
        Emotion[Emotion Agent]
    end

    subgraph AWS [AWS Services]
        Transcribe[Amazon Transcribe]
        Bedrock[Amazon Bedrock Claude]
        Polly[Amazon Polly]
        S3[S3 Audio Storage]
    end

    subgraph Database [PostgreSQL]
        KB[(Knowledge Base)]
        Conv[(Conversations)]
    end

    UI --> WS
    VR --> WS
    WS <--> SIO
    UI --> API
    
    SIO --> Crew
    API --> Crew
    
    Crew --> Brain
    Crew --> Knowledge
    Crew --> Emotion
    
    Brain --> Bedrock
    Knowledge --> KB
    Emotion --> Bedrock
    
    Crew --> Transcribe
    Crew --> Polly
    Polly --> S3
    
    Crew --> Conv
    
    Avatar --> UI
```

---

## Text Chat Flow

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant F as Frontend (Nuxt)
    participant W as WebSocket
    participant A as FastAPI
    participant C as CrewAI
    participant E as Emotion Agent
    participant K as Knowledge Agent
    participant B as Brain Agent
    participant BD as Bedrock (Claude)
    participant P as Polly (TTS)
    participant DB as PostgreSQL
    participant S3 as S3

    U->>F: Type message
    F->>W: Send message via Socket.IO
    W->>A: message event {type: text, content}
    
    A->>C: process_message(text, session_id)
    
    par Parallel Analysis
        C->>E: Analyze user emotion
        E->>BD: Classify emotion
        BD-->>E: emotion: curious
        E-->>C: Emotion analysis result
    and
        C->>K: Search knowledge base
        K->>DB: Query with embeddings
        DB-->>K: Relevant documents
        K-->>C: Knowledge context
    end
    
    C->>B: Generate response
    B->>BD: Generate with context + persona
    BD-->>B: Japanese response
    B-->>C: Response text
    
    C->>P: Synthesize speech
    P-->>C: Audio stream
    C->>S3: Upload audio
    S3-->>C: Audio URL
    
    C->>DB: Save conversation
    
    C-->>A: {text, audioUrl, emotion}
    A-->>W: response event
    W-->>F: Display response
    F->>F: Play audio
    F->>F: Update avatar emotion
    F-->>U: See/hear response
```

### Process Steps

| Step | Component | Action | Details |
|------|-----------|--------|---------|
| 1 | Frontend | User types message | Text input captured |
| 2 | WebSocket | Send to server | Socket.IO `message` event |
| 3 | FastAPI | Receive message | Parse {type, content, sessionId} |
| 4 | CrewAI | Start pipeline | Initialize crew processing |
| 5 | Emotion Agent | Analyze sentiment | Uses Bedrock Claude |
| 6 | Knowledge Agent | Search database | pgvector semantic search |
| 7 | Brain Agent | Generate response | Context-aware, persona-driven |
| 8 | Polly | Text-to-speech | Japanese neural voice |
| 9 | S3 | Store audio | Generate presigned URL |
| 10 | Database | Save conversation | For history/context |
| 11 | Frontend | Display response | Text + audio + avatar |

---

## Voice Chat Flow

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant M as Microphone
    participant F as Frontend (Nuxt)
    participant W as WebSocket
    participant A as FastAPI
    participant T as Transcribe (STT)
    participant S3 as S3
    participant C as CrewAI
    participant P as Polly (TTS)

    U->>M: Speak
    M->>F: Audio stream (WebAudio API)
    F->>F: MediaRecorder captures audio
    U->>F: Stop recording (click mic)
    
    F->>F: Convert to base64
    F->>W: Send {type: audio, content: base64}
    W->>A: message event
    
    A->>S3: Upload audio file
    S3-->>A: S3 URI
    
    A->>T: Start transcription job
    T->>S3: Read audio
    T-->>A: Transcription result
    
    A-->>W: transcript event {text}
    W-->>F: Show user's words
    
    A->>C: process_message(transcript, session_id)
    
    Note over C: Same as text flow...
    
    C-->>A: {text, audioUrl, emotion}
    A-->>W: response event
    W-->>F: Display + play response
```

### Voice Processing Details

```mermaid
flowchart LR
    subgraph Capture [Audio Capture]
        Mic[Microphone] --> WA[Web Audio API]
        WA --> MR[MediaRecorder]
        MR --> Blob[Audio Blob]
    end

    subgraph Transfer [Data Transfer]
        Blob --> B64[Base64 Encode]
        B64 --> WS[WebSocket]
    end

    subgraph STT [Speech-to-Text]
        WS --> S3U[S3 Upload]
        S3U --> TR[Transcribe Job]
        TR --> TXT[Japanese Text]
    end

    subgraph Process [AI Processing]
        TXT --> Crew[CrewAI]
        Crew --> Resp[Response Text]
    end

    subgraph TTS [Text-to-Speech]
        Resp --> Polly[AWS Polly]
        Polly --> MP3[MP3 Audio]
        MP3 --> S3D[S3 Download URL]
    end
```

---

## CrewAI Agent Orchestration

### Agent Hierarchy

```mermaid
flowchart TB
    subgraph Crew [CharacterCrew]
        direction TB
        
        subgraph Parallel [Parallel Tasks]
            E[Emotion Agent]
            K[Knowledge Agent]
        end
        
        B[Brain Agent<br/>Manager]
        
        Parallel --> B
    end

    Input[User Message] --> Parallel
    B --> Output[Response]

    E -.->|Emotion Analysis| B
    K -.->|Context Data| B
```

### Agent Responsibilities

```mermaid
flowchart LR
    subgraph EmotionAgent [Emotion Agent]
        EA1[Receive text]
        EA2[Analyze sentiment]
        EA3[Classify emotion]
        EA4[Recommend tone]
        EA1 --> EA2 --> EA3 --> EA4
    end

    subgraph KnowledgeAgent [Knowledge Agent]
        KA1[Receive query]
        KA2[Get embeddings]
        KA3[Search pgvector]
        KA4[Rank results]
        KA5[Format context]
        KA1 --> KA2 --> KA3 --> KA4 --> KA5
    end

    subgraph BrainAgent [Brain Agent]
        BA1[Receive context]
        BA2[Apply persona]
        BA3[Generate response]
        BA4[Ensure Japanese]
        BA5[Keep concise]
        BA1 --> BA2 --> BA3 --> BA4 --> BA5
    end
```

### Task Flow

```mermaid
stateDiagram-v2
    [*] --> ReceiveMessage
    
    ReceiveMessage --> ParallelAnalysis
    
    state ParallelAnalysis {
        [*] --> EmotionTask
        [*] --> KnowledgeTask
        EmotionTask --> [*]
        KnowledgeTask --> [*]
    }
    
    ParallelAnalysis --> ResponseGeneration
    ResponseGeneration --> SpeechSynthesis
    SpeechSynthesis --> SaveConversation
    SaveConversation --> ReturnResponse
    ReturnResponse --> [*]
```

---

## Database Operations

### Schema Relationships

```mermaid
erDiagram
    KNOWLEDGE_ITEMS {
        uuid id PK
        varchar category
        text question
        text answer
        text[] keywords
        vector embedding
        timestamp created_at
        timestamp updated_at
    }

    CONVERSATIONS {
        uuid id PK
        varchar session_id
        text user_message
        text ai_response
        varchar user_emotion
        varchar response_emotion
        text audio_url
        timestamp created_at
    }

    CHARACTERS {
        uuid id PK
        varchar name UK
        text personality
        varchar voice_id
        jsonb avatar_config
        text system_prompt
        boolean is_active
    }

    CONVERSATIONS ||--o{ KNOWLEDGE_ITEMS : "may reference"
    CHARACTERS ||--o{ CONVERSATIONS : "responds in"
```

### Query Flow

```mermaid
sequenceDiagram
    participant K as Knowledge Agent
    participant B as Bedrock Embeddings
    participant DB as PostgreSQL
    participant PG as pgvector

    K->>B: Get embedding for query
    B-->>K: vector[1536]
    
    K->>DB: SELECT with vector similarity
    DB->>PG: Calculate cosine distance
    PG-->>DB: Ranked results
    DB-->>K: Top 5 matches
    
    K->>K: Format as context
```

---

## WebSocket Real-time Flow

### Connection Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Socket.IO Server
    participant H as Event Handlers

    C->>S: connect()
    S-->>C: connect (sid assigned)
    
    loop Chat Session
        C->>S: message {type, content}
        S->>H: Process message
        H-->>S: transcript (if voice)
        S-->>C: transcript event
        H-->>S: Processing complete
        S-->>C: response event
    end
    
    C->>S: disconnect
    S-->>C: disconnect confirmed
```

### Event Types

```mermaid
flowchart LR
    subgraph ClientToServer [Client → Server]
        MSG[message<br/>{type, content, sessionId}]
    end

    subgraph ServerToClient [Server → Client]
        TR[transcript<br/>{text}]
        RS[response<br/>{text, audioUrl, emotion}]
        ER[error<br/>{message}]
    end

    MSG --> Processing[Server Processing]
    Processing --> TR
    Processing --> RS
    Processing --> ER
```

---

## Complete Request-Response Cycle

```mermaid
flowchart TB
    subgraph Input [User Input]
        TI[Text Input]
        VI[Voice Input]
    end

    subgraph Frontend [Frontend Processing]
        WS[WebSocket Send]
        B64[Base64 Encode Audio]
    end

    subgraph API [API Layer]
        RCV[Receive Message]
        STT[Speech-to-Text]
        CREW[CrewAI Process]
        TTS[Text-to-Speech]
        SAVE[Save to DB]
    end

    subgraph Response [Response]
        JSON[JSON Response]
        AUDIO[Audio URL]
        EMO[Emotion State]
    end

    subgraph UI [UI Update]
        MSG[Display Message]
        PLAY[Play Audio]
        AVT[Update Avatar]
    end

    TI --> WS
    VI --> B64 --> WS
    
    WS --> RCV
    RCV --> STT
    STT --> CREW
    CREW --> TTS
    CREW --> SAVE
    TTS --> AUDIO
    
    CREW --> JSON
    CREW --> EMO
    
    JSON --> MSG
    AUDIO --> PLAY
    EMO --> AVT
```

---

## Error Handling Flow

```mermaid
flowchart TB
    subgraph Errors [Possible Errors]
        E1[Audio capture failed]
        E2[Transcription failed]
        E3[LLM timeout]
        E4[Database error]
        E5[TTS failed]
    end

    subgraph Handling [Error Handling]
        H1[Retry with fallback]
        H2[Return text only]
        H3[Use cached response]
        H4[Graceful degradation]
    end

    subgraph UserFeedback [User Feedback]
        F1[Error message]
        F2[Retry prompt]
        F3[Status indicator]
    end

    E1 --> H1
    E2 --> H2
    E3 --> H3
    E4 --> H4
    E5 --> H2

    H1 --> F2
    H2 --> F1
    H3 --> F1
    H4 --> F3
```

---

## Performance Considerations

| Stage | Expected Latency | Optimization |
|-------|-----------------|--------------|
| WebSocket connect | ~50ms | Keep-alive |
| Audio upload | ~200ms | Compress before send |
| Transcription | ~1-3s | Streaming API |
| LLM Response | ~2-5s | Streaming response |
| TTS Generation | ~500ms | Cache common phrases |
| Total (voice) | ~4-10s | Parallel processing |
| Total (text) | ~3-7s | Skip STT step |

---

## Summary

The AI Character system processes user input through a multi-stage pipeline:

1. **Input Capture**: Text or voice from the browser
2. **Transport**: WebSocket for real-time communication
3. **STT (if voice)**: AWS Transcribe for Japanese
4. **AI Processing**: CrewAI orchestrates 3 specialized agents
5. **Response Generation**: Context-aware, persona-consistent
6. **TTS**: AWS Polly neural voice synthesis
7. **Delivery**: Real-time response with audio playback
8. **UI Update**: Avatar emotion sync, message display

All processing is designed for Japanese language support with AWS services optimized for the `ap-northeast-1` region.

