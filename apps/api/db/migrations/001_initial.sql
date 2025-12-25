-- Initial database schema for Character AI
-- Run with: psql -d character -f 001_initial.sql

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Knowledge base table for storing Q&A pairs and information
CREATE TABLE IF NOT EXISTS knowledge_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    category VARCHAR(100) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    keywords TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for category filtering
CREATE INDEX IF NOT EXISTS idx_knowledge_category 
ON knowledge_items (category);

-- Create GIN index for keyword search
CREATE INDEX IF NOT EXISTS idx_knowledge_keywords 
ON knowledge_items USING GIN (keywords);

-- Conversation history table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(100) NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    user_emotion VARCHAR(50),
    response_emotion VARCHAR(50),
    audio_url TEXT,
    context_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for session lookup
CREATE INDEX IF NOT EXISTS idx_conversations_session 
ON conversations (session_id, created_at DESC);

-- Character definitions table
CREATE TABLE IF NOT EXISTS characters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    personality TEXT NOT NULL,
    voice_id VARCHAR(50) DEFAULT 'Takumi',
    avatar_config JSONB DEFAULT '{}',
    system_prompt TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default character
INSERT INTO characters (name, personality, voice_id, system_prompt)
VALUES (
    'Ai',
    'A kind and knowledgeable AI assistant. Committed to polite communication and clear explanations.',
    'Takumi',
    'You are an AI assistant named "Ai".
Please use polite language and provide clear, concise responses.
Keep your answers to 2-3 sentences, keeping in mind that they will be read aloud.'
)
ON CONFLICT (name) DO NOTHING;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for knowledge_items
DROP TRIGGER IF EXISTS update_knowledge_items_updated_at ON knowledge_items;
CREATE TRIGGER update_knowledge_items_updated_at
    BEFORE UPDATE ON knowledge_items
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for characters
DROP TRIGGER IF EXISTS update_characters_updated_at ON characters;
CREATE TRIGGER update_characters_updated_at
    BEFORE UPDATE ON characters
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Sample knowledge items
INSERT INTO knowledge_items (category, question, answer, keywords)
VALUES 
    ('general', 'Who are you?', 'My name is Ai. As an AI assistant, I am here to help you with various tasks.', ARRAY['introduction', 'name', 'AI']),
    ('general', 'What can you do?', 'I can help with answering questions, searching for information, having conversations, and more. Feel free to ask me anything.', ARRAY['features', 'capabilities', 'help']),
    ('greeting', 'Hello', 'Hello! How can I help you today?', ARRAY['greeting', 'hello'])
ON CONFLICT DO NOTHING;
