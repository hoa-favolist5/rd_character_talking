-- Initial database schema for Character AI (MySQL)
-- Run with: mysql -u root -p dev_warehouse < 001_initial.sql

-- Knowledge base table for storing Q&A pairs and information
CREATE TABLE IF NOT EXISTS knowledge_items (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    category VARCHAR(100) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    keywords JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_knowledge_category (category)
);

-- Conversation history table
CREATE TABLE IF NOT EXISTS conversations (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    session_id VARCHAR(100) NOT NULL,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    user_emotion VARCHAR(50),
    response_emotion VARCHAR(50),
    audio_url TEXT,
    context_used JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_conversations_session (session_id, created_at DESC)
);

-- Character definitions table
CREATE TABLE IF NOT EXISTS characters (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    name VARCHAR(100) NOT NULL UNIQUE,
    personality TEXT NOT NULL,
    voice_id VARCHAR(50) DEFAULT 'Takumi',
    avatar_config JSON DEFAULT NULL,
    system_prompt TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Insert default character (use INSERT IGNORE to skip if exists)
INSERT IGNORE INTO characters (id, name, personality, voice_id, system_prompt)
VALUES (
    UUID(),
    'Ai',
    'A kind and knowledgeable AI assistant. Committed to polite communication and clear explanations.',
    'Takumi',
    'You are an AI assistant named "Ai".
Please use polite language and provide clear, concise responses.
Keep your answers to 2-3 sentences, keeping in mind that they will be read aloud.'
);

-- Sample knowledge items
INSERT IGNORE INTO knowledge_items (id, category, question, answer, keywords)
VALUES 
    (UUID(), 'general', 'Who are you?', 'My name is Ai. As an AI assistant, I am here to help you with various tasks.', '["introduction", "name", "AI"]'),
    (UUID(), 'general', 'What can you do?', 'I can help with answering questions, searching for information, having conversations, and more. Feel free to ask me anything.', '["features", "capabilities", "help"]'),
    (UUID(), 'greeting', 'Hello', 'Hello! How can I help you today?', '["greeting", "hello"]');
