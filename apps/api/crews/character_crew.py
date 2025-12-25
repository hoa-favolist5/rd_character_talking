"""Character Crew - Multi-agent orchestration for AI character interaction."""

import asyncio
import re
from typing import Any

from crewai import Agent, Crew, Process, Task
from langchain_anthropic import ChatAnthropic

from agents.brain_agent import create_brain_agent
from agents.emotion_agent import create_emotion_agent
from agents.knowledge_agent import create_knowledge_agent
from config.settings import get_settings
from config.voices import ContentType, detect_content_type
from services.llm import get_llm_service
from services.speech import speech_service
from tools.database import SaveConversationTool


# Patterns that indicate a knowledge lookup is needed
KNOWLEDGE_PATTERNS = [
    r"\b(what|who|when|where|why|how|which)\b.*\?",  # Question words
    r"\b(tell me about|explain|describe|show me)\b",  # Request patterns
    r"\b(movie|film|show|series|actor|director)\b",  # Domain-specific
    r"\b(recommend|suggest|find)\b",  # Recommendation patterns
]


class CharacterCrew:
    """
    Main crew for handling character interactions.

    Orchestrates multiple agents to:
    1. Analyze user emotion
    2. Retrieve relevant knowledge
    3. Generate character response
    4. Synthesize speech output
    
    Includes a fast path for simple conversational messages.
    """

    def __init__(
        self,
        character_name: str = "Ai",
        personality: str = "A kind and knowledgeable AI assistant",
        voice_id: str = "Takumi",
        system_prompt: str | None = None,
    ) -> None:
        self.character_name = character_name
        self.personality = personality
        self.voice_id = voice_id
        self.settings = get_settings()

        # Initialize Anthropic Claude LLM for CrewAI agents
        self.llm = ChatAnthropic(
            model=self.settings.anthropic_model,
            api_key=self.settings.anthropic_api_key,
            temperature=0.7,
            max_tokens=500,
        )
        
        # Fast Haiku model for emotion analysis in CrewAI
        self.emotion_llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            api_key=self.settings.anthropic_api_key,
            temperature=0.3,
            max_tokens=100,
        )

        # Create agents
        self.brain_agent = create_brain_agent(
            llm=self.llm,
            character_name=character_name,
            personality=personality,
            system_prompt=system_prompt,
        )
        self.knowledge_agent = create_knowledge_agent(self.llm)
        # Use fast Haiku model for emotion agent
        self.emotion_agent = create_emotion_agent(self.emotion_llm)
        
        # System prompt for fast path
        self._simple_system_prompt = f"""You are an AI assistant named "{character_name}".

[Personality]
{personality}

[Response Guidelines]
1. Use polite and friendly language
2. Keep responses concise - about 2-3 sentences
3. Maintain a warm, conversational tone
4. Responses will be read aloud, so keep them natural
"""

    def _requires_knowledge_lookup(self, message: str) -> bool:
        """Check if message requires database/knowledge lookup."""
        message_lower = message.lower()
        for pattern in KNOWLEDGE_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        return False

    def _is_simple_message(self, message: str) -> bool:
        """Determine if message can use fast path (no agents needed)."""
        # Short greetings and simple responses
        if len(message) < 30:
            simple_patterns = [
                r"^(hi|hello|hey|good morning|good afternoon|good evening|thanks|thank you|bye|goodbye|ok|okay)\b",
                r"^(はい|いいえ|ありがとう|こんにちは|こんばんは|おはよう|さようなら)\b",
            ]
            for pattern in simple_patterns:
                if re.search(pattern, message.lower()):
                    return True
        
        # Don't use fast path if knowledge lookup seems needed
        if self._requires_knowledge_lookup(message):
            return False
            
        # Use fast path for short conversational messages
        return len(message) < 50 and "?" not in message

    def _get_simple_action(self, message: str, content_type: ContentType) -> str:
        """Get character action for simple messages (fast path)."""
        message_lower = message.lower()
        
        # Greeting actions
        greeting_patterns = [
            (["hello", "hi", "hey", "こんにちは", "おはよう"], "wave"),
            (["bye", "goodbye", "さようなら", "またね"], "wave"),
            (["thanks", "thank you", "ありがとう"], "smile"),
            (["おはよう", "good morning"], "smile"),
        ]
        
        for keywords, action in greeting_patterns:
            if any(kw in message_lower for kw in keywords):
                return action
        
        # Content type based actions
        content_action_map = {
            ContentType.COMEDY: "laugh",
            ContentType.HORROR: "scared",
            ContentType.THRILLER: "nervous",
            ContentType.ROMANCE: "blush",
            ContentType.DRAMA: "sympathetic",
            ContentType.CHILDREN: "smile",
            ContentType.ANIMATION: "excited",
            ContentType.ACTION: "excited",
            ContentType.SCIFI: "wonder",
            ContentType.FANTASY: "wonder",
            ContentType.DOCUMENTARY: "explain",
            ContentType.MYSTERY: "curious",
        }
        
        return content_action_map.get(content_type, "smile")

    async def _fast_path_response(
        self,
        user_message: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Fast path for simple conversational messages.
        Uses direct LLM call instead of full agent pipeline.
        Still detects content type for appropriate voice selection.
        """
        llm_service = get_llm_service()
        
        # Generate response directly (single LLM call)
        response_text = await llm_service.generate_response(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=self._simple_system_prompt,
            max_tokens=200,
            temperature=0.7,
        )
        
        # Use neutral emotion for simple messages
        response_emotion = "neutral"
        
        # Detect content type from user message for voice selection
        content_type = detect_content_type(user_message)
        
        # Determine action based on content type (simple mapping for fast path)
        character_action = self._get_simple_action(user_message, content_type)
        
        print(f"[FAST PATH] Content type: {content_type.value}, Action: {character_action}")
        
        # Synthesize speech with content-appropriate voice
        try:
            _, audio_url = await speech_service.synthesize_speech(
                text=response_text,
                voice_id=None,  # Let content_type determine voice
                emotion=response_emotion,
                content_type=content_type,
            )
        except Exception as e:
            print(f"Speech synthesis error: {e}")
            audio_url = None

        # Save conversation asynchronously
        save_tool = SaveConversationTool()
        await asyncio.to_thread(
            save_tool._run,
            session_id=session_id,
            user_message=user_message,
            ai_response=response_text,
            user_emotion=response_emotion,
            response_emotion=response_emotion,
            audio_url=audio_url,
        )

        return {
            "text": response_text,
            "audio_url": audio_url,
            "emotion": response_emotion,
            "action": character_action,
            "content_type": content_type.value,
            "session_id": session_id,
        }

    async def _run_emotion_crew(self, user_message: str) -> str:
        """Run emotion analysis in separate crew (for parallel execution)."""
        emotion_task = Task(
            description=f"""
            Analyze the emotion in the following user message:
            
            "{user_message}"
            
            Report the user's emotional state and recommended response tone.
            """,
            expected_output="""
            - User's emotion: [emotion]
            - Recommended response tone: [tone]
            - Empathy points: [points]
            """,
            agent=self.emotion_agent,
        )
        
        crew = Crew(
            agents=[self.emotion_agent],
            tasks=[emotion_task],
            process=Process.sequential,
            verbose=not self.settings.debug,  # Less verbose in production
        )
        
        result = await asyncio.to_thread(crew.kickoff)
        return str(result)

    async def _run_knowledge_crew(self, user_message: str, session_id: str) -> str:
        """Run knowledge retrieval in separate crew (for parallel execution)."""
        knowledge_task = Task(
            description=f"""
            Search for information needed to answer the following user message:
            
            "{user_message}"
            
            Session ID: {session_id}
            
            1. First, check the conversation history to understand the context
            2. Search the knowledge base for relevant information
            3. Organize and report the search results
            """,
            expected_output="Summary of relevant information",
            agent=self.knowledge_agent,
        )
        
        crew = Crew(
            agents=[self.knowledge_agent],
            tasks=[knowledge_task],
            process=Process.sequential,
            verbose=not self.settings.debug,  # Less verbose in production
        )
        
        result = await asyncio.to_thread(crew.kickoff)
        return str(result)

    async def process_message(
        self,
        user_message: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Process a user message through the crew pipeline.

        Uses fast path for simple messages, full pipeline for complex ones.

        Args:
            user_message: The user's text message
            session_id: Session identifier for conversation tracking

        Returns:
            Dict containing response text, audio URL, and emotion
        """
        # Fast path for simple conversational messages
        if self._is_simple_message(user_message):
            print(f"[FAST PATH] Using direct LLM for simple message: {user_message[:50]}...")
            return await self._fast_path_response(user_message, session_id)
        
        print(f"[FULL PIPELINE] Processing complex message: {user_message[:50]}...")

        # Run emotion and knowledge tasks in TRUE parallel using asyncio.gather
        emotion_result, knowledge_result = await asyncio.gather(
            self._run_emotion_crew(user_message),
            self._run_knowledge_crew(user_message, session_id),
        )
        
        parallel_results = f"[Emotion Analysis]\n{emotion_result}\n\n[Knowledge Search]\n{knowledge_result}"

        # Generate response with Brain Agent
        response_task = Task(
            description=f"""
            Generate a response as {self.character_name} to the user's message.
            
            [User's Message]
            "{user_message}"
            
            [Emotion Analysis Results]
            {parallel_results}
            
            [Response Guidelines]
            1. Maintain {self.character_name}'s persona
            2. Consider the emotion analysis results and respond with an appropriate tone
            3. Utilize the searched information
            4. Keep the response to 2-3 sentences
            5. Keep in mind that the response will be read aloud
            """,
            expected_output="Natural response in 2-3 sentences",
            agent=self.brain_agent,
        )

        response_crew = Crew(
            agents=[self.brain_agent],
            tasks=[response_task],
            process=Process.sequential,
            verbose=not self.settings.debug,  # Less verbose in production
        )

        # Run blocking crew in thread pool to not block event loop
        response_result = await asyncio.to_thread(response_crew.kickoff)
        response_text = str(response_result)
        
        print(f"[DEBUG] Response text: {response_text[:100]}...")

        # Determine emotion for TTS
        response_emotion = self._extract_emotion(emotion_result)
        
        # Determine content type and voice for TTS
        content_type = self._extract_content_type(emotion_result, user_message)
        recommended_voice = self._extract_voice_id(emotion_result)
        
        # Determine character action for frontend animation
        character_action = self._extract_action(emotion_result, response_emotion, content_type)
        
        print(f"[DEBUG] Content type: {content_type.value}, Voice: {recommended_voice}, Action: {character_action}")

        # Synthesize speech with content-appropriate voice
        try:
            _, audio_url = await speech_service.synthesize_speech(
                text=response_text,
                voice_id=recommended_voice,  # Use agent-recommended voice if available
                emotion=response_emotion,
                content_type=content_type,
            )
        except Exception as e:
            print(f"Speech synthesis error: {e}")
            audio_url = None

        # Save conversation asynchronously
        save_tool = SaveConversationTool()
        await asyncio.to_thread(
            save_tool._run,
            session_id=session_id,
            user_message=user_message,
            ai_response=response_text,
            user_emotion=response_emotion,
            response_emotion=response_emotion,
            audio_url=audio_url,
        )

        return {
            "text": response_text,
            "audio_url": audio_url,
            "emotion": response_emotion,
            "action": character_action,
            "content_type": content_type.value,
            "session_id": session_id,
        }

    def _extract_emotion(self, analysis_result: str) -> str:
        """Extract emotion from analysis result."""
        emotion_keywords = {
            "happy": ["happy", "joy", "pleased", "positive"],
            "sad": ["sad", "sorrow", "disappointed", "unhappy"],
            "confused": ["confused", "puzzled", "uncertain", "unclear"],
            "surprised": ["surprised", "shock", "unexpected"],
            "frustrated": ["frustrated", "irritated", "annoyed"],
            "curious": ["curious", "interested", "inquisitive"],
        }

        analysis_lower = analysis_result.lower()

        for emotion, keywords in emotion_keywords.items():
            if any(kw in analysis_lower for kw in keywords):
                return emotion

        return "neutral"

    def _extract_content_type(self, analysis_result: str, user_message: str) -> ContentType:
        """
        Extract content type from emotion analysis result or user message.
        
        First tries to parse from the agent's analysis output,
        then falls back to keyword detection from the user message.
        """
        analysis_lower = analysis_result.lower()
        
        # Content type keywords mapping (from agent output)
        content_type_keywords = {
            ContentType.COMEDY: ["comedy", "funny", "コメディ", "cheerful voice"],
            ContentType.HORROR: ["horror", "scary", "ホラー", "deeper voice", "slower"],
            ContentType.THRILLER: ["thriller", "suspense", "スリラー", "tense"],
            ContentType.ROMANCE: ["romance", "romantic", "love", "ロマンス", "恋愛", "soft", "feminine"],
            ContentType.DRAMA: ["drama", "emotional", "ドラマ", "touching", "expressive"],
            ContentType.CHILDREN: ["children", "kids", "child", "子供", "family", "bright", "higher-pitched"],
            ContentType.ANIMATION: ["animation", "anime", "アニメ", "animated", "energetic"],
            ContentType.ACTION: ["action", "アクション", "battle", "fight", "strong", "confident"],
            ContentType.SCIFI: ["sci-fi", "scifi", "space", "future", "SF", "宇宙", "futuristic"],
            ContentType.FANTASY: ["fantasy", "magic", "ファンタジー", "magical", "whimsical"],
            ContentType.DOCUMENTARY: ["documentary", "ドキュメンタリー", "educational", "professional"],
            ContentType.MYSTERY: ["mystery", "detective", "ミステリー", "mysterious", "intriguing"],
        }
        
        # Try to extract from agent's analysis first
        for content_type, keywords in content_type_keywords.items():
            if any(kw in analysis_lower for kw in keywords):
                return content_type
        
        # Fall back to detecting from user message using the voices module
        return detect_content_type(user_message)

    def _extract_voice_id(self, analysis_result: str) -> str | None:
        """
        Extract recommended voice ID from the agent's analysis.
        
        Returns None if no specific voice was recommended.
        """
        analysis_lower = analysis_result.lower()
        
        # Check for explicit voice recommendations
        if "kazuha" in analysis_lower:
            return "Kazuha"
        elif "takumi" in analysis_lower:
            return "Takumi"
        elif "mizuki" in analysis_lower:
            return "Mizuki"
        elif "tomoko" in analysis_lower:
            return "Tomoko"
        
        return None

    def _extract_action(self, analysis_result: str, emotion: str, content_type: ContentType) -> str:
        """
        Extract character action from the emotion analysis result.
        
        First tries to parse from the agent's analysis output,
        then falls back to mapping based on emotion and content type.
        """
        analysis_lower = analysis_result.lower()
        
        # Direct action keywords mapping (from agent output)
        action_keywords = {
            # Basic expressions
            "smile": ["smile", "smiling", "笑顔"],
            "laugh": ["laugh", "laughing", "funny", "hilarious", "笑"],
            "grin": ["grin", "grinning", "big smile"],
            
            # Sad/Sympathetic
            "sad": ["sad expression", "sad face"],
            "cry": ["cry", "crying", "tearful", "tears", "泣"],
            "sympathetic": ["sympathetic", "empathy", "同情"],
            "comfort": ["comfort", "comforting", "console"],
            
            # Curious/Thinking
            "curious": ["curious", "curiosity", "興味"],
            "thinking": ["thinking", "考え"],
            "confused": ["confused", "confusion", "puzzled"],
            "wonder": ["wonder", "wondering", "amazed at"],
            
            # Surprise/Excitement
            "surprised": ["surprised", "surprise expression"],
            "shocked": ["shocked", "shocking"],
            "excited": ["excited", "excitement", "興奮"],
            "amazed": ["amazed", "amazing"],
            
            # Scared/Nervous
            "scared": ["scared", "fear", "frightened", "怖"],
            "nervous": ["nervous", "anxious"],
            "worried": ["worried", "concern", "心配"],
            
            # Affection/Romance
            "blush": ["blush", "blushing", "embarrassed", "照れ"],
            "love": ["love expression", "loving", "heart"],
            "shy": ["shy", "bashful"],
            "wink": ["wink", "playful wink"],
            
            # Agreement/Gestures
            "nod": ["nod", "nodding", "agree"],
            "shake_head": ["shake head", "disagree"],
            "thumbs_up": ["thumbs up", "approval"],
            
            # Speaking/Listening
            "explain": ["explain", "explaining", "説明"],
            
            # Special
            "wave": ["wave", "waving", "hello", "goodbye", "こんにちは", "さようなら"],
            "bow": ["bow", "bowing", "お辞儀"],
            "celebrate": ["celebrate", "celebration"],
            "cheer": ["cheer", "cheering"],
        }
        
        # Try to find action in analysis
        for action, keywords in action_keywords.items():
            if any(kw in analysis_lower for kw in keywords):
                return action
        
        # Fall back to mapping based on content type
        content_action_map = {
            ContentType.COMEDY: "laugh",
            ContentType.HORROR: "scared",
            ContentType.THRILLER: "nervous",
            ContentType.ROMANCE: "blush",
            ContentType.DRAMA: "sympathetic",
            ContentType.CHILDREN: "smile",
            ContentType.ANIMATION: "excited",
            ContentType.ACTION: "excited",
            ContentType.SCIFI: "wonder",
            ContentType.FANTASY: "wonder",
            ContentType.DOCUMENTARY: "explain",
            ContentType.MYSTERY: "curious",
        }
        
        if content_type in content_action_map:
            return content_action_map[content_type]
        
        # Fall back to mapping based on emotion
        emotion_action_map = {
            "happy": "smile",
            "sad": "sad",
            "confused": "confused",
            "surprised": "surprised",
            "frustrated": "worried",
            "curious": "curious",
            "excited": "excited",
        }
        
        return emotion_action_map.get(emotion, "idle")


# Factory function for creating crew instances
def create_character_crew(
    character_name: str = "Ai",
    personality: str = "A kind and knowledgeable AI assistant",
    voice_id: str = "Takumi",
) -> CharacterCrew:
    """Create a new CharacterCrew instance."""
    return CharacterCrew(
        character_name=character_name,
        personality=personality,
        voice_id=voice_id,
    )
