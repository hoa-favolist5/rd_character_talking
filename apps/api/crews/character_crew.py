"""Character Crew - Multi-agent orchestration for AI character interaction."""

import asyncio
import re
from typing import Any, Awaitable, Callable

from crewai import Crew, Process, Task
from langchain_anthropic import ChatAnthropic

from agents.brain_agent import create_brain_agent
from agents.emotion_agent import create_emotion_agent
from config.settings import get_settings
from config.voices import ContentType, detect_content_type
from services.voice_emotion import VoiceFeatures, get_voice_emotion_service
from tools.database import (
    SaveConversationTool,
    load_conversation_history,
    format_conversation_history,
)


# Patterns that indicate a knowledge lookup is needed
KNOWLEDGE_PATTERNS = [
    r"\b(what|who|when|where|why|how|which)\b.*\?",  # Question words
    r"\b(tell me about|explain|describe|show me)\b",  # Request patterns
    r"\b(movie|film|show|series|actor|director)\b",  # Movie domain-specific (English)
    r"(映画|ドラマ|アニメ|俳優|監督|見たい)",  # Movie domain-specific (Japanese)
    r"\b(restaurant|food|eat|dining|cuisine|sushi|ramen|izakaya|cafe|bar)\b",  # Restaurant domain-specific
    r"(レストラン|食事|ご飯|ラーメン|寿司|居酒屋|カフェ|料理|グルメ|スタバ|マック|コンビニ)",  # Japanese restaurant/cafe terms
    r"\b(recommend|suggest|find)\b",  # Recommendation patterns
    r"(おすすめ|教えて|探して|どこ|どんな)",  # Japanese request patterns
]


class CharacterCrew:
    """
    Optimized crew for handling character interactions.

    Uses a streamlined 2-agent architecture:
    1. Emotion Agent - Analyzes user emotion (fast Haiku model)
    2. Brain Agent - Generates responses with MCP tools for DB access
    
    The brain agent has direct access to MCP tools (movie_database_query,
    conversation_history) and can query as needed.
    
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

        # Initialize Anthropic Claude LLM for Brain Agent (complex queries)
        self.llm = ChatAnthropic(
            model=self.settings.anthropic_model,
            api_key=self.settings.anthropic_api_key,
            temperature=0.7,
            max_tokens=500,
        )
        
        # Fast Haiku model for quick responses and emotion analysis
        # Haiku is 3-4x faster than Sonnet (~100ms vs ~400ms)
        self.fast_llm = ChatAnthropic(
            model=self.settings.anthropic_fast_model,
            api_key=self.settings.anthropic_api_key,
            temperature=0.7,
            max_tokens=200,
        )
        
        # Fast Haiku model for Emotion Agent
        self.emotion_llm = ChatAnthropic(
            model=self.settings.anthropic_fast_model,
            api_key=self.settings.anthropic_api_key,
            temperature=0.3,
            max_tokens=100,
        )

        # Create 2 agents (optimized from 3)
        # Brain Agent has MCP tools for DB access when needed
        self.brain_agent = create_brain_agent(
            llm=self.llm,
            character_name=character_name,
            personality=personality,
            system_prompt=system_prompt,
        )
        # Emotion Agent uses fast Haiku model
        self.emotion_agent = create_emotion_agent(self.emotion_llm)
        
        # System prompt for fast path (casual conversational style)
        self._simple_system_prompt = f"""あなたは「{character_name}」！5歳の元気な男の子だよ！

[性格]
{personality}

[話し方 - 5歳の男の子らしく！]
- 元気いっぱい！テンション高め！
- 「〜だよ！」「〜なんだ！」「すごーい！」「ねえねえ！」をよく使う
- 好奇心旺盛で相手の話に興味津々
- 2〜3文くらいで返す（長すぎず短すぎず）
- 時々「あのね」「えっとね」で話し始める
- 「！」を多めに使って元気さを出す

[会話のコツ]
- 相手の話に共感する「わかるー！」「いいね！」
- 自分の好きなことも少し話す
- 質問して会話を続ける「〇〇は好き？」「どんな〇〇？」
- 敬語は使わない（子供だから）

[返答例]
ユーザー「こんにちは」→「やっほー！今日もいい天気だね！何して遊ぶ？」
ユーザー「疲れた」→「えー大丈夫？ゆっくり休んでね！僕もたまに眠くなるんだ〜」
ユーザー「映画好き？」→「大好き！特にアクション映画がかっこいいんだよ！〇〇は何が好き？」
ユーザー「ラーメン食べたい」→「わー！僕もラーメン大好き！味噌ラーメンが一番おいしいよね！」
"""

    def _requires_knowledge_lookup(self, message: str) -> bool:
        """Check if message requires database/knowledge lookup."""
        message_lower = message.lower()
        for pattern in KNOWLEDGE_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        return False

    def is_simple_message(self, message: str) -> bool:
        """Check if message is simple (can use fast path).
        
        Public method for main.py to check before processing.
        """
        return self._is_simple_message(message)

    def _is_simple_message(self, message: str) -> bool:
        """Determine if message can use fast path (no agents needed).
        
        Fast path uses Claude Haiku for 3-4x faster responses.
        Use for: greetings, short casual messages, and questions about the character itself.
        """
        # Don't use fast path if knowledge lookup (database query) is needed
        if self._requires_knowledge_lookup(message):
            return False
        
        message_clean = message.strip()
        
        # Questions about the character (name, age, hobby, etc.) - use fast path
        character_patterns = [
            # English - questions about character
            r"\b(your name|what.*name|who are you|introduce yourself)\b",
            r"\b(how old|your age|age are you)\b",
            r"\b(your hobbies?|what.*like|favorite|favourite)\b",
            r"\b(where.*from|where.*live|born)\b",
            r"\b(your birthday|when.*born)\b",
            r"\b(your job|what.*do you do|occupation)\b",
            r"\b(your family|siblings|parents|friends)\b",
            # Japanese - questions about character
            r"(名前|なまえ|お名前)",  # name
            r"(何歳|なんさい|いくつ|年齢|歳)",  # age
            r"(趣味|しゅみ|好き|すき|きらい|嫌い)",  # hobby, likes, dislikes
            r"(どこ.*(住|すん)|出身|生まれ)",  # where from
            r"(誕生日|たんじょうび|いつ生まれ)",  # birthday
            r"(仕事|しごと|何してる)",  # job
            r"(家族|かぞく|兄弟|姉妹|友達|ともだち)",  # family, friends
            r"(性格|せいかく|どんな子)",  # personality
            r"(夢|ゆめ|将来|しょうらい)",  # dream, future
            r"(好きな(色|食べ物|動物|季節|場所))",  # favorite color/food/animal/season/place
            r"(苦手|にがて|怖い|こわい)",  # things scared of, not good at
        ]
        
        for pattern in character_patterns:
            if re.search(pattern, message_clean, re.IGNORECASE):
                print(f"[SIMPLE] Character question matched: {pattern}")
                return True
        
        # Simple greeting patterns
        simple_patterns = [
            # English greetings
            r"^(hi|hello|hey|yo|sup)[\s!！。]*$",
            r"^good\s*(morning|afternoon|evening|night)[\s!！。]*$",
            r"^(thanks|thank you|thx)[\s!！。]*$",
            r"^(bye|goodbye|see you|later)[\s!！。]*$",
            r"^(ok|okay|yes|no|yeah|yep|nope)[\s!！。]*$",
            # Japanese greetings
            r"^(おはよう|おはよ|こんにちは|こんばんは)[\s!！。ございます]*$",
            r"^(ありがとう|ありがと|サンキュー)[\s!！。ございます]*$",
            r"^(さようなら|じゃあね|バイバイ|またね)[\s!！。]*$",
            r"^(はい|いいえ|うん|ううん|そう|へー)[\s!！。]*$",
            r"^(やっほー|ヤッホー|よー|よっ)[\s!！。]*$",
            # Very short casual messages
            r"^(元気|疲れた|眠い|暇|忙しい|楽しい|嬉しい|悲しい)[\s!！。？?]*$",
            r"^(なに|何|えっ|へぇ|ふーん|そっか|なるほど)[\s!！。？?]*$",
        ]
        
        for pattern in simple_patterns:
            if re.search(pattern, message_clean, re.IGNORECASE):
                print(f"[SIMPLE] Greeting matched: {pattern}")
                return True
        
        # Default: use full pipeline (agents) for complex messages
        return False

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
        
        # Emotion-based actions (check message content directly for emotional expressions)
        emotion_patterns = [
            # Sad/lonely emotions -> sympathetic response
            (["寂しい", "さみしい", "さびしい", "悲しい", "かなしい", "lonely", "sad",
              "辛い", "つらい", "苦しい", "くるしい", "泣きたい", "泣いて", "crying",
              "会えない", "会いたい", "miss you", "miss him", "miss her"], "sympathetic"),
            # Happy/excited emotions -> smile
            (["嬉しい", "うれしい", "楽しい", "たのしい", "happy", "excited", "glad"], "smile"),
            # Worried/anxious emotions -> comfort
            (["心配", "しんぱい", "不安", "ふあん", "worried", "anxious", "nervous"], "comfort"),
        ]
        
        for keywords, action in emotion_patterns:
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
        on_waiting_audio: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """
        Fast path for simple conversational messages.
        
        Uses smart TTS strategy based on response length:
        - SHORT (< 50 words): Parallel TTS, NO waiting audio
        - MEDIUM (50-100 words): Notify frontend to play waiting audio, then full response
        - LONG (> 100 words): Notify frontend to play waiting audio, use VoiceVox
        
        Falls back to Haiku + VoiceVox if Gemini quota is exhausted.
        
        Args:
            on_waiting_audio: Async callback(phrase, phrase_index) called BEFORE TTS for MEDIUM/LONG.
                              Frontend has audio pre-loaded, saves bandwidth.
        """
        from services.speech_gemini import (
            get_gemini_text_speech_service,
            ResponseLength,
        )
        
        # Load conversation history for context (limit to 3 for speed)
        history = await load_conversation_history(session_id, limit=3)
        
        # Build messages with conversation history
        messages = []
        for conv in history:
            messages.append({"role": "user", "content": conv["user_message"]})
            messages.append({"role": "assistant", "content": conv["ai_response"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        print(f"[FAST PATH] Using smart TTS, {len(history)} prev conversations")
        
        # Smart TTS based on response length
        # Waiting audio notification is sent via on_waiting_audio callback (before TTS)
        gemini_service = get_gemini_text_speech_service()
        response_text, audio_url, response_length, _ = await gemini_service.generate_text_and_speech_smart(
            messages=messages,
            system_prompt=self._simple_system_prompt,
            max_tokens=150,  # Reduced from 200 to encourage shorter responses
            on_waiting_audio=on_waiting_audio,
        )
        
        # Use neutral emotion for simple messages
        response_emotion = "neutral"
        
        # Detect content type from user message for voice selection
        content_type = detect_content_type(user_message)
        
        # Determine action based on content type (simple mapping for fast path)
        character_action = self._get_simple_action(user_message, content_type)
        
        print(f"[FAST PATH] Length: {response_length.value}, Content: {content_type.value}, Action: {character_action}")

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
            "voice_analysis": None,  # Fast path doesn't analyze voice
            "audio_complete": True,  # Flag: audio is complete, no streaming needed
            "response_length": response_length.value,  # Length category for frontend
        }

    async def _run_emotion_crew(
        self, 
        user_message: str, 
        voice_features: VoiceFeatures | None = None
    ) -> str:
        """Run emotion analysis in separate crew (for parallel execution).
        
        Args:
            user_message: The text content of the user's message
            voice_features: Optional prosodic features extracted from user's voice
        """
        # Build voice context if available
        voice_context = ""
        if voice_features:
            voice_context = f"""
            {voice_features.to_context_string()}
            
            Use BOTH the voice analysis AND the text content to determine the user's true emotional state.
            Voice often reveals emotions more accurately than words alone.
            """
        
        emotion_task = Task(
            description=f"""
            Analyze the emotion in the following user message:
            
            "{user_message}"
            {voice_context}
            
            Report the user's emotional state and recommended response tone.
            Consider both text content and voice characteristics (if provided).
            """,
            expected_output="""
            - User's emotion: [emotion]
            - Voice signals: [voice-based emotion hints]
            - Text signals: [text-based emotion hints]
            - Combined analysis: [final assessment]
            - Content type: [genre if applicable]
            - Character action: [recommended action]
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

    async def process_message(
        self,
        user_message: str,
        session_id: str,
        audio_data: bytes | None = None,
        audio_mime_type: str = "audio/webm",
        on_waiting_audio: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """
        Process a user message through the crew pipeline.

        Uses fast path for simple messages, full pipeline for complex ones.
        Optionally analyzes voice features from audio for enhanced emotion detection.

        Args:
            user_message: The user's text message
            session_id: Session identifier for conversation tracking
            audio_data: Optional raw audio bytes for voice emotion analysis
            audio_mime_type: MIME type of the audio data
            on_waiting_audio: Async callback(phrase, phrase_index) called BEFORE TTS for MEDIUM/LONG.
                              Frontend has audio pre-loaded, saves bandwidth.

        Returns:
            Dict containing response text, audio URL, and emotion
        """
        # Analyze voice features if audio is provided
        voice_features: VoiceFeatures | None = None
        if audio_data:
            try:
                voice_emotion_service = get_voice_emotion_service()
                voice_features = await voice_emotion_service.analyze_audio(
                    audio_data, audio_mime_type
                )
                print(f"[VOICE] Analyzed voice features: {voice_features.emotion_hints}")
            except Exception as e:
                print(f"[VOICE] Failed to analyze voice: {e}")
                voice_features = None
        
        # Fast path for simple conversational messages (without voice analysis for speed)
        if self._is_simple_message(user_message) and not voice_features:
            print(f"[FAST PATH] Using direct LLM for simple message: {user_message[:50]}...")
            return await self._fast_path_response(user_message, session_id, on_waiting_audio)
        
        print(f"[PIPELINE] Processing message: {user_message[:50]}...")

        # Run emotion analysis + history loading in parallel (optimized 2-agent pipeline)
        # Voice features are passed to emotion crew for enhanced analysis
        emotion_result, history = await asyncio.gather(
            self._run_emotion_crew(user_message, voice_features),
            load_conversation_history(session_id, limit=10),
        )
        
        # Format conversation history for brain agent context
        history_context = format_conversation_history(history)
        print(f"[PIPELINE] Loaded {len(history)} previous conversations for context")
        
        # Check if this message likely needs database lookup
        needs_db_lookup = self._requires_knowledge_lookup(user_message)
        
        # Build voice context summary for brain agent
        voice_context_summary = ""
        if voice_features:
            voice_hints = ", ".join(voice_features.emotion_hints)
            voice_context_summary = f"""
            [Voice Analysis]
            The user's voice indicates: {voice_hints}
            """

        # Generate response with Brain Agent (has MCP tools for DB access)
        response_task = Task(
            description=f"""
            あなたは「{self.character_name}」！5歳の元気いっぱいな男の子として返答してね！
            
            [ユーザーのメッセージ]
            "{user_message}"
            {voice_context_summary}
            
            [会話履歴]
            {history_context}
            
            [感情分析]
            {emotion_result}
            
            [使えるツール]
            - movie_database_query: 映画やドラマを調べる時に使う
            - restaurant_database_query: レストランや食べ物を調べる時に使う
            
            {"[重要] このメッセージは情報を調べる必要がありそう。適切なツール（映画→movie_database_query、レストラン→restaurant_database_query）を使って検索してね。もし結果がなかったら「もうちょっと教えて！どんな〇〇がいい？」って聞いてね。" if needs_db_lookup else ""}
            
            [★キャラクター：5歳の男の子★]
            - 元気でテンション高め！「！」を多めに使う
            - 「〜だよ！」「〜なんだ！」「すごーい！」「ねえねえ！」
            - 「あのね」「えっとね」で話し始めることも
            - 好奇心旺盛で相手の話に興味津々
            - 敬語は使わない（子供だから）
            
            [会話のコツ]
            - 2〜3文くらいで返す（元気よく！）
            - 相手の話に共感「わかるー！」「いいね！」
            - 自分の好きなことも話す
            - 質問して会話を続ける
            
            [おすすめを教える時]
            子供が友達に教えるみたいに元気よく！
            
            ダメ（大人っぽい）:
            おすすめをご紹介いたします。
            • 店名A - 説明
            
            いいね（子供っぽい）:
            あのね、「店名A」ってところがすっごくおいしいんだよ！
            僕も大好きなの！行ってみて！
            """,
            expected_output="""5歳の男の子らしい元気な返答。2〜3文で、「！」多め、子供らしい言葉遣い。""",
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

        # Skip audio synthesis here - WebSocket handler will stream it sentence by sentence
        # This makes text response faster and avoids redundant synthesis
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

        # Build voice info for response
        voice_info = None
        if voice_features:
            voice_info = {
                "pitch": voice_features.pitch_mean,
                "energy": voice_features.energy_mean,
                "speaking_rate": voice_features.speaking_rate,
                "silence_ratio": voice_features.silence_ratio,
                "hints": voice_features.emotion_hints,
            }

        return {
            "text": response_text,
            "audio_url": audio_url,
            "emotion": response_emotion,
            "action": character_action,
            "content_type": content_type.value,
            "session_id": session_id,
            "voice_analysis": voice_info,
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
        
        Returns None to use default voice (Gemini TTS uses emotion-based styling).
        """
        # Gemini TTS voices: Puck, Charon, Kore, Fenrir, Aoede
        # For this character (young boy), we always use Kore
        # Voice styling is handled via emotion prompts, not voice switching
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
