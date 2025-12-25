"""Character Crew - Multi-agent orchestration for AI character interaction."""

import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from langchain_anthropic import ChatAnthropic

from agents.brain_agent import create_brain_agent
from agents.knowledge_agent import create_knowledge_agent
from agents.emotion_agent import create_emotion_agent
from config.settings import get_settings
from services.speech import speech_service
from tools.database import SaveConversationTool


class CharacterCrew:
    """
    Main crew for handling character interactions.

    Orchestrates multiple agents to:
    1. Analyze user emotion
    2. Retrieve relevant knowledge
    3. Generate character response
    4. Synthesize speech output
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

        # Initialize Anthropic Claude LLM
        self.llm = ChatAnthropic(
            model=self.settings.anthropic_model,
            api_key=self.settings.anthropic_api_key,
            temperature=0.7,
            max_tokens=500,
        )

        # Create agents
        self.brain_agent = create_brain_agent(
            llm=self.llm,
            character_name=character_name,
            personality=personality,
            system_prompt=system_prompt,
        )
        self.knowledge_agent = create_knowledge_agent(self.llm)
        self.emotion_agent = create_emotion_agent(self.llm)

    async def process_message(
        self,
        user_message: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Process a user message through the full crew pipeline.

        Args:
            user_message: The user's text message
            session_id: Session identifier for conversation tracking

        Returns:
            Dict containing response text, audio URL, and emotion
        """
        # Create tasks
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

        # Run emotion and knowledge tasks in parallel
        parallel_crew = Crew(
            agents=[self.emotion_agent, self.knowledge_agent],
            # agents=[self.knowledge_agent],
            tasks=[emotion_task, knowledge_task],
            # tasks=[knowledge_task],
            process=Process.sequential,  # They don't depend on each other
            verbose=True,
        )

        # Run blocking crew in thread pool to not block event loop
        import asyncio
        parallel_results = await asyncio.to_thread(parallel_crew.kickoff)

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
            verbose=True,
        )

        # Run blocking crew in thread pool to not block event loop
        response_result = await asyncio.to_thread(response_crew.kickoff)
        response_text = str(response_result)
        
        print(f"[DEBUG] Response text: {response_text[:100]}...")

        # Determine emotion for TTS
        response_emotion = self._extract_emotion(str(parallel_results))

        # Synthesize speech
        try:
            _, audio_url = await speech_service.synthesize_speech(
                text=response_text,
                voice_id=self.voice_id,
                emotion=response_emotion,
            )
        except Exception as e:
            print(f"Speech synthesis error: {e}")
            audio_url = None

        # Save conversation
        save_tool = SaveConversationTool()
        save_tool._run(
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
