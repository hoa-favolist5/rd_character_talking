"""AWS Polly tool for CrewAI agents."""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from services.speech import speech_service


class SynthesizeSpeechInput(BaseModel):
    """Input schema for speech synthesis tool."""

    text: str = Field(description="Text to convert to speech")
    voice_id: str | None = Field(default=None, description="Voice ID to use (e.g., Takumi, Mizuki)")
    emotion: str = Field(default="neutral", description="Emotion (happy, sad, excited, etc.)")


class SynthesizeSpeechTool(BaseTool):
    """Tool for synthesizing speech from text using AWS Polly."""

    name: str = "synthesize_speech"
    description: str = """
    Converts text to speech.
    Use this when outputting AI responses as audio.
    Supports neural voices for natural-sounding speech.
    Use the emotion parameter to adjust the speaking tone.
    """
    args_schema: type[BaseModel] = SynthesizeSpeechInput

    def _run(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
    ) -> str:
        """Execute the speech synthesis."""
        import asyncio

        try:
            # Run async synthesis
            _, audio_url = asyncio.run(
                speech_service.synthesize_speech(text, voice_id, emotion)
            )
            
            return audio_url

        except Exception as e:
            return f"Speech synthesis error: {str(e)}"
