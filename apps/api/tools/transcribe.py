"""AWS Transcribe tool for CrewAI agents."""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from services.speech import speech_service


class TranscribeAudioInput(BaseModel):
    """Input schema for transcription tool."""

    audio_base64: str = Field(description="Base64 encoded audio data")
    mime_type: str = Field(default="audio/webm", description="Audio MIME type")


class TranscribeAudioTool(BaseTool):
    """Tool for transcribing audio to text using AWS Transcribe."""

    name: str = "transcribe_audio"
    description: str = """
    Converts audio files to text.
    Use this when processing voice input from users.
    Supports speech recognition for multiple languages.
    """
    args_schema: type[BaseModel] = TranscribeAudioInput

    def _run(self, audio_base64: str, mime_type: str = "audio/webm") -> str:
        """Execute the transcription."""
        import asyncio
        import base64

        try:
            # Decode base64 audio
            audio_data = base64.b64decode(audio_base64)
            
            # Run async transcription
            transcript = asyncio.run(
                speech_service.transcribe_audio(audio_data, mime_type)
            )
            
            return transcript

        except Exception as e:
            return f"Speech recognition error: {str(e)}"
