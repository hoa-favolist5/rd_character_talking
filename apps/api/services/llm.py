"""Anthropic Claude LLM service."""

from typing import AsyncIterator

from anthropic import AsyncAnthropic

from config.settings import Settings, get_settings


class AnthropicService:
    """LLM service using Anthropic Claude API directly with async client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = AsyncAnthropic(api_key=self._settings.anthropic_api_key)

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str:
        """Generate a response using Claude (truly async)."""
        response = await self._client.messages.create(
            model=model or self._settings.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        )

        return response.content[0].text

    async def generate_response_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using Claude (truly async)."""
        async with self._client.messages.stream(
            model=self._settings.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def analyze_emotion(self, text: str) -> str:
        """Analyze the emotional content of user text using fast Haiku model."""
        messages = [
            {
                "role": "user",
                "content": f"""Analyze the emotion in the following text.
Answer with only one of the following words:
happy, sad, confused, surprised, neutral, frustrated, curious

Text: {text}""",
            }
        ]

        # Use Haiku for fast emotion analysis
        response = await self.generate_response(
            messages=messages,
            system_prompt="You are an emotion analysis expert. Accurately determine the emotion from the text.",
            max_tokens=20,
            temperature=0.3,
            model="claude-3-haiku-20240307",  # Fast model for simple classification
        )

        emotion = response.strip().lower()
        valid_emotions = ["happy", "sad", "confused", "surprised", "neutral", "frustrated", "curious"]
        
        return emotion if emotion in valid_emotions else "neutral"


# Global instance (lazy loaded)
_llm_service: AnthropicService | None = None


def get_llm_service() -> AnthropicService:
    """Get LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = AnthropicService()
    return _llm_service
