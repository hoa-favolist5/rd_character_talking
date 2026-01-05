"""Gemini 2.5 Flash combined Text + TTS service.

Single API call returns both text AND audio - no separate TTS needed!
This replaces the Haiku + Gemini TTS two-step process.
"""

import asyncio
import base64
import io
import time
import wave
from typing import AsyncGenerator

from google import genai
from google.genai import types

from config.settings import get_settings


def _ms(start: float) -> int:
    """Return elapsed milliseconds since start time."""
    return int((time.perf_counter() - start) * 1000)


class RateLimitError(Exception):
    """Raised when API rate limit is hit."""
    pass


class GeminiTextAndSpeechService:
    """Combined text generation + TTS using Gemini 2.5 Flash.
    
    Single API call returns both text and audio - no separate TTS needed.
    This is faster and uses less quota than separate calls.
    
    Optimizations:
    - Native async API
    - Request timeout to avoid hung requests
    - Fallback to Cloud TTS on rate limit
    """

    REQUEST_TIMEOUT = 15.0
    MAX_RETRIES = 2

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: genai.Client | None = None
        self._model = "gemini-2.5-flash-preview-tts"
        self._voice = self._settings.gemini_tts_voice
        
        # Pre-initialize client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Gemini client eagerly."""
        if self._client is None and self._settings.google_api_key:
            try:
                self._client = genai.Client(api_key=self._settings.google_api_key)
            except Exception as e:
                print(f"[Gemini T+S] Failed to init client: {e}")

    def _get_client(self) -> genai.Client:
        """Get Gemini client (creates if needed)."""
        if self._client is None:
            self._client = genai.Client(api_key=self._settings.google_api_key)
        return self._client

    def _pcm_to_wav(self, audio_data: bytes) -> bytes:
        """Convert raw PCM to WAV (24kHz, 16-bit, mono)."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)
        return wav_buffer.getvalue()

    async def _synthesize_with_polly(self, text: str) -> str | None:
        """Fallback TTS using AWS Polly (~100ms, very fast)."""
        import boto3
        
        try:
            start = time.perf_counter()
            
            client = boto3.client(
                "polly",
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id,
                aws_secret_access_key=self._settings.aws_secret_access_key,
            )
            
            # Use Takumi neural voice (Japanese)
            response = await asyncio.to_thread(
                client.synthesize_speech,
                Text=text,
                OutputFormat="mp3",
                VoiceId="Takumi",
                Engine="neural",
                LanguageCode="ja-JP",
            )
            
            audio_data = response["AudioStream"].read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            audio_url = f"data:audio/mp3;base64,{audio_base64}"
            
            elapsed = _ms(start)
            print(f"[Polly] ⚡ {len(text)} chars → {len(audio_data):,}B in {elapsed}ms")
            
            return audio_url
            
        except Exception as e:
            print(f"[Polly] ✗ Failed: {e}")
            return None

    async def generate_text_and_speech(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
    ) -> tuple[str, str | None]:
        """
        Generate text response AND audio in a single API call.
        
        Args:
            messages: Conversation history [{"role": "user/assistant", "content": "..."}]
            system_prompt: System instruction for the character
            max_tokens: Maximum output tokens
        
        Returns:
            (response_text, audio_data_url) - audio_url may be None on fallback
        """
        start = time.perf_counter()
        client = self._get_client()
        
        # Build conversation for Gemini
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role, 
                parts=[types.Part(text=msg["content"])]
            ))
        
        # Speech config
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=self._voice,
                )
            )
        )
        
        # Request BOTH text AND audio output
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_modalities=["TEXT", "AUDIO"],  # KEY: Both text and audio!
            speech_config=speech_config,
            max_output_tokens=max_tokens,
            temperature=0.7,
        )
        
        last_error: Exception | None = None
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            api_start = time.perf_counter()
            try:
                async with asyncio.timeout(self.REQUEST_TIMEOUT):
                    response = await client.aio.models.generate_content(
                        model=self._model,
                        contents=contents,
                        config=config,
                    )
                api_time = _ms(api_start)
                
                # Extract text and audio from response
                if not response.candidates or not response.candidates[0].content:
                    raise RuntimeError("No response from Gemini")
                
                parts = response.candidates[0].content.parts
                
                text_response = ""
                audio_data = None
                
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        text_response = part.text
                    elif hasattr(part, 'inline_data') and part.inline_data:
                        audio_data = part.inline_data.data
                
                if not text_response:
                    raise RuntimeError("No text in response")
                
                # Convert audio to data URL
                audio_url = None
                if audio_data:
                    wav_data = self._pcm_to_wav(audio_data)
                    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                    audio_url = f"data:audio/wav;base64,{audio_base64}"
                    audio_size = len(wav_data)
                    print(f"[Gemini T+S] ✓ Text+Audio in {api_time}ms | {len(text_response)} chars, {audio_size:,}B audio")
                else:
                    print(f"[Gemini T+S] ✓ Text only in {api_time}ms | {len(text_response)} chars (no audio)")
                
                return text_response, audio_url
                
            except asyncio.TimeoutError:
                elapsed = _ms(api_start)
                last_error = TimeoutError(f"Request timed out after {self.REQUEST_TIMEOUT}s")
                print(f"[Gemini T+S] ⚠ Attempt {attempt}/{self.MAX_RETRIES} timeout ({elapsed}ms)")
                
            except Exception as e:
                last_error = e
                elapsed = _ms(api_start)
                error_str = str(e)
                
                # Check for rate limit / quota errors
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"[Gemini T+S] ⚠ Rate limit hit ({elapsed}ms) - will use fallback")
                    raise RateLimitError(f"Gemini quota exhausted: {e}") from e
                
                if attempt < self.MAX_RETRIES:
                    print(f"[Gemini T+S] ⚠ Attempt {attempt}/{self.MAX_RETRIES} failed ({elapsed}ms): {e} - retrying...")
                    await asyncio.sleep(0.05)
                else:
                    print(f"[Gemini T+S] ✗ All attempts failed ({elapsed}ms): {e}")
        
        raise RuntimeError(f"Gemini T+S failed after {self.MAX_RETRIES} attempts") from last_error

    async def generate_text_and_speech_with_fallback(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
    ) -> tuple[str, str | None, bool]:
        """
        Generate text+speech with fallback to separate TTS on rate limit.
        
        Returns:
            (response_text, audio_url, used_fallback)
        """
        try:
            text, audio_url = await self.generate_text_and_speech(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            return text, audio_url, False
            
        except RateLimitError:
            # Fallback: Use Anthropic for text, AWS Polly for audio
            print(f"[Gemini T+S] Falling back to Anthropic + AWS Polly...")
            
            from services.llm import get_llm_service
            
            settings = get_settings()
            llm_service = get_llm_service()
            
            # Generate text with Anthropic Haiku
            text = await llm_service.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                model=settings.anthropic_fast_model,
            )
            
            # Generate audio with AWS Polly (faster and more reliable)
            audio_url = await self._synthesize_with_polly(text)
            
            print(f"[Fallback] ✓ Anthropic + Polly | {len(text)} chars")
            return text, audio_url, True


# Global instance
gemini_text_speech_service = GeminiTextAndSpeechService()


def get_gemini_text_speech_service() -> GeminiTextAndSpeechService:
    """Get the global Gemini text+speech service instance."""
    return gemini_text_speech_service

