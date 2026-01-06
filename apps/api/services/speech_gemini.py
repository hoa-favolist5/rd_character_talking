"""Gemini Text + TTS service.

Two-step process:
1. Generate text with gemini-2.5-flash (fast, smart)  
2. Generate audio with gemini-2.5-flash-preview-tts (TTS only)

Note: gemini-2.5-flash-tts is NOT available yet. Using preview version.
The TTS model only supports AUDIO modality, not TEXT+AUDIO combined.
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


def _now() -> str:
    """Return current time as HH:MM:SS.mmm string."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


class RateLimitError(Exception):
    """Raised when API rate limit is hit."""
    pass


class GeminiTextAndSpeechService:
    """Text generation + TTS using Gemini models.
    
    Two-step process:
    1. Generate text with gemini-2.5-flash (fast, smart)
    2. Generate audio with gemini-2.5-flash-preview-tts (TTS only)
    
    Optimizations:
    - Native async API
    - Request timeout to avoid hung requests
    - Fallback to AWS Polly on rate limit
    - Single TTS call to minimize API usage
    """

    REQUEST_TIMEOUT = 15.0
    MAX_RETRIES = 2

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: genai.Client | None = None
        self._text_model = "gemini-2.5-flash"  # For text generation
        self._tts_model = "gemini-2.5-flash-preview-tts"  # For audio only (stable version not available yet)
        self._voice = self._settings.gemini_tts_voice
        
        # Pre-initialize client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Gemini client eagerly."""
        if self._client is None and self._settings.google_api_key:
            try:
                self._client = genai.Client(api_key=self._settings.google_api_key)
            except Exception as e:
                print(f"[{_now()}] [Gemini T+S] Failed to init client: {e}")

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
            start_ts = _now()
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
            print(f"[{start_ts}] [Polly] ⚡ {len(text)} chars → {len(audio_data):,}B in {elapsed}ms")
            
            return audio_url
            
        except Exception as e:
            print(f"[{_now()}] [Polly] ✗ Failed: {e}")
            return None

    async def _generate_text(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
    ) -> tuple[str, int, str]:
        """Step 1: Generate text response using gemini-2.5-flash.
        
        Returns:
            (text, elapsed_ms, start_timestamp)
        """
        start_time = _now()
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
        
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.7,
        )
        
        async with asyncio.timeout(self.REQUEST_TIMEOUT):
            response = await client.aio.models.generate_content(
                model=self._text_model,
                contents=contents,
                config=config,
            )
        
        if not response.candidates or not response.candidates[0].content:
            raise RuntimeError("No response from Gemini text model")
        
        text = response.candidates[0].content.parts[0].text
        elapsed = _ms(start)
        return text, elapsed, start_time

    async def _generate_audio(self, text: str) -> tuple[bytes | None, int, str]:
        """Step 2: Generate audio from text using TTS model with retries.
        
        Returns:
            (audio_data, elapsed_ms, start_timestamp)
        """
        start_time = _now()
        start = time.perf_counter()
        client = self._get_client()
        
        # Speech config
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=self._voice,
                )
            )
        )
        
        # TTS model only supports AUDIO modality
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        )
        
        last_error: Exception | None = None
        
        # Retry loop for transient errors (400 INVALID_ARGUMENT can be transient)
        for attempt in range(1, self.MAX_RETRIES + 1):
            attempt_start = time.perf_counter()
            try:
                async with asyncio.timeout(self.REQUEST_TIMEOUT):
                    response = await client.aio.models.generate_content(
                        model=self._tts_model,
                        contents=text,
                        config=config,
                    )
                
                elapsed = _ms(attempt_start)
                
                if not response.candidates or not response.candidates[0].content:
                    print(f"[{_now()}] [Gemini TTS] ⚠ Attempt {attempt}/{self.MAX_RETRIES} no audio ({elapsed}ms)")
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(0.1)  # Brief delay before retry
                        continue
                    return None, _ms(start), start_time
                
                # Extract audio data
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        total_elapsed = _ms(start)
                        if attempt > 1:
                            print(f"[{_now()}] [Gemini TTS] ✓ Retry {attempt} succeeded ({elapsed}ms)")
                        return part.inline_data.data, total_elapsed, start_time
                
                # No audio data found
                print(f"[{_now()}] [Gemini TTS] ⚠ Attempt {attempt}/{self.MAX_RETRIES} no audio data ({elapsed}ms)")
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(0.1)
                    continue
                return None, _ms(start), start_time
                
            except Exception as e:
                elapsed = _ms(attempt_start)
                error_str = str(e)
                last_error = e
                
                # Check for rate limit - don't retry, propagate
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"[{_now()}] [Gemini TTS] ⚠ Rate limit hit ({elapsed}ms)")
                    raise RateLimitError(f"Gemini TTS quota exhausted: {e}") from e
                
                # Retry on 400/500 errors (can be transient)
                if attempt < self.MAX_RETRIES:
                    print(f"[{_now()}] [Gemini TTS] ⚠ Attempt {attempt}/{self.MAX_RETRIES} failed ({elapsed}ms): {e} - retrying...")
                    await asyncio.sleep(0.1)
                else:
                    print(f"[{_now()}] [Gemini TTS] ✗ All {self.MAX_RETRIES} attempts failed ({_ms(start)}ms): {e}")
                    return None, _ms(start), start_time
        
        return None, _ms(start), start_time

    async def generate_text_and_speech(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
    ) -> tuple[str, str | None]:
        """
        Generate text response AND audio in two steps.
        
        Step 1: Generate text with gemini-2.5-flash
        Step 2: Generate audio with gemini-2.5-flash-preview-tts (single TTS call)
        
        Args:
            messages: Conversation history [{"role": "user/assistant", "content": "..."}]
            system_prompt: System instruction for the character
            max_tokens: Maximum output tokens
        
        Returns:
            (response_text, audio_data_url) - audio_url may be None on TTS failure
        """
        total_start = time.perf_counter()
        last_error: Exception | None = None
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Step 1: Generate text
                text_response, text_time, text_ts = await self._generate_text(
                    messages=messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
                
                # Step 2: Generate audio (single TTS call)
                audio_data, tts_time, tts_ts = await self._generate_audio(text_response)
                
                # Step 3: Convert audio to data URL
                encode_ts = _now()
                encode_start = time.perf_counter()
                audio_url = None
                audio_size = 0
                if audio_data:
                    wav_data = self._pcm_to_wav(audio_data)
                    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                    audio_url = f"data:audio/wav;base64,{audio_base64}"
                    audio_size = len(wav_data)
                encode_time = _ms(encode_start)
                
                total_time = _ms(total_start)
                end_ts = _now()
                
                # Print detailed time breakdown with timestamps
                print(f"[Gemini T+S] ✓ {len(text_response)} chars, {audio_size:,}B audio")
                print(f"  ├─ [{text_ts}] Text:   {text_time:>4}ms (gemini-2.5-flash)")
                print(f"  ├─ [{tts_ts}] TTS:    {tts_time:>4}ms (gemini-2.5-flash-preview-tts)")
                print(f"  ├─ [{encode_ts}] Encode: {encode_time:>4}ms (PCM→WAV→Base64)")
                print(f"  └─ [{end_ts}] Total:  {total_time:>4}ms")
                
                return text_response, audio_url
                
            except asyncio.TimeoutError:
                elapsed = _ms(total_start)
                last_error = TimeoutError(f"Request timed out after {self.REQUEST_TIMEOUT}s")
                print(f"[{_now()}] [Gemini T+S] ⚠ Attempt {attempt}/{self.MAX_RETRIES} timeout ({elapsed}ms)")
                
            except RateLimitError:
                # Propagate rate limit errors to trigger fallback
                raise
                
            except Exception as e:
                last_error = e
                elapsed = _ms(total_start)
                error_str = str(e)
                
                # Check for rate limit / quota errors
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"[{_now()}] [Gemini T+S] ⚠ Rate limit hit ({elapsed}ms) - will use fallback")
                    raise RateLimitError(f"Gemini quota exhausted: {e}") from e
                
                if attempt < self.MAX_RETRIES:
                    print(f"[{_now()}] [Gemini T+S] ⚠ Attempt {attempt}/{self.MAX_RETRIES} failed ({elapsed}ms): {e} - retrying...")
                    await asyncio.sleep(0.05)
                else:
                    print(f"[{_now()}] [Gemini T+S] ✗ All attempts failed ({elapsed}ms): {e}")
        
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
            total_start = time.perf_counter()
            start_ts = _now()
            print(f"[{start_ts}] [Fallback] Using Anthropic + AWS Polly...")
            
            from services.llm import get_llm_service
            
            settings = get_settings()
            llm_service = get_llm_service()
            
            # Generate text with Anthropic Haiku
            text_ts = _now()
            text_start = time.perf_counter()
            text = await llm_service.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                model=settings.anthropic_fast_model,
            )
            text_time = _ms(text_start)
            
            # Generate audio with AWS Polly (faster and more reliable)
            polly_ts = _now()
            polly_start = time.perf_counter()
            audio_url = await self._synthesize_with_polly(text)
            polly_time = _ms(polly_start)
            
            total_time = _ms(total_start)
            end_ts = _now()
            
            # Print detailed time breakdown with timestamps
            print(f"[Fallback] ✓ {len(text)} chars")
            print(f"  ├─ [{text_ts}] Text:  {text_time:>4}ms (Anthropic {settings.anthropic_fast_model})")
            print(f"  ├─ [{polly_ts}] TTS:   {polly_time:>4}ms (AWS Polly)")
            print(f"  └─ [{end_ts}] Total: {total_time:>4}ms")
            
            return text, audio_url, True


# Global instance
gemini_text_speech_service = GeminiTextAndSpeechService()


def get_gemini_text_speech_service() -> GeminiTextAndSpeechService:
    """Get the global Gemini text+speech service instance."""
    return gemini_text_speech_service

