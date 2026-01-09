"""Anthropic Text + ElevenLabs/Gemini TTS service.

Two-step process:
1. Generate text with Anthropic Claude (fast, avoids Gemini rate limits)
2. Generate audio with parallel TTS (ElevenLabs + Gemini)

Text Generation:
- Uses Anthropic claude-3-5-haiku for fast text generation

TTS Strategy (priority order):
- ElevenLabs (PRIMARY): Fast ~200-500ms, high quality
- Gemini TTS (FALLBACK): ~2-5s, cloud, natural when available

Performance:
- Anthropic Text: ~100-300ms
- ElevenLabs TTS: ~200-500ms, cloud, high quality (PRIMARY)
- Gemini TTS: ~2-5s, cloud, natural voice (FALLBACK)
"""

import asyncio
import base64
import io
import re
import time
import wave
from enum import Enum
from typing import AsyncGenerator, Awaitable, Callable

from google import genai
from google.genai import types

from config.settings import get_settings


class ResponseLength(Enum):
    """Response length categories for TTS strategy selection."""
    SHORT = "short"      # < 50 words: Parallel TTS (ElevenLabs + Gemini)
    MEDIUM = "medium"    # 50-100 words: Waiting audio + Parallel TTS
    LONG = "long"        # > 100 words: ElevenLabs primary (more consistent for long text)


# Thresholds for word count
WORD_COUNT_SHORT = 50
WORD_COUNT_LONG = 100


def count_words(text: str) -> int:
    """Count words in text, supporting both Japanese and English."""
    # For Japanese: count characters (roughly 1 char = 0.5 word equivalent)
    # For English: count word boundaries
    
    # Remove punctuation for cleaner counting
    clean_text = re.sub(r'[。、！？!?,.…]+', ' ', text)
    
    # Count Japanese characters (hiragana, katakana, kanji)
    jp_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', clean_text))
    
    # Count English words
    en_words = len(re.findall(r'[a-zA-Z]+', clean_text))
    
    # Japanese characters count as ~0.5 words each (roughly)
    # This gives a reasonable estimate for mixed text
    total_words = en_words + (jp_chars // 2)
    
    return max(total_words, jp_chars // 3)  # At minimum, count JP chars / 3


def categorize_response_length(text: str) -> ResponseLength:
    """Categorize response length for TTS strategy selection."""
    word_count = count_words(text)
    
    if word_count < WORD_COUNT_SHORT:
        return ResponseLength.SHORT
    elif word_count <= WORD_COUNT_LONG:
        return ResponseLength.MEDIUM
    else:
        return ResponseLength.LONG


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
    """Text generation (Anthropic) + TTS (ElevenLabs/Gemini) service.
    
    Two-step process:
    1. Generate text with Anthropic Claude (avoids Gemini rate limits)
    2. Generate audio with parallel TTS (ElevenLabs + Gemini)
    
    Text Generation:
    - Uses Anthropic claude-3-5-haiku (~100-300ms)
    
    Parallel TTS Strategy:
    - Start both ElevenLabs and Gemini TTS simultaneously
    - ElevenLabs is PRIMARY (faster, ~200-500ms)
    - Gemini is FALLBACK (slower but natural, ~2-5s)
    - Use ElevenLabs if it succeeds, otherwise use Gemini
    
    Performance:
    - Anthropic Text: ~100-300ms
    - ElevenLabs TTS: ~200-500ms, cloud, high quality (PRIMARY)
    - Gemini TTS: ~2-5s, cloud, natural voice (FALLBACK)
    
    Optimizations:
    - Native async API with parallel execution
    - Gemini runs as backup in parallel
    - Cached waiting audio for instant playback
    """

    REQUEST_TIMEOUT = 15.0  # For text generation
    TTS_TIMEOUT = 8.0       # Max wait for Gemini TTS (hard timeout)
    ELEVENLABS_TIMEOUT = 5.0  # Timeout for ElevenLabs before checking Gemini
    MAX_RETRIES = 1         # No retry for TTS - use parallel fallback
    
    # Waiting phrases (randomly selected for variety)
    WAITING_PHRASES = [
        "今からちょっと確認するね。",
        "うん、今確認してるよ。",
        "OK、少し待っててね。",
        "任せて、探してみるね。",
        "ちょっと考えてみるね。",
        "うんうん、今見てるよ。",
        "今確認するね。",
        "今調べてるよ。",
        "すぐ確認するね。",
        "うん、ちょっと待ってね。",
        "了解、今チェックしてるよ。",
        "OK、今確認中だよ。",
        "ちょっと待ってね、今確認するね。",
        "はーい、少し待ってね。",
        "今対応するね。",
        "えっと、確認してみるね。",
        "大丈夫、任せてね。",
        "今確認してるから、少し待ってね。",
        "今探してるところだよ。",
        "ちょっとだけ時間もらうね。"
    ]

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: genai.Client | None = None
        self._text_model = "gemini-2.5-flash"  # For text generation
        self._tts_model = "gemini-2.5-flash-preview-tts"  # For audio only (stable version not available yet)
        self._voice = self._settings.gemini_tts_voice
        
        # ElevenLabs timeout from settings (default 5s)
        self.ELEVENLABS_TIMEOUT = self._settings.elevenlabs_timeout
        
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

    async def _synthesize_with_elevenlabs(self, text: str) -> str | None:
        """Primary TTS using ElevenLabs (~200-500ms, cloud, high quality)."""
        from services.speech_elevenlabs import get_elevenlabs_service
        
        try:
            elevenlabs = get_elevenlabs_service()
            _, audio_url = await elevenlabs.synthesize_speech(text)
            return audio_url
            
        except Exception as e:
            print(f"[{_now()}] [ElevenLabs] ✗ Failed: {e}")
            return None

    async def _synthesize_parallel_tts(self, text: str) -> tuple[str | None, str]:
        """
        Run ElevenLabs and Gemini TTS in parallel.
        
        Strategy:
        - Start both TTS engines simultaneously
        - ElevenLabs is PRIMARY (faster, ~200-500ms)
        - Gemini is FALLBACK (slower, ~2-5s)
        - Use ElevenLabs if it succeeds, otherwise use Gemini
        
        Returns:
            (audio_url, source) - Audio data URL and which engine was used
        """
        from services.speech_elevenlabs import get_elevenlabs_service
        
        start = time.perf_counter()
        start_ts = _now()
        
        # Create tasks for both TTS engines
        async def elevenlabs_tts() -> tuple[str | None, int]:
            """Run ElevenLabs TTS and return (audio_url, elapsed_ms)."""
            el_start = time.perf_counter()
            try:
                elevenlabs = get_elevenlabs_service()
                _, audio_url = await elevenlabs.synthesize_speech(text)
                elapsed = _ms(el_start)
                return audio_url, elapsed
            except Exception as e:
                elapsed = _ms(el_start)
                print(f"[{_now()}] [Parallel TTS] ElevenLabs error ({elapsed}ms): {e}")
                return None, elapsed
        
        async def gemini_tts() -> tuple[bytes | None, int]:
            """Run Gemini TTS and return (audio_data, elapsed_ms)."""
            tts_start = time.perf_counter()
            try:
                audio_data, _, _ = await self._generate_audio(text)
                elapsed = _ms(tts_start)
                return audio_data, elapsed
            except Exception as e:
                elapsed = _ms(tts_start)
                print(f"[{_now()}] [Parallel TTS] Gemini error ({elapsed}ms): {e}")
                return None, elapsed
        
        # Run both in parallel
        elevenlabs_task = asyncio.create_task(elevenlabs_tts())
        gemini_task = asyncio.create_task(gemini_tts())
        
        print(f"[{start_ts}] [Parallel TTS] Starting ElevenLabs + Gemini for: {text[:40]}...")
        
        # Wait for ElevenLabs first (it's usually faster - PRIMARY)
        elevenlabs_result: str | None = None
        elevenlabs_time: int = 0
        
        try:
            elevenlabs_result, elevenlabs_time = await asyncio.wait_for(
                elevenlabs_task,
                timeout=self.ELEVENLABS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            elevenlabs_time = _ms(start)
            print(f"[{_now()}] [Parallel TTS] ElevenLabs timeout after {self.ELEVENLABS_TIMEOUT}s ({elevenlabs_time}ms)")
        except Exception as e:
            elevenlabs_time = _ms(start)
            print(f"[{_now()}] [Parallel TTS] ElevenLabs task failed ({elevenlabs_time}ms): {e}")
        
        # If ElevenLabs succeeded, use it and cancel Gemini
        if elevenlabs_result is not None:
            total_time = _ms(start)
            # Cancel Gemini task - we don't need it
            gemini_task.cancel()
            try:
                await gemini_task
            except asyncio.CancelledError:
                pass
            
            print(f"[{_now()}] [Parallel TTS] ✓ Using ElevenLabs ({elevenlabs_time}ms)")
            print(f"  ├─ ElevenLabs: {elevenlabs_time:>4}ms ← SELECTED (PRIMARY)")
            print(f"  ├─ Gemini:     cancelled (not needed)")
            print(f"  └─ Total:      {total_time:>4}ms")
            
            return elevenlabs_result, "elevenlabs"
        
        # ElevenLabs failed, wait for Gemini (FALLBACK)
        gemini_result: bytes | None = None
        gemini_time: int = 0
        
        try:
            # Calculate remaining time
            elapsed_so_far = _ms(start) / 1000.0
            remaining_timeout = max(0.1, self.TTS_TIMEOUT - elapsed_so_far)
            
            gemini_result, gemini_time = await asyncio.wait_for(
                gemini_task,
                timeout=remaining_timeout,
            )
        except asyncio.TimeoutError:
            gemini_time = _ms(start)
            print(f"[{_now()}] [Parallel TTS] Gemini timeout after {self.TTS_TIMEOUT}s ({gemini_time}ms)")
            gemini_task.cancel()
            try:
                await gemini_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            gemini_time = _ms(start)
            print(f"[{_now()}] [Parallel TTS] Gemini task failed ({gemini_time}ms): {e}")
        
        total_time = _ms(start)
        
        # Decision: Use Gemini as fallback
        if gemini_result is not None:
            # Gemini succeeded - convert to data URL
            wav_data = self._pcm_to_wav(gemini_result)
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            audio_url = f"data:audio/wav;base64,{audio_base64}"
            
            print(f"[{_now()}] [Parallel TTS] ✓ Using Gemini ({gemini_time}ms) [FALLBACK]")
            print(f"  ├─ ElevenLabs: {elevenlabs_time:>4}ms (failed/timeout)")
            print(f"  ├─ Gemini:     {gemini_time:>4}ms ← SELECTED (FALLBACK)")
            print(f"  └─ Total:      {total_time:>4}ms")
            
            return audio_url, "gemini"
        
        else:
            # Both failed
            print(f"[{_now()}] [Parallel TTS] ✗ Both TTS engines failed ({total_time}ms)")
            return None, "none"

    def get_waiting_phrase(self) -> tuple[str, int]:
        """Get a random waiting phrase for medium/long responses.
        
        Frontend has these audio files pre-loaded, so we just send the phrase ID.
        This saves bandwidth by not streaming audio for waiting messages.
        
        Returns:
            (phrase_text, phrase_index) - The waiting phrase and its index (0-based)
        """
        import random
        
        # Select a random phrase index
        phrase_index = random.randint(0, len(self.WAITING_PHRASES) - 1)
        phrase = self.WAITING_PHRASES[phrase_index]
        
        print(f"[{_now()}] [Waiting] Selected phrase #{phrase_index}: {phrase}")
        return phrase, phrase_index

    async def _generate_text(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
    ) -> tuple[str, int, str]:
        """Step 1: Generate text response using Anthropic Claude.
        
        Uses Anthropic instead of Gemini to avoid Gemini rate limits.
        Gemini is reserved for TTS only.
        
        Returns:
            (text, elapsed_ms, start_timestamp)
        """
        from services.llm import get_llm_service
        
        start_time = _now()
        start = time.perf_counter()
        
        llm_service = get_llm_service()
        
        # Use fast Haiku model for quick responses
        text = await llm_service.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            model=self._settings.anthropic_fast_model,  # claude-3-5-haiku
        )
        
        elapsed = _ms(start)
        print(f"[{start_time}] [Anthropic] ✓ Text generated in {elapsed}ms ({len(text)} chars)")
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
        
        print(f"[{_now()}] [Gemini TTS] 🎤 Request: model={self._tts_model}, voice={self._voice}, text={text[:50]}...")
        
        # Single attempt with fast timeout (3s) - no retry, use Polly instead
        try:
            async with asyncio.timeout(self.TTS_TIMEOUT):
                response = await client.aio.models.generate_content(
                    model=self._tts_model,
                    contents=text,
                    config=config,
                )
            
            elapsed = _ms(start)
            
            # Detailed response logging
            print(f"[{_now()}] [Gemini TTS] 📥 Response received ({elapsed}ms)")
            
            # Log response structure
            if not response:
                print(f"[{_now()}] [Gemini TTS]   └─ Response is None/empty")
                return None, elapsed, start_time
            elif not response.candidates:
                print(f"[{_now()}] [Gemini TTS]   └─ No candidates in response")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"[{_now()}] [Gemini TTS]   └─ Prompt feedback: {response.prompt_feedback}")
                return None, elapsed, start_time
            
            candidate = response.candidates[0]
            print(f"[{_now()}] [Gemini TTS]   ├─ Candidates: {len(response.candidates)}")
            
            # Log finish reason
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                print(f"[{_now()}] [Gemini TTS]   ├─ Finish reason: {candidate.finish_reason}")
            
            # Log safety ratings
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                blocked = [r for r in candidate.safety_ratings if hasattr(r, 'blocked') and r.blocked]
                if blocked:
                    print(f"[{_now()}] [Gemini TTS]   ├─ ⚠ Safety blocked: {blocked}")
            
            if not candidate.content:
                print(f"[{_now()}] [Gemini TTS]   └─ Content is None/empty")
                return None, elapsed, start_time
            elif not candidate.content.parts:
                print(f"[{_now()}] [Gemini TTS]   └─ No parts in content")
                return None, elapsed, start_time
            
            print(f"[{_now()}] [Gemini TTS]   ├─ Parts: {len(candidate.content.parts)}")
            for i, part in enumerate(candidate.content.parts):
                part_type = type(part).__name__
                has_inline = hasattr(part, 'inline_data') and part.inline_data
                has_text = hasattr(part, 'text') and part.text
                if has_inline:
                    mime = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else 'unknown'
                    size = len(part.inline_data.data) if part.inline_data.data else 0
                    print(f"[{_now()}] [Gemini TTS]   │  └─ Part[{i}]: {part_type}, inline_data={mime}, size={size:,}B")
                elif has_text:
                    print(f"[{_now()}] [Gemini TTS]   │  └─ Part[{i}]: {part_type}, text={part.text[:50]}...")
                else:
                    print(f"[{_now()}] [Gemini TTS]   │  └─ Part[{i}]: {part_type}, no inline_data or text")
            
            # Extract audio data
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    audio_size = len(part.inline_data.data) if part.inline_data.data else 0
                    print(f"[{_now()}] [Gemini TTS] ✓ Success ({elapsed}ms), audio={audio_size:,}B")
                    return part.inline_data.data, elapsed, start_time
            
            # No audio data found in parts
            print(f"[{_now()}] [Gemini TTS] ⚠ No audio data in parts ({elapsed}ms)")
            return None, elapsed, start_time
            
        except asyncio.TimeoutError:
            elapsed = _ms(start)
            print(f"[{_now()}] [Gemini TTS] ⏱ Timeout after {self.TTS_TIMEOUT}s ({elapsed}ms) → use Polly")
            return None, elapsed, start_time
            
        except Exception as e:
            elapsed = _ms(start)
            error_str = str(e)
            
            # Log full exception details
            print(f"[{_now()}] [Gemini TTS] ❌ Exception ({elapsed}ms): {type(e).__name__}: {error_str}")
            
            # Check for rate limit
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                print(f"[{_now()}] [Gemini TTS] ⚠ Rate limit → use ElevenLabs")
                raise RateLimitError(f"Gemini quota exhausted: {e}") from e
            
            # Any other error → ElevenLabs is primary, this is just fallback
            print(f"[{_now()}] [Gemini TTS] ⚠ Error (fallback failed)")
            return None, elapsed, start_time

    async def generate_text_and_speech(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
    ) -> tuple[str, str | None]:
        """
        Generate text response AND audio in two steps.
        
        Step 1: Generate text with Anthropic Claude
        Step 2: Generate audio with parallel TTS (ElevenLabs + Gemini)
               - ElevenLabs is PRIMARY (faster, ~200-500ms)
               - Gemini is FALLBACK (slower, ~2-5s)
        
        Args:
            messages: Conversation history [{"role": "user/assistant", "content": "..."}]
            system_prompt: System instruction for the character
            max_tokens: Maximum output tokens
        
        Returns:
            (response_text, audio_data_url) - audio_url from TTS engine
        """
        total_start = time.perf_counter()
        
        try:
            # Step 1: Generate text
            text_response, text_time, text_ts = await self._generate_text(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            
            # Step 2: Parallel TTS (ElevenLabs primary + Gemini fallback)
            tts_ts = _now()
            tts_start = time.perf_counter()
            audio_url, tts_source = await self._synthesize_parallel_tts(text_response)
            tts_time = _ms(tts_start)
            
            total_time = _ms(total_start)
            end_ts = _now()
            
            # Print detailed time breakdown with timestamps
            print(f"[Text+TTS] ✓ {len(text_response)} chars")
            print(f"  ├─ [{text_ts}] Text: {text_time:>4}ms (Anthropic)")
            print(f"  ├─ [{tts_ts}] TTS:  {tts_time:>4}ms ({tts_source})")
            print(f"  └─ [{end_ts}] Total: {total_time:>4}ms")
            
            return text_response, audio_url
            
        except RateLimitError:
            # Propagate rate limit errors to trigger full fallback
            raise
            
        except Exception as e:
            elapsed = _ms(total_start)
            error_str = str(e)
            
            # Check for rate limit / quota errors
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                print(f"[{_now()}] [Text+TTS] ⚠ Rate limit ({elapsed}ms) → use fallback")
                raise RateLimitError(f"API quota exhausted: {e}") from e
            
            print(f"[{_now()}] [Text+TTS] ❌ Error ({elapsed}ms): {e}")
            raise RuntimeError(f"Text+TTS failed: {e}") from e

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
            # Fallback: Use Anthropic for text, ElevenLabs for audio
            total_start = time.perf_counter()
            start_ts = _now()
            print(f"[{start_ts}] [Fallback] Using Anthropic + ElevenLabs...")
            
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
            
            # Generate audio with ElevenLabs (fast and high quality)
            elevenlabs_ts = _now()
            elevenlabs_start = time.perf_counter()
            audio_url = await self._synthesize_with_elevenlabs(text)
            elevenlabs_time = _ms(elevenlabs_start)
            
            total_time = _ms(total_start)
            end_ts = _now()
            
            # Print detailed time breakdown with timestamps
            print(f"[Fallback] ✓ {len(text)} chars")
            print(f"  ├─ [{text_ts}] Text:     {text_time:>4}ms (Anthropic {settings.anthropic_fast_model})")
            print(f"  ├─ [{elevenlabs_ts}] TTS:      {elevenlabs_time:>4}ms (ElevenLabs)")
            print(f"  └─ [{end_ts}] Total:    {total_time:>4}ms")
            
            return text, audio_url, True

    async def generate_text_and_speech_smart(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
        on_waiting_audio: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> tuple[str, str | None, ResponseLength, int | None]:
        """
        Smart text+speech generation with response length strategy.
        
        Strategy:
        - SHORT (< 50 words): Parallel TTS (ElevenLabs primary + Gemini fallback)
        - MEDIUM (50-100 words): Notify frontend to play waiting audio, then Parallel TTS
        - LONG (> 100 words): Notify frontend to play waiting audio, then ElevenLabs
        
        Args:
            messages: Conversation history
            system_prompt: System instruction for the character
            max_tokens: Maximum output tokens
            on_waiting_audio: Async callback(phrase, phrase_index) called BEFORE TTS for MEDIUM/LONG.
                              Frontend has audio files pre-loaded, just needs the index.
        
        Returns:
            (response_text, audio_url, response_length, waiting_phrase_index)
            - waiting_phrase_index is set for MEDIUM/LONG responses (0-based index)
        """
        total_start = time.perf_counter()
        start_ts = _now()
        
        try:
            # Step 1: Generate text first
            text_response, text_time, text_ts = await self._generate_text(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            
            # Step 2: Categorize response length
            response_length = categorize_response_length(text_response)
            word_count = count_words(text_response)
            print(f"[{_now()}] [Smart] Response: {word_count} words → {response_length.value}")
            
            # Step 3: For MEDIUM/LONG, notify frontend to play waiting audio BEFORE TTS starts
            waiting_phrase_index = None
            
            if response_length in (ResponseLength.MEDIUM, ResponseLength.LONG):
                # Get waiting phrase (frontend has audio pre-loaded, saves bandwidth)
                wait_phrase, phrase_index = self.get_waiting_phrase()
                waiting_phrase_index = phrase_index
                
                # Call the callback to notify frontend to play waiting audio (before TTS)
                if on_waiting_audio:
                    print(f"[{_now()}] [Smart] Notifying frontend to play waiting audio #{phrase_index}: {wait_phrase}")
                    await on_waiting_audio(wait_phrase, phrase_index)
            
            # Step 4: Generate audio based on length
            if response_length == ResponseLength.LONG:
                # LONG: Use ElevenLabs directly (more consistent for long text)
                print(f"[{_now()}] [Smart] Using ElevenLabs for long response")
                audio_url = await self._synthesize_with_elevenlabs(text_response)
                
            elif response_length == ResponseLength.MEDIUM:
                # MEDIUM: Parallel TTS (waiting audio already sent above)
                audio_url, tts_source = await self._synthesize_parallel_tts(text_response)
                print(f"[{_now()}] [Smart] MEDIUM response used: {tts_source}")
                    
            else:
                # SHORT: Parallel TTS (ElevenLabs primary + Gemini fallback), NO waiting audio
                audio_url, tts_source = await self._synthesize_parallel_tts(text_response)
                print(f"[{_now()}] [Smart] SHORT response used: {tts_source}")
            
            total_time = _ms(total_start)
            print(f"[{_now()}] [Smart] ✓ {len(text_response)} chars, {word_count} words in {total_time}ms")
            
            return text_response, audio_url, response_length, waiting_phrase_index
            
        except RateLimitError:
            # Fallback to Anthropic + ElevenLabs
            print(f"[{_now()}] [Smart] Rate limited, using fallback")
            text, audio_url, _ = await self.generate_text_and_speech_with_fallback(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            response_length = categorize_response_length(text)
            return text, audio_url, response_length, None
            
        except Exception as e:
            # On any error, use fallback
            print(f"[{_now()}] [Smart] Error: {e}, using fallback")
            from services.llm import get_llm_service
            
            settings = get_settings()
            llm_service = get_llm_service()
            
            text = await llm_service.generate_response(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                model=settings.anthropic_fast_model,
            )
            audio_url = await self._synthesize_with_elevenlabs(text)
            response_length = categorize_response_length(text)
            
            return text, audio_url, response_length, None


# Global instance
gemini_text_speech_service = GeminiTextAndSpeechService()


def get_gemini_text_speech_service() -> GeminiTextAndSpeechService:
    """Get the global Gemini text+speech service instance."""
    return gemini_text_speech_service


async def synthesize_parallel_tts(text: str) -> tuple[str | None, str]:
    """
    Standalone parallel TTS function - runs ElevenLabs + Gemini simultaneously.
    
    Use this for sentence-by-sentence streaming where you want the fastest result.
    
    Strategy:
    - Start both ElevenLabs and Gemini TTS simultaneously
    - ElevenLabs is PRIMARY (faster, ~200-500ms)
    - Gemini is FALLBACK (slower, ~2-5s)
    - Use ElevenLabs if it succeeds, otherwise use Gemini
    
    Args:
        text: Text to synthesize
    
    Returns:
        (audio_url, source) - Audio data URL and which engine was used ("elevenlabs" or "gemini")
    """
    service = get_gemini_text_speech_service()
    return await service._synthesize_parallel_tts(text)


# Export response length utilities
__all__ = [
    "GeminiTextAndSpeechService",
    "get_gemini_text_speech_service",
    "synthesize_parallel_tts",
    "ResponseLength",
    "count_words",
    "categorize_response_length",
    "RateLimitError",
]

