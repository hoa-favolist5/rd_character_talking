"""Gemini Text + TTS service with VoiceVox fallback.

Two-step process:
1. Generate text with gemini-2.5-flash (fast, smart)  
2. Generate audio with gemini-2.5-flash-preview-tts (TTS only)

TTS Strategy (parallel execution):
- Run VoiceVox and Gemini TTS simultaneously
- If Gemini TTS completes within 4 seconds, use Gemini (higher quality)
- If Gemini TTS exceeds 4 seconds, use VoiceVox result (faster)

VoiceVox: ~100-300ms, local, consistent quality
Gemini TTS: ~2-5s, cloud, slightly more natural (when fast)

Note: gemini-2.5-flash-tts is NOT available yet. Using preview version.
The TTS model only supports AUDIO modality, not TEXT+AUDIO combined.
"""

import asyncio
import base64
import io
import re
import time
import wave
from enum import Enum
from typing import AsyncGenerator

from google import genai
from google.genai import types

from config.settings import get_settings


class ResponseLength(Enum):
    """Response length categories for TTS strategy selection."""
    SHORT = "short"      # < 50 words: Parallel TTS (prefer Gemini if fast)
    MEDIUM = "medium"    # 50-100 words: Waiting audio + Parallel TTS
    LONG = "long"        # > 100 words: VoiceVox (more consistent for long text)


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
    """Text generation + TTS using Gemini models with VoiceVox fallback.
    
    Two-step process:
    1. Generate text with gemini-2.5-flash (fast, smart)
    2. Generate audio with parallel TTS (Gemini + VoiceVox)
    
    Parallel TTS Strategy:
    - Start both Gemini TTS and VoiceVox simultaneously
    - Wait for Gemini up to VOICEVOX_TIMEOUT (4s default)
    - If Gemini completes in time, use Gemini (higher quality)
    - If Gemini exceeds timeout, use VoiceVox result (faster)
    
    Performance:
    - VoiceVox: ~100-300ms, local, very consistent
    - Gemini TTS: ~2-5s, cloud, slightly more natural when fast
    
    Optimizations:
    - Native async API with parallel execution
    - VoiceVox always runs as backup
    - Cached waiting audio for instant playback
    """

    REQUEST_TIMEOUT = 15.0  # For text generation
    TTS_TIMEOUT = 8.0       # Max wait for Gemini TTS (hard timeout)
    VOICEVOX_TIMEOUT = 2.0  # Prefer VoiceVox if Gemini slower than this
    MAX_RETRIES = 1         # No retry for TTS - use parallel fallback
    
    # Waiting phrases (randomly selected for variety)
    WAITING_PHRASES = [
        "ちょっと待ってね",
        "えーっと、ちょっと待って",
        "うーんと、待ってね",
        "少し待ってね",
    ]

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: genai.Client | None = None
        self._text_model = "gemini-2.5-flash"  # For text generation
        self._tts_model = "gemini-2.5-flash-preview-tts"  # For audio only (stable version not available yet)
        self._voice = self._settings.gemini_tts_voice
        
        # VoiceVox timeout from settings (default 4s)
        self.VOICEVOX_TIMEOUT = self._settings.voicevox_timeout
        
        # Cached waiting audio (generated on first use)
        self._waiting_audio_cache: dict[str, str] = {}
        
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

    async def _synthesize_with_voicevox(self, text: str) -> str | None:
        """Fallback TTS using VoiceVox (~100-300ms, local, fast and reliable)."""
        from services.speech_voicevox import get_voicevox_service
        
        try:
            voicevox = get_voicevox_service()
            _, audio_url = await voicevox.synthesize_speech(text)
            return audio_url
            
        except Exception as e:
            print(f"[{_now()}] [VoiceVox] ✗ Failed: {e}")
            return None

    async def _synthesize_parallel_tts(self, text: str) -> tuple[str | None, str]:
        """
        Run Gemini TTS and VoiceVox in parallel.
        
        Strategy:
        - Start both TTS engines simultaneously
        - If Gemini finishes within VOICEVOX_TIMEOUT (4s), use Gemini
        - Otherwise, use VoiceVox result
        
        Returns:
            (audio_url, source) - Audio data URL and which engine was used
        """
        from services.speech_voicevox import get_voicevox_service
        
        start = time.perf_counter()
        start_ts = _now()
        
        # Create tasks for both TTS engines
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
        
        async def voicevox_tts() -> tuple[str | None, int]:
            """Run VoiceVox TTS and return (audio_url, elapsed_ms)."""
            vv_start = time.perf_counter()
            try:
                voicevox = get_voicevox_service()
                _, audio_url = await voicevox.synthesize_speech(text)
                elapsed = _ms(vv_start)
                return audio_url, elapsed
            except Exception as e:
                elapsed = _ms(vv_start)
                print(f"[{_now()}] [Parallel TTS] VoiceVox error ({elapsed}ms): {e}")
                return None, elapsed
        
        # Run both in parallel
        gemini_task = asyncio.create_task(gemini_tts())
        voicevox_task = asyncio.create_task(voicevox_tts())
        
        print(f"[{start_ts}] [Parallel TTS] Starting Gemini + VoiceVox for: {text[:40]}...")
        
        # Wait for VoiceVox first (it's usually faster)
        voicevox_result: str | None = None
        voicevox_time: int = 0
        
        try:
            voicevox_result, voicevox_time = await voicevox_task
        except Exception as e:
            print(f"[{_now()}] [Parallel TTS] VoiceVox task failed: {e}")
        
        # Now wait for Gemini with timeout (VOICEVOX_TIMEOUT)
        gemini_result: bytes | None = None
        gemini_time: int = 0
        
        try:
            # Calculate remaining time after VoiceVox
            elapsed_so_far = _ms(start) / 1000.0
            remaining_timeout = max(0.1, self.VOICEVOX_TIMEOUT - elapsed_so_far)
            
            gemini_result, gemini_time = await asyncio.wait_for(
                gemini_task,
                timeout=remaining_timeout,
            )
        except asyncio.TimeoutError:
            gemini_time = _ms(start)
            print(f"[{_now()}] [Parallel TTS] Gemini timeout after {self.VOICEVOX_TIMEOUT}s ({gemini_time}ms)")
            # Cancel the gemini task if still running
            gemini_task.cancel()
            try:
                await gemini_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            gemini_time = _ms(start)
            print(f"[{_now()}] [Parallel TTS] Gemini task failed ({gemini_time}ms): {e}")
        
        total_time = _ms(start)
        
        # Decision: Use Gemini if it succeeded within timeout, otherwise VoiceVox
        if gemini_result is not None:
            # Gemini succeeded - convert to data URL
            wav_data = self._pcm_to_wav(gemini_result)
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            audio_url = f"data:audio/wav;base64,{audio_base64}"
            
            print(f"[{_now()}] [Parallel TTS] ✓ Using Gemini ({gemini_time}ms)")
            print(f"  ├─ Gemini:   {gemini_time:>4}ms ← SELECTED")
            print(f"  ├─ VoiceVox: {voicevox_time:>4}ms (backup ready)")
            print(f"  └─ Total:    {total_time:>4}ms")
            
            return audio_url, "gemini"
        
        elif voicevox_result is not None:
            # VoiceVox succeeded, Gemini failed or timed out
            print(f"[{_now()}] [Parallel TTS] ✓ Using VoiceVox ({voicevox_time}ms)")
            print(f"  ├─ Gemini:   {gemini_time:>4}ms (failed/timeout)")
            print(f"  ├─ VoiceVox: {voicevox_time:>4}ms ← SELECTED")
            print(f"  └─ Total:    {total_time:>4}ms")
            
            return voicevox_result, "voicevox"
        
        else:
            # Both failed
            print(f"[{_now()}] [Parallel TTS] ✗ Both TTS engines failed ({total_time}ms)")
            return None, "none"

    async def get_waiting_audio(self) -> tuple[str, str]:
        """Get cached waiting audio for medium-length responses.
        
        Returns:
            (phrase_text, audio_url) - The waiting phrase and its audio
        """
        import random
        
        # Select a random phrase
        phrase = random.choice(self.WAITING_PHRASES)
        
        # Check cache
        if phrase in self._waiting_audio_cache:
            print(f"[{_now()}] [Waiting] Cache hit: {phrase}")
            return phrase, self._waiting_audio_cache[phrase]
        
        # Generate with VoiceVox (fast and reliable)
        print(f"[{_now()}] [Waiting] Generating: {phrase}")
        audio_url = await self._synthesize_with_voicevox(phrase)
        
        if audio_url:
            self._waiting_audio_cache[phrase] = audio_url
        
        return phrase, audio_url or ""
    
    async def preload_waiting_audio(self) -> None:
        """Pre-generate all waiting audio on startup for instant playback."""
        print(f"[{_now()}] [Waiting] Pre-loading waiting audio with VoiceVox...")
        for phrase in self.WAITING_PHRASES:
            if phrase not in self._waiting_audio_cache:
                audio_url = await self._synthesize_with_voicevox(phrase)
                if audio_url:
                    self._waiting_audio_cache[phrase] = audio_url
        print(f"[{_now()}] [Waiting] Pre-loaded {len(self._waiting_audio_cache)} waiting phrases")

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
                print(f"[{_now()}] [Gemini TTS] ⚠ Rate limit → use Polly")
                raise RateLimitError(f"Gemini quota exhausted: {e}") from e
            
            # Any other error → use Polly
            print(f"[{_now()}] [Gemini TTS] ⚠ Error → use Polly")
            return None, elapsed, start_time

    async def generate_text_and_speech(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
    ) -> tuple[str, str | None]:
        """
        Generate text response AND audio in two steps.
        
        Step 1: Generate text with gemini-2.5-flash
        Step 2: Generate audio with parallel TTS (Gemini + VoiceVox)
               - If Gemini completes within 4s, use Gemini
               - Otherwise use VoiceVox result
        
        Args:
            messages: Conversation history [{"role": "user/assistant", "content": "..."}]
            system_prompt: System instruction for the character
            max_tokens: Maximum output tokens
        
        Returns:
            (response_text, audio_data_url) - audio_url from faster TTS engine
        """
        total_start = time.perf_counter()
        
        try:
            # Step 1: Generate text
            text_response, text_time, text_ts = await self._generate_text(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            
            # Step 2: Parallel TTS (Gemini + VoiceVox with 4s timeout)
            tts_ts = _now()
            tts_start = time.perf_counter()
            audio_url, tts_source = await self._synthesize_parallel_tts(text_response)
            tts_time = _ms(tts_start)
            
            total_time = _ms(total_start)
            end_ts = _now()
            
            # Print detailed time breakdown with timestamps
            print(f"[Gemini T+S] ✓ {len(text_response)} chars")
            print(f"  ├─ [{text_ts}] Text: {text_time:>4}ms (gemini-2.5-flash)")
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
                print(f"[{_now()}] [Gemini T+S] ⚠ Rate limit ({elapsed}ms) → use fallback")
                raise RateLimitError(f"Gemini quota exhausted: {e}") from e
            
            print(f"[{_now()}] [Gemini T+S] ❌ Error ({elapsed}ms): {e}")
            raise RuntimeError(f"Gemini T+S failed: {e}") from e

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
            # Fallback: Use Anthropic for text, VoiceVox for audio
            total_start = time.perf_counter()
            start_ts = _now()
            print(f"[{start_ts}] [Fallback] Using Anthropic + VoiceVox...")
            
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
            
            # Generate audio with VoiceVox (fast and reliable)
            voicevox_ts = _now()
            voicevox_start = time.perf_counter()
            audio_url = await self._synthesize_with_voicevox(text)
            voicevox_time = _ms(voicevox_start)
            
            total_time = _ms(total_start)
            end_ts = _now()
            
            # Print detailed time breakdown with timestamps
            print(f"[Fallback] ✓ {len(text)} chars")
            print(f"  ├─ [{text_ts}] Text:     {text_time:>4}ms (Anthropic {settings.anthropic_fast_model})")
            print(f"  ├─ [{voicevox_ts}] TTS:      {voicevox_time:>4}ms (VoiceVox)")
            print(f"  └─ [{end_ts}] Total:    {total_time:>4}ms")
            
            return text, audio_url, True

    async def generate_text_and_speech_smart(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 200,
    ) -> tuple[str, str | None, ResponseLength, str | None]:
        """
        Smart text+speech generation with response length strategy.
        
        Strategy:
        - SHORT (< 50 words): Parallel TTS (Gemini + VoiceVox with 4s timeout)
        - MEDIUM (50-100 words): Return waiting audio hint, then Parallel TTS
        - LONG (> 100 words): VoiceVox directly (more consistent for long text)
        
        Returns:
            (response_text, audio_url, response_length, waiting_audio_url)
            - waiting_audio_url is only set for MEDIUM responses
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
            
            # Step 3: Generate audio based on length
            waiting_audio_url = None
            
            if response_length == ResponseLength.LONG:
                # LONG: Use VoiceVox directly (more consistent for long text)
                print(f"[{_now()}] [Smart] Using VoiceVox for long response")
                audio_url = await self._synthesize_with_voicevox(text_response)
                
            elif response_length == ResponseLength.MEDIUM:
                # MEDIUM: Get waiting audio, then parallel TTS
                _, waiting_audio_url = await self.get_waiting_audio()
                
                # Generate audio with parallel TTS (Gemini + VoiceVox)
                audio_url, tts_source = await self._synthesize_parallel_tts(text_response)
                print(f"[{_now()}] [Smart] MEDIUM response used: {tts_source}")
                    
            else:
                # SHORT: Parallel TTS (Gemini + VoiceVox with 4s timeout)
                audio_url, tts_source = await self._synthesize_parallel_tts(text_response)
                print(f"[{_now()}] [Smart] SHORT response used: {tts_source}")
            
            total_time = _ms(total_start)
            print(f"[{_now()}] [Smart] ✓ {len(text_response)} chars, {word_count} words in {total_time}ms")
            
            return text_response, audio_url, response_length, waiting_audio_url
            
        except RateLimitError:
            # Fallback to Anthropic + VoiceVox
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
            audio_url = await self._synthesize_with_voicevox(text)
            response_length = categorize_response_length(text)
            
            return text, audio_url, response_length, None


# Global instance
gemini_text_speech_service = GeminiTextAndSpeechService()


def get_gemini_text_speech_service() -> GeminiTextAndSpeechService:
    """Get the global Gemini text+speech service instance."""
    return gemini_text_speech_service


# Export response length utilities
__all__ = [
    "GeminiTextAndSpeechService",
    "get_gemini_text_speech_service",
    "ResponseLength",
    "count_words",
    "categorize_response_length",
    "RateLimitError",
]

