"""Gemini 2.5 Flash Preview TTS service - Optimized for low latency.

Falls back to Google Cloud TTS when Gemini quota is exhausted.

Note: This service is used for streaming TTS. For parallel TTS with ElevenLabs,
see speech_gemini.py which uses ElevenLabs as primary and Gemini as fallback.
"""

import asyncio
import base64
import hashlib
import io
import re
import time
import wave
from collections import OrderedDict
from typing import AsyncGenerator

from google import genai
from google.genai import types

from config.settings import get_settings


class RateLimitError(Exception):
    """Raised when API rate limit is hit."""
    pass


def _ms(start: float) -> int:
    """Return elapsed milliseconds since start time."""
    return int((time.perf_counter() - start) * 1000)


class LRUCache:
    """Simple LRU cache for audio data."""
    
    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> str | None:
        async with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    # Remove oldest item
                    self._cache.popitem(last=False)
                self._cache[key] = value
    
    def get_sync(self, key: str) -> str | None:
        """Sync version for non-async contexts."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def set_sync(self, key: str, value: str) -> None:
        """Sync version for non-async contexts."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value


class SpeechService:
    """Text-to-Speech service using Gemini 2.5 Flash Preview TTS.
    
    Optimized for low latency:
    - Native async API calls (no thread pool)
    - LRU cache for repeated phrases
    - Semaphore to prevent rate limiting
    - Request timeout to avoid hung requests
    - Single TTS call for small responses (avoids 429 rate limits)
    """

    # Timeout for individual TTS requests (seconds)
    REQUEST_TIMEOUT = 10.0
    # Max concurrent requests to avoid rate limiting
    MAX_CONCURRENT = 5
    # Threshold for splitting into sentences (chars) - below this, use single TTS call
    SMALL_TEXT_THRESHOLD = 100

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: genai.Client | None = None
        self._cache = LRUCache(max_size=200)  # Cache common phrases
        self._semaphore: asyncio.Semaphore | None = None
        
        # Default voice from settings
        self._default_voice = self._settings.gemini_tts_voice
        self._model = "gemini-2.5-flash-preview-tts"
        
        # Pre-initialize client for faster first request
        self._init_client()

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore (must be in async context)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        return self._semaphore

    def _init_client(self) -> None:
        """Initialize Gemini client eagerly."""
        if self._client is None and self._settings.google_api_key:
            try:
                self._client = genai.Client(api_key=self._settings.google_api_key)
            except Exception as e:
                print(f"[Gemini TTS] Failed to init client: {e}")

    def _get_client(self) -> genai.Client:
        """Get Gemini client (creates if needed)."""
        if self._client is None:
            self._client = genai.Client(api_key=self._settings.google_api_key)
        return self._client

    def _get_cache_key(self, text: str, voice: str, emotion: str) -> str:
        """Generate cache key for text+voice+emotion combination."""
        content = f"{text}:{voice}:{emotion}"
        return hashlib.md5(content.encode()).hexdigest()

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS - remove formatting that causes issues."""
        # Remove bullet points and list markers
        text = re.sub(r'^[\s]*[-•*]\s*', '', text, flags=re.MULTILINE)
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()

    def _pcm_to_wav(self, audio_data: bytes) -> bytes:
        """Convert raw PCM to WAV (24kHz, 16-bit, mono)."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)
        return wav_buffer.getvalue()

    async def _synthesize_async(
        self,
        text: str,
        voice_name: str,
        max_retries: int = 2,
    ) -> bytes:
        """Async speech synthesis with timeout and retries."""
        client = self._get_client()
        
        # Clean text - remove formatting that causes TTS issues
        prompt = self._clean_text_for_tts(text)
        
        if not prompt:
            raise ValueError("Empty text after cleaning")
        
        # Configure speech settings
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name,
                )
            )
        )
        
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        )
        
        last_error: Exception | None = None
        
        for attempt in range(1, max_retries + 1):
            api_start = time.perf_counter()
            try:
                # Use async API with timeout
                async with asyncio.timeout(self.REQUEST_TIMEOUT):
                    # Use native async API
                    response = await client.aio.models.generate_content(
                        model=self._model,
                        contents=prompt,
                        config=config,
                    )
                api_time = _ms(api_start)
                
                # Extract audio data with error handling
                if not response.candidates:
                    raise RuntimeError("No candidates returned")
                
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    raise RuntimeError("Empty content returned")
                
                part = candidate.content.parts[0]
                if not hasattr(part, 'inline_data') or not part.inline_data:
                    raise RuntimeError("No audio data in response")
                
                audio_data = part.inline_data.data
                
                if not audio_data:
                    raise RuntimeError("Audio data is empty")
                
                # Convert raw PCM to WAV
                result = self._pcm_to_wav(audio_data)
                
                # Log timing on success
                if attempt > 1:
                    print(f"[Gemini TTS] ✓ Retry {attempt} succeeded: {prompt[:25]}... ({api_time}ms)")
                
                return result
                
            except asyncio.TimeoutError:
                elapsed = _ms(api_start)
                last_error = TimeoutError(f"Request timed out after {self.REQUEST_TIMEOUT}s")
                print(f"[Gemini TTS] ⚠ Attempt {attempt}/{max_retries} timeout ({elapsed}ms): {prompt[:25]}...")
                
            except Exception as e:
                last_error = e
                elapsed = _ms(api_start)
                error_str = str(e)
                
                # Check for rate limit / quota errors - don't retry, fail fast to fallback
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    print(f"[Gemini TTS] ⚠ Rate limit hit ({elapsed}ms): {prompt[:25]}... - switching to fallback")
                    raise RateLimitError(f"Gemini quota exhausted: {e}") from e
                
                if attempt < max_retries:
                    # Brief async backoff: 50ms
                    print(f"[Gemini TTS] ⚠ Attempt {attempt}/{max_retries} failed ({elapsed}ms): {e} - retrying...")
                    await asyncio.sleep(0.05)
                else:
                    print(f"[Gemini TTS] ✗ All {max_retries} attempts failed ({elapsed}ms): {prompt[:30]}... Error: {e}")
        
        # All retries exhausted
        raise RuntimeError(f"TTS failed after {max_retries} attempts for: {prompt[:50]}...") from last_error

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech with caching and concurrency control.
        
        Falls back to Google Cloud TTS when Gemini quota is exhausted.
        
        Returns audio as base64 data URL for instant playback.
        """
        total_start = time.perf_counter()
        voice_name = voice_id or self._default_voice
        
        # Check cache first
        cache_key = self._get_cache_key(text, voice_name, emotion)
        cached_url = await self._cache.get(cache_key)
        if cached_url:
            print(f"[Gemini TTS] Cache hit ({_ms(total_start)}ms): {text[:30]}...")
            # Decode cached URL to get bytes
            audio_base64 = cached_url.split(",")[1]
            audio_data = base64.b64decode(audio_base64)
            return audio_data, cached_url
        
        print(f"[Gemini TTS] Synthesizing: {text[:40]}..., Voice: {voice_name}")
        
        try:
            # Use semaphore to limit concurrent requests (prevents rate limiting)
            async with self._get_semaphore():
                audio_data = await self._synthesize_async(text, voice_name)
            
            # Create data URL
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            audio_url = f"data:audio/wav;base64,{audio_base64}"
            
            # Cache the result
            await self._cache.set(cache_key, audio_url)
            
            total_time = _ms(total_start)
            chars = len(text)
            bytes_size = len(audio_data)
            print(f"[Gemini TTS] ✓ {bytes_size:,}B in {total_time}ms | {chars} chars")
            
            return audio_data, audio_url
            
        except RateLimitError:
            # Fallback to Google Cloud TTS
            return await self._fallback_to_cloud_tts(text, cache_key, total_start)
    
    async def _fallback_to_cloud_tts(
        self,
        text: str,
        cache_key: str,
        start_time: float,
    ) -> tuple[bytes, str]:
        """Fallback to Google Cloud TTS when Gemini is rate-limited."""
        from services.speech_fast import get_fast_speech_service
        
        print(f"[Fallback] Using Cloud TTS for: {text[:40]}...")
        
        try:
            fast_service = get_fast_speech_service()
            audio_data, audio_url = await fast_service.synthesize_speech(text=text)
            
            # Cache the result
            await self._cache.set(cache_key, audio_url)
            
            total_time = _ms(start_time)
            print(f"[Fallback] ✓ Cloud TTS {len(audio_data):,}B in {total_time}ms | {len(text)} chars")
            
            return audio_data, audio_url
            
        except Exception as e:
            print(f"[Fallback] ✗ Cloud TTS also failed: {e}")
            raise RuntimeError(f"Both Gemini and Cloud TTS failed for: {text[:50]}...") from e

    async def synthesize_speech_stream(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio in chunks."""
        audio_data, _ = await self.synthesize_speech(
            text=text,
            voice_id=voice_id,
            emotion=emotion,
            content_type=content_type,
        )
        
        # Yield in chunks
        chunk_size = 8192  # Larger chunks for efficiency
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]

    async def synthesize_sentences(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Synthesize text for streaming playback.
        
        Optimized strategy for voice consistency:
        - Try Gemini TTS on FULL text first (consistent prosody across sentences)
        - If Gemini succeeds → return single audio (best quality)
        - If Gemini fails/slow → fall back to ElevenLabs sentence-by-sentence
        """
        stream_start = time.perf_counter()
        
        # Clean text first - remove bullet points, normalize whitespace
        clean_text = self._clean_text_for_tts(text)
        
        if not clean_text:
            print("[TTS] No valid text to synthesize")
            return
        
        voice_name = voice_id or self._default_voice
        
        # For small responses, use parallel TTS (ElevenLabs + Gemini)
        if len(clean_text) <= self.SMALL_TEXT_THRESHOLD:
            print(f"[Parallel TTS] Small text ({len(clean_text)} chars) - single parallel call")
            try:
                from services.speech_gemini import synthesize_parallel_tts
                audio_url, source = await synthesize_parallel_tts(clean_text)
                total_time = _ms(stream_start)
                if audio_url:
                    print(f"[Parallel TTS] ✓ Single call complete in {total_time}ms ({source})")
                    yield clean_text, audio_url
                else:
                    print(f"[Parallel TTS] ✗ Both TTS failed ({total_time}ms)")
                return
            except Exception as e:
                print(f"[Parallel TTS] ✗ Single call failed ({_ms(stream_start)}ms): {e}")
                return
        
        # For large text: Try Gemini on FULL text first (better voice consistency)
        print(f"[TTS] Large text ({len(clean_text)} chars) - trying Gemini full-text first")
        
        from services.speech_gemini import synthesize_parallel_tts
        from services.speech_elevenlabs import get_elevenlabs_service
        
        # Try Gemini on full text with timeout
        gemini_timeout = 10.0  # Max wait for full-text Gemini
        gemini_start = time.perf_counter()
        
        try:
            # Run Gemini TTS on full text
            audio_data, _, _ = await asyncio.wait_for(
                self.synthesize_speech(text=clean_text, voice_id=voice_name, emotion=emotion),
                timeout=gemini_timeout,
            )
            
            gemini_time = _ms(gemini_start)
            if audio_data:
                total_time = _ms(stream_start)
                print(f"[TTS] ✓ Gemini full-text success in {gemini_time}ms (consistent voice)")
                yield clean_text, audio_data
                return
        except asyncio.TimeoutError:
            print(f"[TTS] Gemini full-text timeout ({gemini_timeout}s) → ElevenLabs fallback")
        except Exception as e:
            print(f"[TTS] Gemini full-text failed ({_ms(gemini_start)}ms): {e} → ElevenLabs fallback")
        
        # Fallback: ElevenLabs sentence-by-sentence (faster, but less consistent prosody)
        sentences = re.split(r'(?<=[。！？!?])', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        
        if not sentences:
            sentences = [clean_text]
        
        total = len(sentences)
        print(f"[ElevenLabs] Fallback: {total} sentences")
        
        elevenlabs = get_elevenlabs_service()
        
        # Synthesize first sentence with ElevenLabs
        first_start = time.perf_counter()
        
        try:
            _, first_url = await elevenlabs.synthesize_speech(sentences[0])
            first_latency = _ms(first_start)
            if first_url:
                print(f"[ElevenLabs] 1/{total} ready ({first_latency}ms)")
                yield sentences[0], first_url
            else:
                print(f"[ElevenLabs] ✗ First sentence failed ({first_latency}ms)")
        except Exception as e:
            print(f"[ElevenLabs] ✗ First sentence exception ({_ms(first_start)}ms): {e}")
        
        if total == 1:
            total_time = _ms(stream_start)
            print(f"[ElevenLabs] Stream complete: 1/1 in {total_time}ms")
            return
        
        remaining = sentences[1:]
        
        # Continue with ElevenLabs for remaining sentences (parallel)
        async def synth_one(sentence: str, idx: int) -> tuple[int, str, str | None, int]:
            """Synthesize one sentence with ElevenLabs, return (index, sentence, audio_url, elapsed_ms)."""
            start = time.perf_counter()
            try:
                _, audio_url = await elevenlabs.synthesize_speech(sentence)
                return idx, sentence, audio_url, _ms(start)
            except Exception as e:
                elapsed = _ms(start)
                print(f"[ElevenLabs] ✗ Sentence {idx} failed ({elapsed}ms): {e}")
                return idx, sentence, None, elapsed
        
        # Start remaining sentences in parallel
        tasks = [
            asyncio.create_task(synth_one(s, i + 2))  # +2 because first is already done
            for i, s in enumerate(remaining)
        ]
        
        success_count = 1  # First already succeeded
        fail_count = 0
        results: dict[int, tuple[str, str | None, int]] = {}
        next_to_yield = 2  # Start from second
        
        # Use as_completed to yield results as soon as each finishes
        for completed in asyncio.as_completed(tasks):
            idx, sentence, audio_url, elapsed = await completed
            results[idx] = (sentence, audio_url, elapsed)
            
            # Yield all ready results in order
            while next_to_yield in results:
                sentence, audio_url, elapsed = results.pop(next_to_yield)
                if audio_url:
                    success_count += 1
                    print(f"[ElevenLabs] {next_to_yield}/{total} ready ({elapsed}ms)")
                    yield sentence, audio_url
                else:
                    fail_count += 1
                next_to_yield += 1
        
        total_time = _ms(stream_start)
        print(f"[ElevenLabs] Fallback complete: {success_count}/{total} ok, {fail_count} failed in {total_time}ms")

    async def close(self) -> None:
        """Clean up resources."""
        self._client = None
        self._cache = LRUCache(max_size=200)
        self._semaphore = None


# Global instance (eagerly initialized)
speech_service = SpeechService()


def get_speech_service() -> SpeechService:
    """Get speech service instance."""
    return speech_service
