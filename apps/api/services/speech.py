"""Gemini 2.5 Flash Preview TTS service - Optimized for low latency."""

import asyncio
import base64
import hashlib
import io
import re
import time
import wave
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from google import genai
from google.genai import types

from config.settings import get_settings


def _ms(start: float) -> int:
    """Return elapsed milliseconds since start time."""
    return int((time.perf_counter() - start) * 1000)


class LRUCache:
    """Simple LRU cache for audio data."""
    
    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size
    
    def get(self, key: str) -> str | None:
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                # Remove oldest item
                self._cache.popitem(last=False)
            self._cache[key] = value


class SpeechService:
    """Text-to-Speech service using Gemini 2.5 Flash Preview TTS.
    
    Optimized for low latency:
    - Persistent client connection
    - LRU cache for repeated phrases
    - Parallel sentence synthesis
    - ThreadPool for non-blocking I/O
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: genai.Client | None = None
        self._executor = ThreadPoolExecutor(max_workers=6)  # More workers for parallelism
        self._cache = LRUCache(max_size=200)  # Cache common phrases
        
        # Default voice from settings
        self._default_voice = self._settings.gemini_tts_voice
        self._model = "gemini-2.5-flash-preview-tts"
        
        # Pre-initialize client for faster first request
        self._init_client()

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

    def _synthesize_sync(
        self,
        text: str,
        voice_name: str,
        max_retries: int = 3,
    ) -> bytes:
        """Synchronous speech synthesis (runs in executor) with retries."""
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
        
        last_error: Exception | None = None
        
        for attempt in range(1, max_retries + 1):
            api_start = time.perf_counter()
            try:
                # Generate audio
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=speech_config,
                    ),
                )
                api_time = _ms(api_start)
                
                # Extract audio data with error handling
                if not response.candidates:
                    raise RuntimeError(f"No candidates returned")
                
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    raise RuntimeError(f"Empty content returned")
                
                part = candidate.content.parts[0]
                if not hasattr(part, 'inline_data') or not part.inline_data:
                    raise RuntimeError(f"No audio data in response")
                
                audio_data = part.inline_data.data
                
                if not audio_data:
                    raise RuntimeError(f"Audio data is empty")
                
                # Convert raw PCM to WAV (24kHz, 16-bit, mono)
                wav_start = time.perf_counter()
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(24000)
                    wav_file.writeframes(audio_data)
                wav_time = _ms(wav_start)
                
                result = wav_buffer.getvalue()
                
                # Log timing on success
                if attempt > 1:
                    print(f"[Gemini TTS] ✓ Retry {attempt} succeeded: {prompt[:25]}... (api={api_time}ms, wav={wav_time}ms)")
                
                return result
                
            except Exception as e:
                last_error = e
                elapsed = _ms(api_start)
                
                if attempt < max_retries:
                    # Brief backoff: 100ms, 200ms
                    backoff_ms = attempt * 100
                    print(f"[Gemini TTS] ⚠ Attempt {attempt}/{max_retries} failed ({elapsed}ms): {e} - retrying in {backoff_ms}ms...")
                    time.sleep(backoff_ms / 1000)
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
        Synthesize text to speech with caching.
        
        Returns audio as base64 data URL for instant playback.
        """
        total_start = time.perf_counter()
        voice_name = voice_id or self._default_voice
        
        # Check cache first
        cache_key = self._get_cache_key(text, voice_name, emotion)
        cached_url = self._cache.get(cache_key)
        if cached_url:
            print(f"[Gemini TTS] Cache hit ({_ms(total_start)}ms): {text[:30]}...")
            # Decode cached URL to get bytes
            audio_base64 = cached_url.split(",")[1]
            audio_data = base64.b64decode(audio_base64)
            return audio_data, cached_url
        
        print(f"[Gemini TTS] Synthesizing: {text[:40]}..., Voice: {voice_name}")
        
        # Run sync synthesis in executor
        loop = asyncio.get_running_loop()
        synth_start = time.perf_counter()
        audio_data = await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
            voice_name,
        )
        synth_time = _ms(synth_start)
        
        # Create data URL
        encode_start = time.perf_counter()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        audio_url = f"data:audio/wav;base64,{audio_base64}"
        encode_time = _ms(encode_start)
        
        # Cache the result
        self._cache.set(cache_key, audio_url)
        
        total_time = _ms(total_start)
        chars = len(text)
        bytes_size = len(audio_data)
        print(f"[Gemini TTS] ✓ {bytes_size:,}B in {total_time}ms (synth={synth_time}ms, encode={encode_time}ms) | {chars} chars")
        
        return audio_data, audio_url

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
        Synthesize text sentence by sentence for streaming playback.
        
        Optimized strategy:
        1. First sentence: Synthesize and yield immediately (lowest latency)
        2. Remaining: Synthesize ALL in parallel while first plays
        """
        stream_start = time.perf_counter()
        
        # Clean text first - remove bullet points, normalize whitespace
        clean_text = self._clean_text_for_tts(text)
        
        # Split into sentences (Japanese + English punctuation)
        sentences = re.split(r'(?<=[。！？!?])', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        
        if not sentences:
            # If no sentences after splitting, use whole cleaned text
            sentences = [clean_text] if clean_text else []
        
        if not sentences:
            print("[Gemini TTS] No valid text to synthesize")
            return
        
        voice_name = voice_id or self._default_voice
        total = len(sentences)
        
        print(f"[Gemini TTS] Streaming {total} sentences...")
        
        async def synth_one(sentence: str, idx: int) -> tuple[int, str, str | None, int]:
            """Synthesize one sentence, return (index, sentence, audio_url, elapsed_ms)."""
            start = time.perf_counter()
            try:
                _, audio_url = await self.synthesize_speech(
                    text=sentence,
                    voice_id=voice_name,
                    emotion=emotion,
                )
                return idx, sentence, audio_url, _ms(start)
            except Exception as e:
                elapsed = _ms(start)
                print(f"[Gemini TTS] ✗ Sentence {idx} failed ({elapsed}ms): {e}")
                return idx, sentence, None, elapsed
        
        # Optimized strategy: Start ALL sentences in parallel immediately
        # This gives remaining sentences a head start while first synthesizes
        
        first = sentences[0]
        remaining = sentences[1:]
        
        # Start background tasks for remaining sentences BEFORE waiting for first
        background_futures: list[asyncio.Task] = []
        if remaining:
            background_futures = [
                asyncio.create_task(synth_one(s, i + 2)) 
                for i, s in enumerate(remaining)
            ]
        
        # 1. Synthesize and yield first sentence
        first_start = time.perf_counter()
        print(f"[Gemini TTS] First sentence starting (others in background)...")
        
        first_url = None
        try:
            _, first_url = await self.synthesize_speech(
                text=first,
                voice_id=voice_name,
                emotion=emotion,
            )
        except Exception as e:
            print(f"[Gemini TTS] ✗ First sentence failed ({_ms(first_start)}ms): {e}")
        
        success_count = 0
        fail_count = 0
        
        if first_url:
            success_count += 1
            first_latency = _ms(first_start)
            print(f"[Gemini TTS] 1/{total} ready (first latency: {first_latency}ms)")
            yield first, first_url
        else:
            fail_count += 1
        
        if total == 1:
            total_time = _ms(stream_start)
            print(f"[Gemini TTS] Stream complete: {success_count}/{total} ok in {total_time}ms")
            return
        
        # 2. Wait for background tasks (they've been running while first synthesized)
        results = await asyncio.gather(*background_futures)
        
        # Yield in order
        for idx, sentence, audio_url, elapsed in sorted(results, key=lambda x: x[0]):
            if audio_url:
                success_count += 1
                print(f"[Gemini TTS] {idx}/{total} ready ({elapsed}ms)")
                yield sentence, audio_url
            else:
                fail_count += 1
        
        total_time = _ms(stream_start)
        print(f"[Gemini TTS] Stream complete: {success_count}/{total} ok, {fail_count} failed in {total_time}ms")

    async def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self._client = None
        self._cache = LRUCache(max_size=200)


# Global instance (eagerly initialized)
speech_service = SpeechService()


def get_speech_service() -> SpeechService:
    """Get speech service instance."""
    return speech_service
