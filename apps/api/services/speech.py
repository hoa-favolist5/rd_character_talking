"""ElevenLabs TTS Service - Primary TTS for streaming audio.

Simple, fast TTS service using ElevenLabs API.
No fallbacks - ElevenLabs is reliable and fast (~200-500ms).

For streaming generation with token-level TTS, use speech_streaming.py instead.
"""

import asyncio
import base64
import hashlib
import re
import time
from collections import OrderedDict
from typing import AsyncGenerator

from config.settings import get_settings
from services.speech_elevenlabs import get_elevenlabs_service


def _ms(start: float) -> int:
    """Return elapsed milliseconds since start time."""
    return int((time.perf_counter() - start) * 1000)


def _now() -> str:
    """Return current time as HH:MM:SS.mmm string."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


class LRUCache:
    """Simple LRU cache for audio data."""
    
    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> str | None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value


class SpeechService:
    """Text-to-Speech service using ElevenLabs.
    
    Features:
    - LRU cache for repeated phrases
    - Semaphore for rate limiting
    - Sentence-level streaming for long text
    """

    REQUEST_TIMEOUT = 10.0
    MAX_CONCURRENT = 5

    def __init__(self) -> None:
        self._settings = get_settings()
        self._elevenlabs = get_elevenlabs_service()
        self._cache = LRUCache(max_size=200)
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore (must be in async context)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        return self._semaphore

    def _get_cache_key(self, text: str, emotion: str) -> str:
        """Generate cache key for text+emotion combination."""
        content = f"{text}:{emotion}"
        return hashlib.md5(content.encode()).hexdigest()

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS - remove formatting that causes issues."""
        text = re.sub(r'^[\s]*[-•*]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech with caching.
        
        Returns audio as base64 data URL for instant playback.
        """
        total_start = time.perf_counter()
        
        # Clean text
        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            raise ValueError("Empty text after cleaning")
        
        # Check cache first
        cache_key = self._get_cache_key(clean_text, emotion)
        cached_url = await self._cache.get(cache_key)
        if cached_url:
            print(f"[{_now()}] [TTS] Cache hit ({_ms(total_start)}ms): {clean_text[:30]}...")
            audio_base64 = cached_url.split(",")[1]
            audio_data = base64.b64decode(audio_base64)
            return audio_data, cached_url
        
        print(f"[{_now()}] [TTS] Synthesizing: {clean_text[:40]}...")
        
        # Use semaphore to limit concurrent requests
        async with self._get_semaphore():
            audio_data, audio_url = await self._elevenlabs.synthesize_speech(
                text=clean_text,
                voice_id=voice_id,
            )
        
        # Cache the result
        await self._cache.set(cache_key, audio_url)
        
        total_time = _ms(total_start)
        print(f"[{_now()}] [TTS] ✓ {len(audio_data):,}B in {total_time}ms | {len(clean_text)} chars")
        
        return audio_data, audio_url

    async def synthesize_sentences(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Synthesize text sentence by sentence for streaming playback.
        
        Yields (sentence, audio_url) tuples as they're ready.
        """
        stream_start = time.perf_counter()
        
        # Clean text
        clean_text = self._clean_text_for_tts(text)
        if not clean_text:
            return
        
        # Split into sentences
        sentences = re.split(r'(?<=[。！？!?])', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        
        if not sentences:
            sentences = [clean_text]
        
        total = len(sentences)
        print(f"[{_now()}] [TTS Stream] {total} sentences")
        
        # Synthesize first sentence immediately
        first_start = time.perf_counter()
        try:
            _, first_url = await self._elevenlabs.synthesize_speech(
                sentences[0], voice_id=voice_id
            )
            first_latency = _ms(first_start)
            print(f"[{_now()}] [TTS] 1/{total} ready ({first_latency}ms)")
            yield sentences[0], first_url
        except Exception as e:
            print(f"[{_now()}] [TTS] ✗ First sentence failed: {e}")
            return
        
        if total == 1:
            return
        
        # Synthesize remaining sentences in parallel
        remaining = sentences[1:]
        
        async def synth_one(sentence: str, idx: int) -> tuple[int, str, str | None, int]:
            start = time.perf_counter()
            try:
                _, audio_url = await self._elevenlabs.synthesize_speech(
                    sentence, voice_id=voice_id
                )
                return idx, sentence, audio_url, _ms(start)
            except Exception as e:
                print(f"[{_now()}] [TTS] ✗ Sentence {idx} failed: {e}")
                return idx, sentence, None, _ms(start)
        
        # Start all remaining sentences in parallel
        tasks = [
            asyncio.create_task(synth_one(s, i + 2))
            for i, s in enumerate(remaining)
        ]
        
        results: dict[int, tuple[str, str | None, int]] = {}
        next_to_yield = 2
        
        # Yield results in order as they complete
        for completed in asyncio.as_completed(tasks):
            idx, sentence, audio_url, elapsed = await completed
            results[idx] = (sentence, audio_url, elapsed)
            
            while next_to_yield in results:
                sentence, audio_url, elapsed = results.pop(next_to_yield)
                if audio_url:
                    print(f"[{_now()}] [TTS] {next_to_yield}/{total} ready ({elapsed}ms)")
                    yield sentence, audio_url
                next_to_yield += 1
        
        total_time = _ms(stream_start)
        print(f"[{_now()}] [TTS Stream] ✓ Complete: {total} sentences in {total_time}ms")

    async def close(self) -> None:
        """Clean up resources."""
        await self._elevenlabs.close()
        self._cache = LRUCache(max_size=200)
        self._semaphore = None


# Global instance
speech_service = SpeechService()


def get_speech_service() -> SpeechService:
    """Get speech service instance."""
    return speech_service
