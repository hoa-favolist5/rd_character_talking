"""Gemini 2.5 Flash Preview TTS service - Optimized for low latency."""

import asyncio
import base64
import hashlib
import io
import re
import wave
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from google import genai
from google.genai import types

from config.settings import get_settings


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
    ) -> bytes:
        """Synchronous speech synthesis (runs in executor)."""
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
        
        # Generate audio
        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            ),
        )
        
        # Extract audio data with error handling
        if not response.candidates:
            raise RuntimeError(f"Gemini TTS returned no candidates for: {prompt[:50]}...")
        
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            raise RuntimeError(f"Gemini TTS returned empty content for: {prompt[:50]}...")
        
        part = candidate.content.parts[0]
        if not hasattr(part, 'inline_data') or not part.inline_data:
            raise RuntimeError(f"Gemini TTS returned no audio data for: {prompt[:50]}...")
        
        audio_data = part.inline_data.data
        
        # Convert raw PCM to WAV (24kHz, 16-bit, mono)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)
        
        return wav_buffer.getvalue()

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
        voice_name = voice_id or self._default_voice
        
        # Check cache first
        cache_key = self._get_cache_key(text, voice_name, emotion)
        cached_url = self._cache.get(cache_key)
        if cached_url:
            print(f"[Gemini TTS] Cache hit: {text[:30]}...")
            # Decode cached URL to get bytes
            audio_base64 = cached_url.split(",")[1]
            audio_data = base64.b64decode(audio_base64)
            return audio_data, cached_url
        
        print(f"[Gemini TTS] Synthesizing: {text[:40]}..., Voice: {voice_name}")
        
        # Run sync synthesis in executor
        loop = asyncio.get_running_loop()
        audio_data = await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
            voice_name,
        )
        
        print(f"[Gemini TTS] Generated {len(audio_data)} bytes")
        
        # Create data URL
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        audio_url = f"data:audio/wav;base64,{audio_base64}"
        
        # Cache the result
        self._cache.set(cache_key, audio_url)
        
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
        
        async def synth_one(sentence: str, idx: int) -> tuple[int, str, str | None]:
            """Synthesize one sentence, return (index, sentence, audio_url)."""
            try:
                _, audio_url = await self.synthesize_speech(
                    text=sentence,
                    voice_id=voice_name,
                    emotion=emotion,
                )
                return idx, sentence, audio_url
            except Exception as e:
                print(f"[Gemini TTS] Error on sentence {idx}: {e}")
                return idx, sentence, None
        
        # Optimized strategy: Start ALL sentences in parallel immediately
        # This gives remaining sentences a head start while first synthesizes
        
        first = sentences[0]
        remaining = sentences[1:]
        
        # Start background tasks for remaining sentences BEFORE waiting for first
        background_tasks = None
        if remaining:
            background_tasks = [synth_one(s, i + 2) for i, s in enumerate(remaining)]
            # Create tasks but don't await yet - they run in background
            background_futures = [asyncio.create_task(t) for t in background_tasks]
        
        # 1. Synthesize and yield first sentence
        print(f"[Gemini TTS] First sentence starting (others in background)...")
        
        _, first_url = await self.synthesize_speech(
            text=first,
            voice_id=voice_name,
            emotion=emotion,
        )
        
        if first_url:
            print(f"[Gemini TTS] 1/{total} ready")
            yield first, first_url
        
        if total == 1:
            return
        
        # 2. Wait for background tasks (they've been running while first synthesized)
        results = await asyncio.gather(*background_futures)
        
        # Yield in order
        for idx, sentence, audio_url in sorted(results, key=lambda x: x[0]):
            if audio_url:
                print(f"[Gemini TTS] {idx}/{total} ready")
                yield sentence, audio_url

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
