"""Google Cloud TTS service - FAST alternative (~200ms vs 3-4s).

Use this when speed is priority over voice naturalness.
Gemini TTS: More natural, but ~3-4 seconds per sentence
Cloud TTS: Less natural, but ~200-500ms per sentence
"""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator
import re

from google.cloud import texttospeech

from config.settings import get_settings


class FastSpeechService:
    """Fast TTS using Google Cloud Text-to-Speech.
    
    ~10x faster than Gemini TTS but slightly less natural.
    """
    
    # Threshold for splitting into sentences (chars) - below this, use single TTS call
    SMALL_TEXT_THRESHOLD = 100

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: texttospeech.TextToSpeechClient | None = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Use Neural2 voice for best quality
        self._voice_name = "ja-JP-Neural2-D"  # Young male
        self._language_code = "ja-JP"

    def _get_client(self) -> texttospeech.TextToSpeechClient:
        """Get or create TTS client."""
        if self._client is None:
            self._client = texttospeech.TextToSpeechClient()
        return self._client

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous speech synthesis."""
        import time
        start = time.perf_counter()
        
        client = self._get_client()
        
        # Configure input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=self._language_code,
            name=self._voice_name,
        )
        
        # Configure audio (MP3 for smaller size)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.1,  # Slightly faster for kid character
            pitch=2.0,  # Higher pitch for young voice
        )
        
        # Synthesize
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        
        duration_ms = (time.perf_counter() - start) * 1000
        print(f"[Cloud TTS] ⚡ {len(text)} chars → {len(response.audio_content)} bytes ({duration_ms:.0f}ms)")
        
        return response.audio_content

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: str | None = None,
    ) -> tuple[bytes, str]:
        """Synthesize text to speech."""
        loop = asyncio.get_running_loop()
        audio_data = await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
        )
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        audio_url = f"data:audio/mp3;base64,{audio_base64}"
        
        return audio_data, audio_url

    async def synthesize_sentences(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Synthesize sentences - single call for small text, parallel for large."""
        clean_text = text.strip()
        
        if not clean_text:
            return
        
        # For small responses, use single TTS call
        if len(clean_text) <= self.SMALL_TEXT_THRESHOLD:
            print(f"[Cloud TTS] Small text ({len(clean_text)} chars) - single call")
            try:
                _, audio_url = await self.synthesize_speech(text=clean_text)
                yield clean_text, audio_url
                return
            except Exception as e:
                print(f"[Cloud TTS] ✗ Single call failed: {e}")
                return
        
        # For large text, split into sentences
        sentences = re.split(r'(?<=[。！？!?])', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
        
        if not sentences:
            sentences = [clean_text]
        
        total = len(sentences)
        print(f"[Cloud TTS] Large text ({len(clean_text)} chars) - {total} sentences (PARALLEL)")
        
        async def synth_one(sentence: str, idx: int) -> tuple[int, str, str | None]:
            try:
                _, audio_url = await self.synthesize_speech(text=sentence)
                return idx, sentence, audio_url
            except Exception as e:
                print(f"[Cloud TTS] Error on sentence {idx}: {e}")
                return idx, sentence, None
        
        # Synthesize ALL in parallel - Cloud TTS handles rate limits well
        tasks = [synth_one(s, i + 1) for i, s in enumerate(sentences)]
        results = await asyncio.gather(*tasks)
        
        # Yield in order
        for idx, sentence, audio_url in sorted(results, key=lambda x: x[0]):
            if audio_url:
                print(f"[Cloud TTS] ✓ {idx}/{total} ready")
                yield sentence, audio_url

    async def close(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=False)
        self._client = None


# Global instance
fast_speech_service = FastSpeechService()


def get_fast_speech_service() -> FastSpeechService:
    """Get fast speech service instance."""
    return fast_speech_service

