"""ElevenLabs TTS service - High-quality cloud TTS.

ElevenLabs provides high-quality, low-latency text-to-speech synthesis.
This is the PRIMARY (and only) TTS service.

Typical response time: ~200-500ms (fast cloud service)
"""

import asyncio
import base64
import time
from typing import Any

import httpx

from config.settings import get_settings


def _ms(start: float) -> int:
    """Return elapsed milliseconds since start time."""
    return int((time.perf_counter() - start) * 1000)


def _now() -> str:
    """Return current time as HH:MM:SS.mmm string."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


class ElevenLabsService:
    """Text-to-Speech service using ElevenLabs API.
    
    ElevenLabs provides high-quality, natural-sounding speech synthesis.
    Supports multiple voices and languages including Japanese.
    
    Optimizations:
    - Uses httpx async client for non-blocking requests
    - Connection pooling for faster repeated requests
    - Configurable voice and model settings
    - Automatic retry with backoff for rate limit (429) errors
    """

    # Default timeout for ElevenLabs requests (seconds)
    REQUEST_TIMEOUT = 10.0
    
    # API base URL
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    # Retry configuration for 429 errors
    MAX_RETRIES = 3
    INITIAL_BACKOFF_MS = 500  # Start with 500ms delay
    MAX_BACKOFF_MS = 4000     # Max 4 second delay

    def __init__(self) -> None:
        self._settings = get_settings()
        self._api_key = self._settings.elevenlabs_api_key
        self._voice_id = self._settings.elevenlabs_voice_id
        self._model_id = self._settings.elevenlabs_model_id
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(self.REQUEST_TIMEOUT),
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str | None = None,
        model_id: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech using ElevenLabs.
        
        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID (optional, uses default if not specified)
            model_id: ElevenLabs model ID (optional, uses default if not specified)
        
        Returns:
            (audio_bytes, data_url) - Audio data and base64 data URL
        """
        start = time.perf_counter()
        start_ts = _now()
        
        voice = voice_id or self._voice_id
        model = model_id or self._model_id
        
        if not self._api_key:
            raise RuntimeError("ElevenLabs API key not configured")
        
        # Retry loop with exponential backoff for 429 errors
        last_error: Exception | None = None
        backoff_ms = self.INITIAL_BACKOFF_MS
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                client = await self._get_client()
                
                # Request body for TTS
                request_body = {
                    "text": text,
                    "model_id": model,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True,
                    }
                }
                
                # Make TTS request
                response = await client.post(
                    f"/text-to-speech/{voice}",
                    json=request_body,
                    headers={"Accept": "audio/mpeg"},
                )
                response.raise_for_status()
                
                audio_data = response.content
                total_time = _ms(start)
                
                # Create data URL (ElevenLabs returns MP3 format)
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                audio_url = f"data:audio/mpeg;base64,{audio_base64}"
                
                retry_info = f" (retry {attempt})" if attempt > 0 else ""
                print(f"[{start_ts}] [ElevenLabs] ⚡ Voice {voice[:8]}...: {len(text)} chars → {len(audio_data):,}B{retry_info}")
                print(f"  └─ Total:     {total_time:>4}ms")
                
                return audio_data, audio_url
                
            except httpx.HTTPStatusError as e:
                # Check for rate limit - retry with backoff
                if e.response.status_code == 429 and attempt < self.MAX_RETRIES:
                    print(f"[{_now()}] [ElevenLabs] ⚠ Rate limit, retry {attempt + 1}/{self.MAX_RETRIES} in {backoff_ms}ms...")
                    await asyncio.sleep(backoff_ms / 1000)
                    backoff_ms = min(backoff_ms * 2, self.MAX_BACKOFF_MS)
                    last_error = e
                    continue
                
                elapsed = _ms(start)
                
                # Rate limit exhausted after retries
                if e.response.status_code == 429:
                    print(f"[{_now()}] [ElevenLabs] ✗ Rate limit exhausted ({elapsed}ms)")
                    raise RuntimeError(f"ElevenLabs rate limit: {e}") from e
                
                # Check for quota exceeded
                if e.response.status_code == 401:
                    print(f"[{_now()}] [ElevenLabs] ✗ Unauthorized ({elapsed}ms): Check API key")
                    raise RuntimeError(f"ElevenLabs unauthorized: {e}") from e
                    
                print(f"[{_now()}] [ElevenLabs] ✗ HTTP error ({elapsed}ms): {e}")
                raise RuntimeError(f"ElevenLabs error: {e}") from e
                
            except httpx.ConnectError as e:
                elapsed = _ms(start)
                print(f"[{_now()}] [ElevenLabs] ✗ Connection failed ({elapsed}ms): {e}")
                raise RuntimeError(f"ElevenLabs not available: {e}") from e
                
            except httpx.TimeoutException as e:
                elapsed = _ms(start)
                print(f"[{_now()}] [ElevenLabs] ✗ Timeout ({elapsed}ms): {e}")
                raise RuntimeError(f"ElevenLabs timeout: {e}") from e
                
            except Exception as e:
                elapsed = _ms(start)
                print(f"[{_now()}] [ElevenLabs] ✗ Error ({elapsed}ms): {e}")
                raise RuntimeError(f"ElevenLabs error: {e}") from e
        
        # Should not reach here, but just in case
        if last_error:
            raise RuntimeError(f"ElevenLabs failed after {self.MAX_RETRIES} retries") from last_error
        raise RuntimeError("ElevenLabs unexpected error")

    async def get_voices(self) -> list[dict[str, Any]]:
        """Get list of available voices from ElevenLabs.
        
        Returns:
            List of voice info dictionaries with voice_id, name, labels, etc.
        """
        try:
            client = await self._get_client()
            response = await client.get("/voices")
            response.raise_for_status()
            data = response.json()
            return data.get("voices", [])
        except Exception as e:
            print(f"[{_now()}] [ElevenLabs] ✗ Failed to get voices: {e}")
            return []

    async def is_available(self) -> bool:
        """Check if ElevenLabs API is available and responding."""
        if not self._api_key:
            return False
        try:
            client = await self._get_client()
            response = await client.get("/user", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Clean up resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# Global instance
elevenlabs_service = ElevenLabsService()


def get_elevenlabs_service() -> ElevenLabsService:
    """Get the global ElevenLabs service instance."""
    return elevenlabs_service


# ElevenLabs recommended voices for Japanese
# Full list available via GET /voices endpoint
#
# RECOMMENDED for Japanese:
#   - Kokoro (Japanese female, natural)
#   - Yuki (Japanese female, soft)
#   - Hana (Japanese female, expressive)
#
ELEVENLABS_JAPANESE_VOICES = {
    # These are example voice IDs - actual IDs from your ElevenLabs account may differ
    # Use the /voices endpoint to get your available voices
    "default": "pNInz6obpgDQGcFmaJgB",  # Adam - multilingual, clear
}
