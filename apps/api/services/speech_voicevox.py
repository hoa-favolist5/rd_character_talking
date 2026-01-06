"""VoiceVox TTS service - Local high-quality Japanese TTS.

VoiceVox is a free, high-quality Japanese TTS engine that runs locally.
It provides natural Japanese speech synthesis with multiple speakers.

Typical response time: ~100-300ms (very fast)
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


class VoiceVoxService:
    """Text-to-Speech service using VoiceVox local engine.
    
    VoiceVox provides high-quality Japanese speech synthesis.
    The engine must be running locally (typically at localhost:50021).
    
    Workflow:
    1. POST /audio_query - Get audio query parameters for text + speaker
    2. POST /synthesis - Generate audio from query parameters
    
    Optimizations:
    - Uses httpx async client for non-blocking requests
    - Connection pooling for faster repeated requests
    - Configurable speaker ID for different voices
    """

    # Default timeout for VoiceVox requests
    REQUEST_TIMEOUT = 10.0

    def __init__(self) -> None:
        self._settings = get_settings()
        self._base_url = self._settings.voicevox_url
        self._speaker_id = self._settings.voicevox_speaker_id
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self.REQUEST_TIMEOUT),
            )
        return self._client

    async def synthesize_speech(
        self,
        text: str,
        speaker_id: int | None = None,
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech using VoiceVox.
        
        Args:
            text: Text to synthesize
            speaker_id: VoiceVox speaker ID (optional, uses default if not specified)
        
        Returns:
            (audio_bytes, data_url) - Audio data and base64 data URL
        """
        start = time.perf_counter()
        start_ts = _now()
        speaker = speaker_id or self._speaker_id
        
        try:
            client = await self._get_client()
            
            # Step 1: Get audio query
            query_start = time.perf_counter()
            query_response = await client.post(
                "/audio_query",
                params={"text": text, "speaker": speaker},
            )
            query_response.raise_for_status()
            audio_query = query_response.json()
            query_time = _ms(query_start)
            
            # Step 2: Synthesize audio
            synth_start = time.perf_counter()
            synth_response = await client.post(
                "/synthesis",
                params={"speaker": speaker},
                json=audio_query,
            )
            synth_response.raise_for_status()
            audio_data = synth_response.content
            synth_time = _ms(synth_start)
            
            # Create data URL (VoiceVox returns WAV format)
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            audio_url = f"data:audio/wav;base64,{audio_base64}"
            
            total_time = _ms(start)
            print(f"[{start_ts}] [VoiceVox] ⚡ Speaker {speaker}: {len(text)} chars → {len(audio_data):,}B")
            print(f"  ├─ Query:     {query_time:>4}ms")
            print(f"  ├─ Synthesis: {synth_time:>4}ms")
            print(f"  └─ Total:     {total_time:>4}ms")
            
            return audio_data, audio_url
            
        except httpx.ConnectError as e:
            elapsed = _ms(start)
            print(f"[{_now()}] [VoiceVox] ✗ Connection failed ({elapsed}ms): {e}")
            print(f"  └─ Make sure VoiceVox engine is running at {self._base_url}")
            raise RuntimeError(f"VoiceVox not available: {e}") from e
            
        except httpx.TimeoutException as e:
            elapsed = _ms(start)
            print(f"[{_now()}] [VoiceVox] ✗ Timeout ({elapsed}ms): {e}")
            raise RuntimeError(f"VoiceVox timeout: {e}") from e
            
        except Exception as e:
            elapsed = _ms(start)
            print(f"[{_now()}] [VoiceVox] ✗ Error ({elapsed}ms): {e}")
            raise RuntimeError(f"VoiceVox error: {e}") from e

    async def get_speakers(self) -> list[dict[str, Any]]:
        """Get list of available speakers from VoiceVox.
        
        Returns:
            List of speaker info dictionaries with name, speaker_uuid, styles, etc.
        """
        try:
            client = await self._get_client()
            response = await client.get("/speakers")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[{_now()}] [VoiceVox] ✗ Failed to get speakers: {e}")
            return []

    async def is_available(self) -> bool:
        """Check if VoiceVox engine is available and responding."""
        try:
            client = await self._get_client()
            response = await client.get("/version", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Clean up resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# Global instance
voicevox_service = VoiceVoxService()


def get_voicevox_service() -> VoiceVoxService:
    """Get the global VoiceVox service instance."""
    return voicevox_service


# VoiceVox Speaker IDs (common ones)
# Full list available via GET /speakers endpoint
#
# RECOMMENDED: Match Gemini "Kore" (bright, young, friendly female)
#   → 春日部つむぎ (Tsumugi) ID=8 - cheerful, bright female voice
#
VOICEVOX_SPEAKERS = {
    # 春日部つむぎ (Kasukabe Tsumugi) - Female, cheerful, bright ← RECOMMENDED (matches Kore)
    "tsumugi_normal": 8,
    
    # 四国めたん (Shikoku Metan) - Female, energetic
    "metan_normal": 2,
    "metan_amaama": 0,     # Sweet
    "metan_tsun": 6,       # Tsundere
    "metan_sexy": 4,       # Sexy
    
    # ずんだもん (Zundamon) - Mascot character, cute voice
    "zundamon_normal": 3,
    "zundamon_amaama": 1,  # Sweet
    "zundamon_tsun": 7,    # Tsundere
    "zundamon_sexy": 5,    # Sexy
    
    # 雨晴はう (Amehare Hau) - Female, soft
    "hau_normal": 10,
    
    # 冥鳴ひまり (Meimei Himari) - Female, gentle
    "himari_normal": 14,
    
    # 九州そら (Kyushu Sora) - Female, various styles
    "sora_normal": 16,
    "sora_amaama": 15,
    "sora_tsun": 18,
    "sora_sexy": 17,
    "sora_sasayaki": 19,  # Whisper
    
    # 波音リツ (Namine Ritsu) - Male, cool
    "ritsu_normal": 9,
    
    # 玄野武宏 (Kurono Takehiro) - Male, mature
    "takehiro_normal": 11,
    "takehiro_whisper": 39,  # Whisper style
    
    # 白上虎太郎 (Shirakami Kotaro) - Male, young boy
    "kotaro_normal": 12,
    
    # 青山龍星 (Aoyama Ryusei) - Male, energetic
    "ryusei_normal": 13,
}

