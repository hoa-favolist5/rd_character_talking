"""VOICEVOX TTS service - Optimized for low latency."""

import base64
import re
from typing import AsyncGenerator

import httpx

from config.settings import get_settings


class SpeechService:
    """Text-to-Speech service using VOICEVOX.
    
    Optimized for minimal latency:
    - Reuses HTTP client connection
    - Returns audio as base64 data URL (no S3 upload)
    - ~100-200ms total response time
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: httpx.AsyncClient | None = None
        
        # Default speaker from settings
        self._default_speaker = self._settings.voicevox_speaker

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._settings.voicevox_url,
                timeout=30.0,
            )
        return self._client

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech using VOICEVOX.
        
        Returns audio as base64 data URL for instant playback (no S3 upload).

        Args:
            text: Text to synthesize
            voice_id: Speaker ID override (optional)
            emotion: Emotion for voice adjustment
            content_type: Content type (unused, for API compatibility)

        Returns:
            Tuple of (audio_bytes, audio_data_url)
        """
        speaker = int(voice_id) if voice_id else self._default_speaker
        
        print(f"[VOICEVOX] Text: {text[:50]}..., Speaker: {speaker}")
        
        client = await self._get_client()
        
        # Step 1: Create audio query from text
        query_response = await client.post(
            "/audio_query",
            params={"text": text, "speaker": speaker},
        )
        
        if query_response.status_code != 200:
            raise RuntimeError(f"VOICEVOX audio_query failed: {query_response.text}")
        
        audio_query = query_response.json()
        
        # Adjust parameters based on emotion
        self._apply_emotion(audio_query, emotion)
        
        # Step 2: Synthesize audio
        synthesis_response = await client.post(
            "/synthesis",
            params={"speaker": speaker},
            json=audio_query,
        )
        
        if synthesis_response.status_code != 200:
            raise RuntimeError(f"VOICEVOX synthesis failed: {synthesis_response.text}")
        
        audio_data = synthesis_response.content
        
        print(f"[VOICEVOX] Generated {len(audio_data)} bytes")
        
        # Return as base64 data URL (instant, no S3 upload needed)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        audio_url = f"data:audio/wav;base64,{audio_base64}"
        
        return audio_data, audio_url

    def _apply_emotion(self, audio_query: dict, emotion: str) -> None:
        """Apply emotion-based voice adjustments."""
        # speedScale: 0.5 ~ 2.0 (default 1.0)
        # pitchScale: -0.15 ~ 0.15 (default 0.0)  
        # intonationScale: 0.0 ~ 2.0 (default 1.0)
        # volumeScale: 0.0 ~ 2.0 (default 1.0)
        
        if emotion == "happy":
            audio_query["speedScale"] = 1.1
            audio_query["intonationScale"] = 1.15
            audio_query["volumeScale"] = 1.05
        elif emotion == "excited":
            audio_query["speedScale"] = 1.15
            audio_query["intonationScale"] = 1.2
            audio_query["volumeScale"] = 1.1
        elif emotion == "sad":
            audio_query["speedScale"] = 0.9
            audio_query["pitchScale"] = -0.05
            audio_query["intonationScale"] = 0.85
            audio_query["volumeScale"] = 0.9
        elif emotion == "calm":
            audio_query["speedScale"] = 0.95
            audio_query["intonationScale"] = 0.9
        # neutral uses defaults

    async def synthesize_speech_stream(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis from VOICEVOX.

        Yields:
            Audio chunks
        """
        speaker = int(voice_id) if voice_id else self._default_speaker
        
        client = await self._get_client()
        
        # Step 1: Create audio query
        query_response = await client.post(
            "/audio_query",
            params={"text": text, "speaker": speaker},
        )
        
        if query_response.status_code != 200:
            raise RuntimeError(f"VOICEVOX audio_query failed: {query_response.text}")
        
        audio_query = query_response.json()
        self._apply_emotion(audio_query, emotion)
        
        # Step 2: Stream synthesis
        async with client.stream(
            "POST",
            "/synthesis",
            params={"speaker": speaker},
            json=audio_query,
        ) as response:
            async for chunk in response.aiter_bytes(chunk_size=4096):
                yield chunk

    async def synthesize_sentences(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Synthesize text sentence by sentence for streaming playback.
        
        Yields audio for each sentence immediately, allowing frontend
        to play first sentence while others are still generating.
        
        Yields:
            Tuple of (sentence_text, audio_data_url)
        """
        # Split into sentences (Japanese + English punctuation)
        sentences = re.split(r'(?<=[。！？!?])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            sentences = [text]
        
        speaker = int(voice_id) if voice_id else self._default_speaker
        client = await self._get_client()
        
        print(f"[VOICEVOX STREAM] Generating {len(sentences)} sentences...")
        
        for i, sentence in enumerate(sentences):
            try:
                # Generate audio for this sentence
                query_response = await client.post(
                    "/audio_query",
                    params={"text": sentence, "speaker": speaker},
                )
                
                if query_response.status_code != 200:
                    continue
                
                audio_query = query_response.json()
                self._apply_emotion(audio_query, emotion)
                
                synthesis_response = await client.post(
                    "/synthesis",
                    params={"speaker": speaker},
                    json=audio_query,
                )
                
                if synthesis_response.status_code != 200:
                    continue
                
                audio_data = synthesis_response.content
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                audio_url = f"data:audio/wav;base64,{audio_base64}"
                
                print(f"[VOICEVOX STREAM] Sentence {i+1}/{len(sentences)}: {len(audio_data)} bytes")
                
                yield sentence, audio_url
                
            except Exception as e:
                print(f"[VOICEVOX STREAM] Error on sentence {i+1}: {e}")
                continue

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# Global instance
speech_service = SpeechService()


def get_speech_service() -> SpeechService:
    """Get speech service instance."""
    return speech_service
