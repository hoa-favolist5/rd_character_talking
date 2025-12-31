"""Google Cloud Text-to-Speech service - High-quality natural voices."""

import asyncio
import base64
import re
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from google.cloud import texttospeech

from config.settings import get_settings


class SpeechService:
    """Text-to-Speech service using Google Cloud TTS.
    
    Features:
    - Neural2 voices for highly natural Japanese speech
    - Emotion-based voice adjustments via SSML
    - Returns audio as base64 data URL for instant playback
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: texttospeech.TextToSpeechClient | None = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Default voice from settings
        self._default_voice = self._settings.google_tts_voice
        self._language_code = self._settings.google_tts_language

    def _get_client(self) -> texttospeech.TextToSpeechClient:
        """Get or create TTS client (sync, to be run in executor)."""
        if self._client is None:
            self._client = texttospeech.TextToSpeechClient()
        return self._client

    def _build_ssml(self, text: str, emotion: str) -> str:
        """Build SSML with emotion-based adjustments.
        
        Args:
            text: Plain text to speak
            emotion: Emotion type for voice adjustment
            
        Returns:
            SSML string with prosody adjustments
        """
        # Escape special XML characters
        escaped_text = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
        
        # Emotion-based prosody adjustments
        # rate: x-slow, slow, medium, fast, x-fast, or percentage (e.g., "120%")
        # pitch: x-low, low, medium, high, x-high, or semitones (e.g., "+2st")
        # volume: silent, x-soft, soft, medium, loud, x-loud, or dB (e.g., "+3dB")
        
        if emotion == "happy":
            rate = "110%"
            pitch = "+1st"
            volume = "+2dB"
        elif emotion == "excited":
            rate = "115%"
            pitch = "+2st"
            volume = "+3dB"
        elif emotion == "sad":
            rate = "90%"
            pitch = "-2st"
            volume = "-2dB"
        elif emotion == "calm":
            rate = "95%"
            pitch = "0st"
            volume = "0dB"
        else:  # neutral
            rate = "100%"
            pitch = "0st"
            volume = "0dB"
        
        ssml = f"""<speak>
  <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
    {escaped_text}
  </prosody>
</speak>"""
        
        return ssml

    def _synthesize_sync(
        self,
        text: str,
        voice_name: str | None = None,
        emotion: str = "neutral",
    ) -> bytes:
        """Synchronous speech synthesis (runs in executor).
        
        Args:
            text: Text to synthesize
            voice_name: Voice name override (optional)
            emotion: Emotion for voice adjustment
            
        Returns:
            Audio bytes (MP3 format)
        """
        client = self._get_client()
        voice_name = voice_name or self._default_voice
        
        # Build SSML with emotion
        ssml = self._build_ssml(text, emotion)
        
        # Configure input
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
        
        # Configure voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=self._language_code,
            name=voice_name,
        )
        
        # Configure audio output (MP3 for smaller size, good quality)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Already controlled via SSML
            pitch=0.0,  # Already controlled via SSML
        )
        
        # Synthesize
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        
        return response.audio_content

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech using Google Cloud TTS.
        
        Returns audio as base64 data URL for instant playback.

        Args:
            text: Text to synthesize
            voice_id: Voice name override (optional)
            emotion: Emotion for voice adjustment
            content_type: Content type (unused, for API compatibility)

        Returns:
            Tuple of (audio_bytes, audio_data_url)
        """
        voice_name = voice_id or self._default_voice
        
        print(f"[Google TTS] Text: {text[:50]}..., Voice: {voice_name}, Emotion: {emotion}")
        
        # Run sync synthesis in executor to avoid blocking
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
            voice_name,
            emotion,
        )
        
        print(f"[Google TTS] Generated {len(audio_data)} bytes")
        
        # Return as base64 data URL (instant, no S3 upload needed)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        audio_url = f"data:audio/mp3;base64,{audio_base64}"
        
        return audio_data, audio_url

    async def synthesize_speech_stream(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis from Google Cloud TTS.

        Note: Google TTS doesn't support true streaming synthesis,
        so we synthesize the full audio and yield it in chunks.

        Yields:
            Audio chunks
        """
        audio_data, _ = await self.synthesize_speech(
            text=text,
            voice_id=voice_id,
            emotion=emotion,
            content_type=content_type,
        )
        
        # Yield in chunks
        chunk_size = 4096
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
        
        Strategy: "First Fast, Rest Parallel"
        - Sentence 1: Synthesize immediately and yield ASAP (lowest latency)
        - Sentences 2+: Start in background while sentence 1 is playing
        
        This ensures the user hears audio quickly while remaining
        sentences are prepared during playback.
        
        Yields:
            Tuple of (sentence_text, audio_data_url)
        """
        # Split into sentences (Japanese + English punctuation)
        sentences = re.split(r'(?<=[。！？!?])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            sentences = [text]
        
        voice_name = voice_id or self._default_voice
        
        print(f"[Google TTS STREAM] Generating {len(sentences)} sentences (first-fast strategy)...")
        
        async def synthesize_one(sentence: str) -> str | None:
            """Synthesize a single sentence, return audio_url or None."""
            try:
                _, audio_url = await self.synthesize_speech(
                    text=sentence,
                    voice_id=voice_name,
                    emotion=emotion,
                )
                return audio_url
            except Exception as e:
                print(f"[Google TTS STREAM] Error: {e}")
                return None
        
        # === STRATEGY: First Fast, Rest Parallel ===
        
        # 1. Synthesize and yield FIRST sentence immediately (no waiting)
        first_sentence = sentences[0]
        print(f"[Google TTS STREAM] Synthesizing first sentence immediately...")
        
        first_audio = await synthesize_one(first_sentence)
        if first_audio:
            print(f"[Google TTS STREAM] Sentence 1/{len(sentences)} ready, yielding immediately")
            yield first_sentence, first_audio
        
        # 2. If only one sentence, we're done
        if len(sentences) == 1:
            return
        
        # 3. Synthesize remaining sentences (can be parallel, they have time)
        remaining = sentences[1:]
        
        # Process in small batches of 3 to balance speed vs API rate limits
        BATCH_SIZE = 3
        
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start:batch_start + BATCH_SIZE]
            
            # Synthesize batch in parallel
            tasks = [synthesize_one(s) for s in batch]
            results = await asyncio.gather(*tasks)
            
            # Yield results in order
            for i, (sentence, audio_url) in enumerate(zip(batch, results)):
                if audio_url:
                    sentence_num = batch_start + i + 2  # +2 because first sentence is 1
                    print(f"[Google TTS STREAM] Sentence {sentence_num}/{len(sentences)} ready")
                    yield sentence, audio_url

    async def close(self) -> None:
        """Close the TTS client and executor."""
        self._executor.shutdown(wait=False)
        self._client = None


# Global instance
speech_service = SpeechService()


def get_speech_service() -> SpeechService:
    """Get speech service instance."""
    return speech_service
