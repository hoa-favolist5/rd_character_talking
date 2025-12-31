"""Gemini 2.5 Flash Preview TTS service - Natural AI-generated voices."""

import asyncio
import base64
import re
import wave
import io
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from google import genai
from google.genai import types

from config.settings import get_settings


class SpeechService:
    """Text-to-Speech service using Gemini 2.5 Flash Preview TTS.
    
    Features:
    - Gemini's most natural AI-generated voices
    - Multiple voice options (Puck, Charon, Kore, Fenrir, Aoede)
    - Emotion-based voice prompting
    - Returns audio as base64 data URL for instant playback
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: genai.Client | None = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Default voice from settings
        self._default_voice = self._settings.gemini_tts_voice
        self._model = "gemini-2.5-flash-preview-tts"

    def _get_client(self) -> genai.Client:
        """Get or create Gemini client (sync, to be run in executor)."""
        if self._client is None:
            self._client = genai.Client(api_key=self._settings.google_api_key)
        return self._client

    def _build_prompt(self, text: str, emotion: str) -> str:
        """Build prompt with emotion context for natural speech.
        
        Args:
            text: Plain text to speak
            emotion: Emotion type for voice style
            
        Returns:
            Prompt string with emotion guidance
        """
        # Emotion-based speaking style guidance
        emotion_styles = {
            "happy": "Say this in a cheerful, upbeat, and happy tone:",
            "excited": "Say this with excitement and high energy:",
            "sad": "Say this in a gentle, soft, and slightly melancholic tone:",
            "calm": "Say this in a calm, soothing, and relaxed manner:",
            "neutral": "Say this naturally:",
        }
        
        style = emotion_styles.get(emotion, emotion_styles["neutral"])
        return f"{style} {text}"

    def _synthesize_sync(
        self,
        text: str,
        voice_name: str | None = None,
        emotion: str = "neutral",
    ) -> bytes:
        """Synchronous speech synthesis using Gemini TTS (runs in executor).
        
        Args:
            text: Text to synthesize
            voice_name: Voice name override (optional)
            emotion: Emotion for voice style
            
        Returns:
            Audio bytes (WAV format, 24kHz)
        """
        client = self._get_client()
        voice_name = voice_name or self._default_voice
        
        # Build prompt with emotion
        prompt = self._build_prompt(text, emotion)
        
        # Configure speech settings
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name,
                )
            )
        )
        
        # Generate audio using Gemini TTS
        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            ),
        )
        
        # Extract audio data from response
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        
        # Convert raw PCM to WAV format (Gemini returns raw PCM at 24kHz)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz sample rate
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
        Synthesize text to speech using Gemini 2.5 Flash Preview TTS.
        
        Returns audio as base64 data URL for instant playback.

        Args:
            text: Text to synthesize
            voice_id: Voice name override (optional)
            emotion: Emotion for voice style
            content_type: Content type (unused, for API compatibility)

        Returns:
            Tuple of (audio_bytes, audio_data_url)
        """
        voice_name = voice_id or self._default_voice
        
        print(f"[Gemini TTS] Text: {text[:50]}..., Voice: {voice_name}, Emotion: {emotion}")
        
        # Run sync synthesis in executor to avoid blocking
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            self._executor,
            self._synthesize_sync,
            text,
            voice_name,
            emotion,
        )
        
        print(f"[Gemini TTS] Generated {len(audio_data)} bytes")
        
        # Return as base64 data URL (instant, no upload needed)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        audio_url = f"data:audio/wav;base64,{audio_base64}"
        
        return audio_data, audio_url

    async def synthesize_speech_stream(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis from Gemini TTS.

        Note: Gemini TTS doesn't support true streaming synthesis,
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
        
        print(f"[Gemini TTS STREAM] Generating {len(sentences)} sentences (first-fast strategy)...")
        
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
                print(f"[Gemini TTS STREAM] Error: {e}")
                return None
        
        # === STRATEGY: First Fast, Rest Parallel ===
        
        # 1. Synthesize and yield FIRST sentence immediately (no waiting)
        first_sentence = sentences[0]
        print(f"[Gemini TTS STREAM] Synthesizing first sentence immediately...")
        
        first_audio = await synthesize_one(first_sentence)
        if first_audio:
            print(f"[Gemini TTS STREAM] Sentence 1/{len(sentences)} ready, yielding immediately")
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
                    print(f"[Gemini TTS STREAM] Sentence {sentence_num}/{len(sentences)} ready")
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
