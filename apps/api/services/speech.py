"""AWS Speech services (Transcribe & Polly) integration using async aioboto3."""

import uuid
from typing import AsyncGenerator

import aioboto3

from config.settings import get_settings
from config.voices import ContentType, VoiceConfig, get_voice_config


class SpeechService:
    """Service for speech-to-text and text-to-speech using AWS (async)."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session = aioboto3.Session(
            aws_access_key_id=self._settings.aws_access_key_id or None,
            aws_secret_access_key=self._settings.aws_secret_access_key or None,
            region_name=self._settings.aws_region,
        )

    def _get_client_kwargs(self) -> dict:
        """Get common kwargs for client creation."""
        return {
            "region_name": self._settings.aws_region,
        }

    async def transcribe_audio(self, audio_data: bytes, mime_type: str = "audio/webm") -> str:
        """
        Transcribe audio to text using AWS Transcribe (async).

        Args:
            audio_data: Raw audio bytes
            mime_type: Audio MIME type

        Returns:
            Transcribed text
        """
        import asyncio
        import httpx

        # Upload audio to S3 for Transcribe
        job_name = f"transcribe-{uuid.uuid4().hex}"
        s3_key = f"transcribe/{job_name}.webm"

        # Determine media format
        format_map = {
            "audio/webm": "webm",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
            "audio/mp3": "mp3",
        }
        media_format = format_map.get(mime_type, "webm")

        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            # Upload to S3
            await s3.put_object(
                Bucket=self._settings.s3_bucket_audio,
                Key=s3_key,
                Body=audio_data,
                ContentType=mime_type,
            )

        s3_uri = f"s3://{self._settings.s3_bucket_audio}/{s3_key}"

        async with self._session.client("transcribe", **self._get_client_kwargs()) as transcribe:
            # Start transcription job
            await transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": s3_uri},
                MediaFormat=media_format,
                LanguageCode=self._settings.transcribe_language_code,
            )

            # Wait for completion
            while True:
                status = await transcribe.get_transcription_job(TranscriptionJobName=job_name)
                job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]

                if job_status == "COMPLETED":
                    break
                elif job_status == "FAILED":
                    raise RuntimeError("Transcription failed")

                await asyncio.sleep(0.5)

        # Get transcript
        transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        async with httpx.AsyncClient() as client:
            response = await client.get(transcript_uri)
            result = response.json()

        # Clean up S3
        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            await s3.delete_object(Bucket=self._settings.s3_bucket_audio, Key=s3_key)

        return result["results"]["transcripts"][0]["transcript"]

    async def synthesize_speech(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str = "neutral",
        content_type: ContentType | str | None = None,
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech using AWS Polly (async).

        Args:
            text: Text to synthesize
            voice_id: Polly voice ID (override, default from settings or content_type)
            emotion: Emotion to apply (for SSML prosody)
            content_type: Content type for dynamic voice selection

        Returns:
            Tuple of (audio_bytes, audio_url)
        """
        # Get voice config based on content type if provided
        if content_type:
            if isinstance(content_type, str):
                try:
                    content_type = ContentType(content_type)
                except ValueError:
                    content_type = ContentType.NEUTRAL
            voice_config = get_voice_config(content_type)
        else:
            voice_config = None
        
        # Use voice_id override, then content-based voice, then default
        if voice_id:
            voice = voice_id
            voice_source = "override"
        elif voice_config:
            voice = voice_config.voice_id
            voice_source = f"content_type:{content_type.value}"
        else:
            voice = self._settings.polly_voice_id
            voice_source = "default"
        
        print(f"[TTS DEBUG] Voice: {voice} (source: {voice_source}), Content type: {content_type}")

        # Build SSML with emotion and content-based prosody
        ssml_text = self._build_ssml(text, emotion, voice_config)
        print(f"[TTS DEBUG] SSML: {ssml_text[:100]}...")

        async with self._session.client("polly", **self._get_client_kwargs()) as polly:
            response = await polly.synthesize_speech(
                Text=ssml_text,
                TextType="ssml",
                OutputFormat="mp3",
                VoiceId=voice,
                Engine=self._settings.polly_engine,
                LanguageCode="ja-JP",
            )

            # Read audio stream asynchronously
            async with response["AudioStream"] as stream:
                audio_data = await stream.read()

        # Upload to S3 and get URL
        audio_key = f"audio/{uuid.uuid4().hex}.mp3"
        
        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            await s3.put_object(
                Bucket=self._settings.s3_bucket_audio,
                Key=audio_key,
                Body=audio_data,
                ContentType="audio/mpeg",
            )

            # Generate presigned URL (valid for 1 hour)
            audio_url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._settings.s3_bucket_audio, "Key": audio_key},
                ExpiresIn=3600,
            )

        return audio_data, audio_url

    def _build_ssml(
        self, 
        text: str, 
        emotion: str, 
        voice_config: VoiceConfig | None = None,
        use_neural: bool = True,
    ) -> str:
        """Build SSML with emotion and content-based prosody markers.
        
        Note: AWS Polly Neural voices support 'rate' and 'volume' in prosody tag.
        The 'pitch' attribute is NOT supported for neural voices.
        
        Volume values: silent, x-soft, soft, medium, loud, x-loud, or +/-NdB
        Rate values: 20% to 200%
        """
        # Escape special XML/SSML characters
        escaped_text = self._escape_ssml(text)
        
        # Start with voice config rate if available
        if voice_config and voice_config.rate != "100%":
            base_rate = voice_config.rate
        else:
            base_rate = "100%"
        
        # Get volume from voice config
        volume = getattr(voice_config, 'volume', 'medium') if voice_config else 'medium'
        
        # Emotion-based rate adjustments
        emotion_rate_adjustments = {
            "happy": 10,
            "sad": -15,
            "surprised": 15,
            "calm": -8,
            "excited": 20,
            "frustrated": 8,
            "curious": 0,
            "neutral": 0,
        }
        
        # Emotion-based volume adjustments (in dB, applied to base volume)
        emotion_volume_adjustments = {
            "happy": "+2dB",
            "sad": "-2dB",
            "surprised": "+3dB",
            "calm": "-1dB",
            "excited": "+4dB",
            "frustrated": "+2dB",
            "curious": "+0dB",
            "neutral": "+0dB",
        }
        
        rate_mod = emotion_rate_adjustments.get(emotion, 0)
        volume_mod = emotion_volume_adjustments.get(emotion, "+0dB")
        
        # Parse and combine rate
        base_rate_num = int(base_rate.rstrip('%'))
        final_rate = base_rate_num + rate_mod
        final_rate = max(50, min(200, final_rate))  # Clamp to valid range (AWS allows up to 200%)
        
        # Determine final volume (combine base volume with emotion adjustment)
        # If emotion has a dB adjustment and base volume is not already in dB, use emotion's dB
        if volume_mod != "+0dB" and not volume.endswith("dB"):
            final_volume = volume_mod
        else:
            final_volume = volume
        
        # Build prosody attributes
        prosody_attrs = []
        if final_rate != 100:
            prosody_attrs.append(f'rate="{final_rate}%"')
        if final_volume != "medium":
            prosody_attrs.append(f'volume="{final_volume}"')
        
        if prosody_attrs:
            attrs_str = " ".join(prosody_attrs)
            return f'<speak><prosody {attrs_str}>{escaped_text}</prosody></speak>'
        return f"<speak>{escaped_text}</speak>"
    
    def _escape_ssml(self, text: str) -> str:
        """Escape special characters for SSML."""
        # Must escape in this order: & first, then others
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text

    async def synthesize_speech_stream(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to speech with streaming output (async).

        Yields:
            Audio chunks
        """
        voice = voice_id or self._settings.polly_voice_id

        escaped_text = self._escape_ssml(text)
        
        async with self._session.client("polly", **self._get_client_kwargs()) as polly:
            response = await polly.synthesize_speech(
                Text=f"<speak>{escaped_text}</speak>",
                TextType="ssml",
                OutputFormat="mp3",
                VoiceId=voice,
                Engine=self._settings.polly_engine,
                LanguageCode="ja-JP",
            )

            # Stream audio in chunks asynchronously
            chunk_size = 4096
            async with response["AudioStream"] as stream:
                while True:
                    chunk = await stream.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk


# Global instance
speech_service = SpeechService()


def get_speech_service() -> SpeechService:
    """Get speech service instance."""
    return speech_service

