"""AWS Speech services (Transcribe & Polly) integration using async aioboto3."""

import uuid
from typing import AsyncGenerator

import aioboto3

from config.settings import get_settings


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
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech using AWS Polly (async).

        Args:
            text: Text to synthesize
            voice_id: Polly voice ID (default from settings)
            emotion: Emotion to apply (for SSML)

        Returns:
            Tuple of (audio_bytes, audio_url)
        """
        voice = voice_id or self._settings.polly_voice_id

        # Build SSML with emotion
        ssml_text = self._build_ssml(text, emotion)

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

    def _build_ssml(self, text: str, emotion: str) -> str:
        """Build SSML with emotion markers."""
        # Escape special XML/SSML characters
        escaped_text = self._escape_ssml(text)
        
        # Polly neural voices support some prosody adjustments
        prosody_map = {
            "happy": 'rate="105%" pitch="+5%"',
            "sad": 'rate="90%" pitch="-5%"',
            "surprised": 'rate="110%" pitch="+10%"',
            "calm": 'rate="95%"',
            "excited": 'rate="115%" pitch="+8%"',
            "neutral": "",
        }

        prosody = prosody_map.get(emotion, "")
        
        if prosody:
            return f'<speak><prosody {prosody}>{escaped_text}</prosody></speak>'
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

