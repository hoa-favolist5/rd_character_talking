"""
Amazon Nova 2 Sonic - Speech-to-Speech Service.

Model ID: amazon.nova-2-sonic-v1:0

Note: Nova 2 Sonic uses bidirectional streaming.
This implementation tries converse_stream API.
"""

import base64
import json
import re
from typing import AsyncGenerator

import boto3

from config.settings import get_settings


class NovaSonicService:
    """Text-to-Speech using Amazon Nova 2 Sonic."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = None
        
    def _get_client(self):
        """Get Bedrock Runtime client."""
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._settings.aws_region,
                aws_access_key_id=self._settings.aws_access_key_id,
                aws_secret_access_key=self._settings.aws_secret_access_key,
            )
        return self._client

    async def synthesize_speech(
        self,
        text: str,
        system_prompt: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Synthesize text to speech using Nova 2 Sonic via converse_stream.
        """
        client = self._get_client()
        
        print(f"[NOVA 2 SONIC] TTS: {text[:50]}...")
        
        try:
            # Use converse_stream for streaming response
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": f"Please say this in a friendly voice: {text}"}
                    ]
                }
            ]
            
            system = []
            if system_prompt:
                system = [{"text": system_prompt}]
            
            response = client.converse_stream(
                modelId=self._settings.nova_sonic_model_id,
                messages=messages,
                system=system,
                inferenceConfig={
                    "maxTokens": 1024,
                    "temperature": 0.7,
                },
            )
            
            # Collect response
            audio_chunks = []
            response_text = ""
            
            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        response_text += delta["text"]
                    if "audio" in delta:
                        audio_data = delta["audio"].get("data", "")
                        if audio_data:
                            audio_chunks.append(base64.b64decode(audio_data))
            
            if audio_chunks:
                audio_bytes = b"".join(audio_chunks)
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                audio_url = f"data:audio/mp3;base64,{audio_base64}"
                print(f"[NOVA 2 SONIC] Generated {len(audio_bytes)} bytes")
                return audio_bytes, audio_url
            else:
                print(f"[NOVA 2 SONIC] No audio, text response: {response_text[:100]}")
                raise RuntimeError(f"No audio output. Response: {response_text[:100]}")
                
        except Exception as e:
            print(f"[NOVA 2 SONIC] Error: {e}")
            raise

    async def synthesize_sentences(
        self,
        text: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Synthesize text sentence by sentence."""
        sentences = re.split(r'(?<=[。！？!?.])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            sentences = [text]
        
        print(f"[NOVA 2 SONIC] Streaming {len(sentences)} sentences...")
        
        for i, sentence in enumerate(sentences):
            try:
                _, audio_url = await self.synthesize_speech(sentence, system_prompt)
                print(f"[NOVA 2 SONIC] Sentence {i+1}/{len(sentences)} ready")
                yield sentence, audio_url
            except Exception as e:
                print(f"[NOVA 2 SONIC] Sentence {i+1} error: {e}")
                continue

    async def process_text_to_speech(
        self,
        text: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Process text and stream audio response."""
        try:
            _, audio_url = await self.synthesize_speech(text, system_prompt)
            
            yield {"type": "response_text", "text": text}
            yield {"type": "audio", "data": audio_url.split(",")[1]}
            yield {"type": "end"}
            
        except Exception as e:
            print(f"[NOVA 2 SONIC] Error: {e}")
            raise


# Global instance
_service = None


def get_nova_sonic_service() -> NovaSonicService:
    """Get Nova 2 Sonic service instance."""
    global _service
    if _service is None:
        _service = NovaSonicService()
    return _service
