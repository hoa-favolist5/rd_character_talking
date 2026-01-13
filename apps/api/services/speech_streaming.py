"""Streaming Speech Service - Token streaming + ElevenLabs TTS.

This service provides ultra-low latency by:
1. Streaming tokens from Anthropic Claude
2. Accumulating into sentences
3. Immediately sending each sentence to ElevenLabs TTS
4. Streaming audio back via WebRTC or WebSocket

Result: First audio byte arrives ~300-500ms after request
(vs ~1-2s for full response + TTS)
"""

import asyncio
import base64
import re
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Awaitable, Callable

from anthropic import AsyncAnthropic

from config.settings import get_settings
from services.speech_elevenlabs import get_elevenlabs_service


def _ms(start: float) -> int:
    """Return elapsed milliseconds since start time."""
    return int((time.perf_counter() - start) * 1000)


def _now() -> str:
    """Return current time as HH:MM:SS.mmm string."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    sentence: str
    audio_data: bytes
    audio_url: str
    index: int
    elapsed_ms: int


@dataclass
class StreamingResponse:
    """Complete streaming response with all audio chunks."""
    full_text: str
    chunks: list[AudioChunk]
    total_time_ms: int
    first_chunk_time_ms: int


# Japanese sentence-ending patterns
SENTENCE_ENDINGS = re.compile(r'[。！？!?]+')
# Pause patterns (comma, etc) - for natural breaks
PAUSE_PATTERNS = re.compile(r'[、,]+')


class StreamingSpeechService:
    """Streaming LLM + TTS service for ultra-low latency.
    
    Architecture:
    1. Stream tokens from Anthropic Claude (Haiku for speed)
    2. Accumulate tokens into a sentence buffer
    3. When sentence boundary detected → immediately send to ElevenLabs
    4. Return audio chunk as soon as it's ready
    
    This achieves ~300-500ms time-to-first-audio vs ~1-2s for non-streaming.
    """
    
    # Minimum sentence length before we'll split on a comma
    MIN_SENTENCE_LENGTH = 15
    # Maximum buffer before forced split (for very long sentences)
    MAX_BUFFER_LENGTH = 100
    # Timeout for TTS requests
    TTS_TIMEOUT = 8.0
    # Max concurrent TTS requests (avoid ElevenLabs 429 rate limit)
    MAX_CONCURRENT_TTS = 2
    # Delay between TTS requests (ms) to avoid burst
    TTS_REQUEST_DELAY_MS = 100
    
    def __init__(self) -> None:
        self._settings = get_settings()
        self._anthropic = AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        self._elevenlabs = get_elevenlabs_service()
        # Semaphore to limit concurrent TTS requests
        self._tts_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_TTS)
    
    def _detect_sentence_boundary(self, buffer: str) -> tuple[str | None, str]:
        """Detect if buffer contains a complete sentence.
        
        Returns:
            (sentence, remaining) if sentence found, else (None, buffer)
        """
        # Check for sentence-ending punctuation
        match = SENTENCE_ENDINGS.search(buffer)
        if match:
            end_pos = match.end()
            sentence = buffer[:end_pos].strip()
            remaining = buffer[end_pos:].strip()
            return sentence, remaining
        
        # For long buffers, split on comma/pause
        if len(buffer) >= self.MIN_SENTENCE_LENGTH:
            match = PAUSE_PATTERNS.search(buffer)
            if match and match.start() >= self.MIN_SENTENCE_LENGTH:
                end_pos = match.end()
                sentence = buffer[:end_pos].strip()
                remaining = buffer[end_pos:].strip()
                return sentence, remaining
        
        # Force split for very long buffers (prevents memory issues)
        if len(buffer) >= self.MAX_BUFFER_LENGTH:
            # Find last space to avoid cutting words
            last_space = buffer.rfind(' ', 0, self.MAX_BUFFER_LENGTH)
            if last_space > self.MIN_SENTENCE_LENGTH:
                sentence = buffer[:last_space].strip()
                remaining = buffer[last_space:].strip()
                return sentence, remaining
            # No good split point, force at MAX_BUFFER_LENGTH
            return buffer[:self.MAX_BUFFER_LENGTH], buffer[self.MAX_BUFFER_LENGTH:]
        
        return None, buffer
    
    async def _synthesize_sentence(
        self,
        sentence: str,
        index: int,
        previous_text: str | None = None,
        next_text: str | None = None,
    ) -> AudioChunk | None:
        """Synthesize a single sentence to audio with context for consistency.
        
        Uses semaphore to limit concurrent requests and avoid rate limiting.
        
        Args:
            sentence: Text to synthesize
            index: Chunk index for ordering
            previous_text: Text spoken before this sentence (for prosody consistency)
            next_text: Text that will be spoken after (for prosody consistency)
        """
        if not sentence.strip():
            return None
        
        # Wait for semaphore (limits concurrent TTS requests)
        async with self._tts_semaphore:
            start = time.perf_counter()
            
            try:
                async with asyncio.timeout(self.TTS_TIMEOUT):
                    audio_data, audio_url = await self._elevenlabs.synthesize_speech(
                        sentence,
                        previous_text=previous_text,
                        next_text=next_text,
                    )
                
                elapsed = _ms(start)
                context_info = ""
                if previous_text or next_text:
                    context_info = f" [ctx: prev={len(previous_text or '')}c, next={len(next_text or '')}c]"
                print(f"[{_now()}] [TTS] Chunk {index}: {len(sentence)} chars → {len(audio_data):,}B ({elapsed}ms){context_info}")
                
                return AudioChunk(
                    sentence=sentence,
                    audio_data=audio_data,
                    audio_url=audio_url,
                    index=index,
                    elapsed_ms=elapsed,
                )
                
            except asyncio.TimeoutError:
                print(f"[{_now()}] [TTS] Chunk {index} timeout ({self.TTS_TIMEOUT}s)")
                return None
            except Exception as e:
                print(f"[{_now()}] [TTS] Chunk {index} error: {e}")
                return None
    
    async def generate_streaming(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        model: str | None = None,
        use_fast_model: bool = False,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        on_audio_chunk: Callable[[AudioChunk], Awaitable[None]] | None = None,
    ) -> StreamingResponse:
        """
        Generate response with streaming tokens and audio.
        
        This is the main entry point for streaming generation.
        
        Args:
            messages: Conversation history
            system_prompt: System instruction
            max_tokens: Max output tokens
            temperature: Sampling temperature
            model: Model to use (defaults to main Sonnet model for consistency)
            use_fast_model: If True, use Haiku for faster but simpler responses
            on_token: Callback for each token (for live text display)
            on_audio_chunk: Callback for each audio chunk (for streaming playback)
        
        Returns:
            StreamingResponse with full text and all audio chunks
        """
        total_start = time.perf_counter()
        first_chunk_time: int | None = None
        
        # Use main model (Sonnet) for consistency with REST API
        # Or fast model (Haiku) if explicitly requested
        if model:
            llm_model = model
        elif use_fast_model:
            llm_model = self._settings.anthropic_fast_model
        else:
            llm_model = self._settings.anthropic_model  # Sonnet for quality
        
        print(f"[{_now()}] [Stream] Starting with {llm_model}")
        
        # State for accumulation
        token_buffer = ""
        full_text_parts: list[str] = []
        chunk_index = 0
        chunks: list[AudioChunk] = []
        
        # Queue for TTS tasks (run in parallel with token streaming)
        tts_tasks: list[asyncio.Task] = []
        
        # Track previous sentences for context (prosody consistency)
        # Keep last 2-3 sentences for context (ElevenLabs recommendation)
        MAX_CONTEXT_SENTENCES = 3
        
        def get_previous_context() -> str | None:
            """Get previous sentences as context for TTS."""
            if not full_text_parts:
                return None
            # Join last few sentences as context
            context_parts = full_text_parts[-MAX_CONTEXT_SENTENCES:]
            return " ".join(context_parts)
        
        try:
            # Stream tokens from Claude
            async with self._anthropic.messages.stream(
                model=llm_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
            ) as stream:
                async for token in stream.text_stream:
                    # Notify token callback (for live text display)
                    if on_token:
                        await on_token(token)
                    
                    # Accumulate token
                    token_buffer += token
                    
                    # Check for sentence boundary
                    sentence, token_buffer = self._detect_sentence_boundary(token_buffer)
                    
                    if sentence:
                        full_text_parts.append(sentence)
                        chunk_index += 1
                        
                        # Start TTS with context for prosody consistency
                        current_index = chunk_index
                        current_sentence = sentence
                        # Get context from previously completed sentences
                        prev_context = get_previous_context()
                        # Note: next_text not available in streaming mode
                        
                        async def synthesize_and_notify(sent: str, idx: int, prev_text: str | None):
                            chunk = await self._synthesize_sentence(
                                sent, idx,
                                previous_text=prev_text,
                                next_text=None,  # Not available in streaming
                            )
                            if chunk:
                                nonlocal first_chunk_time
                                if first_chunk_time is None:
                                    first_chunk_time = _ms(total_start)
                                    print(f"[{_now()}] [Stream] ⚡ First audio at {first_chunk_time}ms")
                                
                                chunks.append(chunk)
                                if on_audio_chunk:
                                    await on_audio_chunk(chunk)
                            return chunk
                        
                        task = asyncio.create_task(
                            synthesize_and_notify(current_sentence, current_index, prev_context)
                        )
                        tts_tasks.append(task)
                        
                        # Small delay between scheduling to prevent burst
                        await asyncio.sleep(self.TTS_REQUEST_DELAY_MS / 1000)
            
            # Handle any remaining buffer
            if token_buffer.strip():
                full_text_parts.append(token_buffer.strip())
                chunk_index += 1
                
                # Pass context for the final sentence
                prev_context = get_previous_context()
                chunk = await self._synthesize_sentence(
                    token_buffer.strip(), chunk_index,
                    previous_text=prev_context,
                )
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = _ms(total_start)
                    chunks.append(chunk)
                    if on_audio_chunk:
                        await on_audio_chunk(chunk)
            
            # Wait for all TTS tasks to complete
            if tts_tasks:
                await asyncio.gather(*tts_tasks, return_exceptions=True)
            
            # Sort chunks by index (they may have completed out of order)
            chunks.sort(key=lambda c: c.index)
            
            total_time = _ms(total_start)
            full_text = "".join(full_text_parts)
            
            print(f"[{_now()}] [Stream] ✓ Complete: {len(full_text)} chars, {len(chunks)} chunks, {total_time}ms total")
            if first_chunk_time:
                print(f"[{_now()}] [Stream]   First audio: {first_chunk_time}ms")
            
            return StreamingResponse(
                full_text=full_text,
                chunks=chunks,
                total_time_ms=total_time,
                first_chunk_time_ms=first_chunk_time or total_time,
            )
            
        except Exception as e:
            total_time = _ms(total_start)
            print(f"[{_now()}] [Stream] ✗ Error after {total_time}ms: {e}")
            raise
    
    async def generate_streaming_simple(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 500,
        use_fast_model: bool = False,
    ) -> AsyncGenerator[tuple[str, str, int], None]:
        """
        Simplified streaming interface - yields (sentence, audio_url, index) tuples.
        
        Use this for WebSocket streaming where you want to send chunks as they arrive.
        
        Yields:
            (sentence, audio_url, chunk_index) for each sentence
        """
        total_start = time.perf_counter()
        
        # Use main model (Sonnet) for consistency, or Haiku if explicitly requested
        llm_model = self._settings.anthropic_fast_model if use_fast_model else self._settings.anthropic_model
        
        print(f"[{_now()}] [SimpleStream] Starting with {llm_model}")
        
        # State
        token_buffer = ""
        chunk_index = 0
        previous_sentences: list[str] = []  # Track for context
        MAX_CONTEXT_SENTENCES = 3
        
        def get_previous_context() -> str | None:
            """Get previous sentences as context for prosody consistency."""
            if not previous_sentences:
                return None
            return " ".join(previous_sentences[-MAX_CONTEXT_SENTENCES:])
        
        # Stream tokens
        async with self._anthropic.messages.stream(
            model=llm_model,
            max_tokens=max_tokens,
            temperature=0.7,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for token in stream.text_stream:
                token_buffer += token
                
                # Check for sentence boundary
                sentence, token_buffer = self._detect_sentence_boundary(token_buffer)
                
                if sentence:
                    chunk_index += 1
                    
                    # Synthesize with context for prosody consistency
                    prev_context = get_previous_context()
                    chunk = await self._synthesize_sentence(
                        sentence, chunk_index,
                        previous_text=prev_context,
                    )
                    
                    previous_sentences.append(sentence)
                    
                    if chunk:
                        yield sentence, chunk.audio_url, chunk_index
        
        # Handle remaining buffer
        if token_buffer.strip():
            chunk_index += 1
            prev_context = get_previous_context()
            chunk = await self._synthesize_sentence(
                token_buffer.strip(), chunk_index,
                previous_text=prev_context,
            )
            
            if chunk:
                yield token_buffer.strip(), chunk.audio_url, chunk_index
        
        total_time = _ms(total_start)
        print(f"[{_now()}] [SimpleStream] ✓ Complete: {chunk_index} chunks, {total_time}ms")


# Global instance
_streaming_service: StreamingSpeechService | None = None


def get_streaming_speech_service() -> StreamingSpeechService:
    """Get the global streaming speech service instance."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingSpeechService()
    return _streaming_service
