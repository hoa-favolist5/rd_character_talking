"""WebRTC Audio Streaming Service.

Provides ultra-low latency audio transport using WebRTC.
Replaces base64-over-WebSocket for significant latency reduction.

Architecture:
- Socket.IO for signaling (SDP offer/answer, ICE candidates)
- WebRTC DataChannel for reliable audio chunk delivery
- Optional: WebRTC MediaStream for raw audio (future)

Benefits over base64 WebSocket:
- ~50-100ms lower latency
- No base64 encoding/decoding overhead
- Better congestion control
- Native browser audio handling
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.media import MediaPlayer, MediaRecorder

from config.settings import get_settings


def _now() -> str:
    """Return current time as HH:MM:SS.mmm string."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


@dataclass
class WebRTCSession:
    """Represents a WebRTC peer connection session."""
    session_id: str
    peer_connection: RTCPeerConnection
    data_channel: RTCDataChannel | None = None
    created_at: float = field(default_factory=time.time)
    audio_chunks_sent: int = 0
    
    async def close(self):
        """Close the peer connection."""
        if self.data_channel:
            self.data_channel.close()
        await self.peer_connection.close()


class WebRTCService:
    """WebRTC service for audio streaming.
    
    Handles:
    - Peer connection lifecycle
    - SDP negotiation via Socket.IO signaling
    - Audio chunk transmission via DataChannel
    
    Usage:
    1. Client sends 'webrtc_offer' via Socket.IO
    2. Server creates answer, sends back 'webrtc_answer'
    3. ICE candidates exchanged via 'webrtc_ice' events
    4. Once connected, audio sent via DataChannel
    """
    
    # STUN servers for NAT traversal
    ICE_SERVERS = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
    
    def __init__(self) -> None:
        self._settings = get_settings()
        self._sessions: dict[str, WebRTCSession] = {}
        self._on_message_handlers: dict[str, Callable[[str, Any], Awaitable[None]]] = {}
    
    def _create_peer_connection(self) -> RTCPeerConnection:
        """Create a new RTCPeerConnection with ICE servers."""
        config = {
            "iceServers": self.ICE_SERVERS,
        }
        return RTCPeerConnection(configuration=config)
    
    async def create_session(
        self,
        session_id: str,
        offer_sdp: str,
        offer_type: str = "offer",
    ) -> tuple[str, str]:
        """
        Create a new WebRTC session from client's offer.
        
        Args:
            session_id: Unique session identifier
            offer_sdp: Client's SDP offer
            offer_type: SDP type (usually "offer")
        
        Returns:
            (answer_sdp, answer_type) - Server's SDP answer
        """
        print(f"[{_now()}] [WebRTC] Creating session {session_id[:8]}...")
        
        # Create peer connection
        pc = self._create_peer_connection()
        
        # Handle data channel creation from client
        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            print(f"[{_now()}] [WebRTC] DataChannel '{channel.label}' opened")
            session = self._sessions.get(session_id)
            if session:
                session.data_channel = channel
            
            @channel.on("message")
            async def on_message(message):
                # Handle incoming messages from client
                handler = self._on_message_handlers.get(session_id)
                if handler:
                    try:
                        data = json.loads(message) if isinstance(message, str) else message
                        await handler(session_id, data)
                    except Exception as e:
                        print(f"[{_now()}] [WebRTC] Message handler error: {e}")
        
        # Handle connection state changes
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            print(f"[{_now()}] [WebRTC] Connection state: {pc.connectionState}")
            if pc.connectionState == "failed":
                await self.close_session(session_id)
            elif pc.connectionState == "closed":
                self._sessions.pop(session_id, None)
        
        # Handle ICE connection state
        @pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            print(f"[{_now()}] [WebRTC] ICE state: {pc.iceConnectionState}")
        
        # Set remote description (client's offer)
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Store session
        session = WebRTCSession(
            session_id=session_id,
            peer_connection=pc,
        )
        self._sessions[session_id] = session
        
        print(f"[{_now()}] [WebRTC] Session {session_id[:8]} created, answer ready")
        
        return pc.localDescription.sdp, pc.localDescription.type
    
    async def add_ice_candidate(
        self,
        session_id: str,
        candidate: dict,
    ) -> bool:
        """
        Add ICE candidate from client.
        
        Args:
            session_id: Session identifier
            candidate: ICE candidate dict with 'candidate', 'sdpMid', 'sdpMLineIndex'
        
        Returns:
            True if successful
        """
        session = self._sessions.get(session_id)
        if not session:
            print(f"[{_now()}] [WebRTC] Session {session_id[:8]} not found for ICE candidate")
            return False
        
        try:
            from aiortc import RTCIceCandidate
            
            # Parse ICE candidate
            ice_candidate = RTCIceCandidate(
                sdpMid=candidate.get("sdpMid"),
                sdpMLineIndex=candidate.get("sdpMLineIndex"),
                candidate=candidate.get("candidate"),
            )
            
            await session.peer_connection.addIceCandidate(ice_candidate)
            return True
            
        except Exception as e:
            print(f"[{_now()}] [WebRTC] ICE candidate error: {e}")
            return False
    
    async def send_audio_chunk(
        self,
        session_id: str,
        audio_data: bytes,
        chunk_index: int,
        sentence: str = "",
        is_last: bool = False,
    ) -> bool:
        """
        Send audio chunk via WebRTC DataChannel.
        
        Args:
            session_id: Session identifier
            audio_data: Raw audio bytes (MP3 from ElevenLabs)
            chunk_index: Chunk sequence number
            sentence: The text that was synthesized
            is_last: Whether this is the last chunk
        
        Returns:
            True if sent successfully
        """
        session = self._sessions.get(session_id)
        if not session or not session.data_channel:
            return False
        
        if session.data_channel.readyState != "open":
            print(f"[{_now()}] [WebRTC] DataChannel not open (state: {session.data_channel.readyState})")
            return False
        
        try:
            # Send as binary with metadata header
            # Format: [4 bytes: chunk_index][1 byte: is_last][audio_data]
            header = chunk_index.to_bytes(4, 'big') + (1 if is_last else 0).to_bytes(1, 'big')
            message = header + audio_data
            
            session.data_channel.send(message)
            session.audio_chunks_sent += 1
            
            print(f"[{_now()}] [WebRTC] Sent chunk {chunk_index}: {len(audio_data):,}B")
            return True
            
        except Exception as e:
            print(f"[{_now()}] [WebRTC] Send error: {e}")
            return False
    
    async def send_json(
        self,
        session_id: str,
        data: dict,
    ) -> bool:
        """
        Send JSON message via DataChannel.
        
        Args:
            session_id: Session identifier
            data: JSON-serializable data
        
        Returns:
            True if sent successfully
        """
        session = self._sessions.get(session_id)
        if not session or not session.data_channel:
            return False
        
        if session.data_channel.readyState != "open":
            return False
        
        try:
            message = json.dumps(data)
            session.data_channel.send(message)
            return True
        except Exception as e:
            print(f"[{_now()}] [WebRTC] JSON send error: {e}")
            return False
    
    def set_message_handler(
        self,
        session_id: str,
        handler: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        """Set handler for incoming DataChannel messages."""
        self._on_message_handlers[session_id] = handler
    
    def is_connected(self, session_id: str) -> bool:
        """Check if session is connected and DataChannel is open."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        pc_connected = session.peer_connection.connectionState == "connected"
        dc_open = session.data_channel and session.data_channel.readyState == "open"
        
        return pc_connected and dc_open
    
    async def close_session(self, session_id: str) -> None:
        """Close and remove a WebRTC session."""
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()
            print(f"[{_now()}] [WebRTC] Session {session_id[:8]} closed")
        
        self._on_message_handlers.pop(session_id, None)
    
    async def close_all(self) -> None:
        """Close all sessions."""
        for session_id in list(self._sessions.keys()):
            await self.close_session(session_id)
    
    def get_session_stats(self, session_id: str) -> dict | None:
        """Get statistics for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "connection_state": session.peer_connection.connectionState,
            "ice_state": session.peer_connection.iceConnectionState,
            "data_channel_state": session.data_channel.readyState if session.data_channel else "none",
            "audio_chunks_sent": session.audio_chunks_sent,
            "uptime_seconds": time.time() - session.created_at,
        }


# Global instance
_webrtc_service: WebRTCService | None = None


def get_webrtc_service() -> WebRTCService:
    """Get the global WebRTC service instance."""
    global _webrtc_service
    if _webrtc_service is None:
        _webrtc_service = WebRTCService()
    return _webrtc_service
