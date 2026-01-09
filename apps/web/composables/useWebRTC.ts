/**
 * WebRTC composable for low-latency audio streaming.
 * 
 * Provides ~50-100ms lower latency than base64-over-WebSocket by:
 * - Using native WebRTC DataChannel for audio delivery
 * - Eliminating base64 encoding/decoding overhead
 * - Better congestion control
 * 
 * Usage:
 * 1. Call initWebRTC() after Socket.IO connects
 * 2. If successful, set useWebRTC: true in message payload
 * 3. Audio chunks arrive via onAudioChunk callback
 */

import { ref, onUnmounted } from 'vue'
import type { Socket } from 'socket.io-client'

interface WebRTCConfig {
  socket: Socket
  sessionId: string
  onAudioChunk?: (audioData: Uint8Array, chunkIndex: number, isLast: boolean) => void
  onConnected?: () => void
  onDisconnected?: () => void
  onError?: (error: Error) => void
}

// STUN servers for NAT traversal
const ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
]

export function useWebRTC() {
  const isConnected = ref(false)
  const isConnecting = ref(false)
  const connectionState = ref<RTCPeerConnectionState>('new')
  
  let peerConnection: RTCPeerConnection | null = null
  let dataChannel: RTCDataChannel | null = null
  let config: WebRTCConfig | null = null
  
  // Audio queue for received chunks
  const audioQueue: { data: Uint8Array; index: number }[] = []
  let isProcessingQueue = false
  
  /**
   * Initialize WebRTC connection.
   * 
   * @param cfg Configuration with socket, session ID, and callbacks
   * @returns Promise that resolves when connected, or rejects on error
   */
  const initWebRTC = async (cfg: WebRTCConfig): Promise<boolean> => {
    if (isConnected.value || isConnecting.value) {
      console.log('[WebRTC] Already connected or connecting')
      return isConnected.value
    }
    
    config = cfg
    isConnecting.value = true
    
    try {
      console.log('[WebRTC] Initializing...')
      
      // Create peer connection
      peerConnection = new RTCPeerConnection({
        iceServers: ICE_SERVERS,
      })
      
      // Handle connection state changes
      peerConnection.onconnectionstatechange = () => {
        connectionState.value = peerConnection?.connectionState || 'new'
        console.log('[WebRTC] Connection state:', connectionState.value)
        
        if (connectionState.value === 'connected') {
          isConnected.value = true
          isConnecting.value = false
          config?.onConnected?.()
        } else if (connectionState.value === 'failed' || connectionState.value === 'closed') {
          isConnected.value = false
          isConnecting.value = false
          config?.onDisconnected?.()
        }
      }
      
      // Handle ICE candidates - send to server
      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          console.log('[WebRTC] Sending ICE candidate')
          config?.socket.emit('webrtc_ice', {
            candidate: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid,
            sdpMLineIndex: event.candidate.sdpMLineIndex,
            sessionId: config?.sessionId,
          })
        }
      }
      
      // Create data channel for audio
      dataChannel = peerConnection.createDataChannel('audio', {
        ordered: true,  // Ensure chunks arrive in order
      })
      
      dataChannel.binaryType = 'arraybuffer'
      
      dataChannel.onopen = () => {
        console.log('[WebRTC] DataChannel opened')
        isConnected.value = true
        isConnecting.value = false
      }
      
      dataChannel.onclose = () => {
        console.log('[WebRTC] DataChannel closed')
        isConnected.value = false
      }
      
      dataChannel.onerror = (error) => {
        console.error('[WebRTC] DataChannel error:', error)
        config?.onError?.(new Error('DataChannel error'))
      }
      
      // Handle incoming audio data
      dataChannel.onmessage = (event) => {
        handleIncomingData(event.data)
      }
      
      // Create offer
      const offer = await peerConnection.createOffer()
      await peerConnection.setLocalDescription(offer)
      
      console.log('[WebRTC] Created offer, sending to server')
      
      // Send offer via Socket.IO
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('WebRTC connection timeout'))
          cleanup()
        }, 10000)  // 10 second timeout
        
        // Handle answer from server
        const handleAnswer = async (data: { sdp: string; type: RTCSdpType }) => {
          try {
            console.log('[WebRTC] Received answer from server')
            
            const answer = new RTCSessionDescription({
              sdp: data.sdp,
              type: data.type,
            })
            
            await peerConnection?.setRemoteDescription(answer)
            
            clearTimeout(timeout)
            config?.socket.off('webrtc_answer', handleAnswer)
            
            // Wait for connection to establish
            if (isConnected.value) {
              resolve(true)
            } else {
              // Wait for connection state change
              const checkConnection = setInterval(() => {
                if (isConnected.value) {
                  clearInterval(checkConnection)
                  resolve(true)
                } else if (connectionState.value === 'failed') {
                  clearInterval(checkConnection)
                  reject(new Error('Connection failed'))
                }
              }, 100)
              
              // Timeout for connection
              setTimeout(() => {
                clearInterval(checkConnection)
                if (!isConnected.value) {
                  reject(new Error('Connection timeout'))
                }
              }, 5000)
            }
          } catch (error) {
            clearTimeout(timeout)
            reject(error)
          }
        }
        
        // Handle ICE candidates from server
        const handleIce = async (data: { candidate: string; sdpMid: string; sdpMLineIndex: number }) => {
          try {
            if (peerConnection && data.candidate) {
              await peerConnection.addIceCandidate(new RTCIceCandidate({
                candidate: data.candidate,
                sdpMid: data.sdpMid,
                sdpMLineIndex: data.sdpMLineIndex,
              }))
            }
          } catch (error) {
            console.error('[WebRTC] Failed to add ICE candidate:', error)
          }
        }
        
        config?.socket.on('webrtc_answer', handleAnswer)
        config?.socket.on('webrtc_ice', handleIce)
        
        // Send offer
        config?.socket.emit('webrtc_offer', {
          sdp: offer.sdp,
          type: offer.type,
          sessionId: config?.sessionId,
        })
      })
      
    } catch (error) {
      console.error('[WebRTC] Init error:', error)
      isConnecting.value = false
      config?.onError?.(error instanceof Error ? error : new Error(String(error)))
      cleanup()
      return false
    }
  }
  
  /**
   * Handle incoming binary data from DataChannel.
   * 
   * Format: [4 bytes: chunk_index][1 byte: is_last][audio_data]
   */
  const handleIncomingData = (data: ArrayBuffer | string) => {
    if (typeof data === 'string') {
      // JSON message (metadata)
      try {
        const json = JSON.parse(data)
        if (json.type === 'audio_complete') {
          console.log('[WebRTC] Audio complete, total chunks:', json.totalChunks)
        }
      } catch (e) {
        console.error('[WebRTC] Invalid JSON:', e)
      }
      return
    }
    
    // Binary audio data
    const buffer = new Uint8Array(data)
    
    if (buffer.length < 5) {
      console.error('[WebRTC] Invalid audio chunk: too small')
      return
    }
    
    // Parse header
    const view = new DataView(data)
    const chunkIndex = view.getUint32(0)
    const isLast = view.getUint8(4) === 1
    
    // Extract audio data (skip 5-byte header)
    const audioData = buffer.slice(5)
    
    console.log(`[WebRTC] Received chunk ${chunkIndex}: ${audioData.length} bytes`)
    
    // Call callback
    config?.onAudioChunk?.(audioData, chunkIndex, isLast)
  }
  
  /**
   * Send a message via DataChannel.
   */
  const sendMessage = (data: object): boolean => {
    if (!dataChannel || dataChannel.readyState !== 'open') {
      return false
    }
    
    try {
      dataChannel.send(JSON.stringify(data))
      return true
    } catch (error) {
      console.error('[WebRTC] Send error:', error)
      return false
    }
  }
  
  /**
   * Clean up WebRTC connection.
   */
  const cleanup = () => {
    if (dataChannel) {
      dataChannel.close()
      dataChannel = null
    }
    
    if (peerConnection) {
      peerConnection.close()
      peerConnection = null
    }
    
    isConnected.value = false
    isConnecting.value = false
    connectionState.value = 'closed'
  }
  
  /**
   * Close WebRTC connection gracefully.
   */
  const close = () => {
    if (config?.socket) {
      config.socket.emit('webrtc_close', {
        sessionId: config.sessionId,
      })
    }
    
    cleanup()
  }
  
  // Cleanup on unmount
  onUnmounted(() => {
    close()
  })
  
  return {
    // State
    isConnected,
    isConnecting,
    connectionState,
    
    // Methods
    initWebRTC,
    sendMessage,
    close,
    cleanup,
  }
}
