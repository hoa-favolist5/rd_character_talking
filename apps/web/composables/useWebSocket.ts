/**
 * WebSocket composable for real-time communication with the API.
 * 
 * NOW WITH STREAMING SUPPORT:
 * - Token streaming for live text display
 * - WebRTC for low-latency audio (optional)
 * - Falls back to base64 WebSocket if WebRTC unavailable
 * 
 * Performance improvement: ~300-500ms to first audio (vs ~1-2s before)
 */

import { ref, onMounted, onUnmounted } from 'vue'
import { io, Socket } from 'socket.io-client'

interface AIResponse {
  text: string
  audioUrl?: string
  emotion?: string
  action?: string
  contentType?: string
  userTranscript?: string
  audioStreaming?: boolean  // True if audio will be streamed separately
  streamingComplete?: boolean  // True when streaming is done
  firstChunkMs?: number  // Time to first audio chunk (ms)
  totalMs?: number  // Total processing time (ms)
  chunkCount?: number  // Number of audio chunks
}

interface AudioChunk {
  sentence: string
  audioUrl: string
  index: number
  isLast: boolean
  totalSentences?: number
}

interface TokenEvent {
  token: string
}

interface ThinkingEvent {
  status: string
  message: string
}

interface WaitingEvent {
  phraseIndex: number
  message: string
}

interface TakingLongEvent {
  audioUrl: string | null
  message: string
}

type ResponseHandler = (response: AIResponse) => void
type ThinkingHandler = (event: ThinkingEvent) => void
type AudioChunkHandler = (chunk: AudioChunk) => void
type TokenHandler = (token: string) => void
type WaitingHandler = (event: WaitingEvent) => void
type TakingLongHandler = (event: TakingLongEvent) => void
type WebRTCAudioHandler = (audioData: Uint8Array, chunkIndex: number, isLast: boolean) => void

export function useWebSocket() {
  const config = useRuntimeConfig()
  const socket = ref<Socket | null>(null)
  const isConnected = ref(false)
  const sessionId = ref<string>('')
  const isThinking = ref(false)
  const isWaiting = ref(false)  // Waiting for longer response
  const isStreaming = ref(false)  // True while receiving streaming response
  
  // WebRTC state
  const webrtcConnected = ref(false)
  const useWebRTC = ref(false)  // Whether to try WebRTC for audio
  
  const responseHandlers = ref<ResponseHandler[]>([])
  const thinkingHandlers = ref<ThinkingHandler[]>([])
  const audioChunkHandlers = ref<AudioChunkHandler[]>([])
  const tokenHandlers = ref<TokenHandler[]>([])
  const waitingHandlers = ref<WaitingHandler[]>([])
  const takingLongHandlers = ref<TakingLongHandler[]>([])
  const webrtcAudioHandlers = ref<WebRTCAudioHandler[]>([])

  const connect = () => {
    socket.value = io(config.public.wsUrl as string, {
      transports: ['websocket'],
      autoConnect: true,
    })

    socket.value.on('connect', () => {
      isConnected.value = true
      sessionId.value = socket.value?.id || crypto.randomUUID()
      console.log('[WS] Connected:', sessionId.value)
    })

    socket.value.on('disconnect', () => {
      isConnected.value = false
      webrtcConnected.value = false
      console.log('[WS] Disconnected')
    })

    // Handle "thinking" event for immediate feedback
    socket.value.on('thinking', (data: ThinkingEvent) => {
      console.log('[WS] Thinking:', data.message?.substring(0, 30))
      isThinking.value = true
      isWaiting.value = false
      isStreaming.value = true
      thinkingHandlers.value.forEach((handler) => handler(data))
    })

    // Handle "waiting" event for medium/long responses
    socket.value.on('waiting', (data: WaitingEvent) => {
      console.log('[WS] Waiting:', data.phraseIndex, data.message)
      isWaiting.value = true
      waitingHandlers.value.forEach((handler) => handler(data))
    })

    // Handle "taking_long" event (legacy)
    socket.value.on('taking_long', (data: TakingLongEvent) => {
      console.log('[WS] TakingLong:', data.message)
      isWaiting.value = true
      takingLongHandlers.value.forEach((handler) => handler(data))
    })

    // Handle streaming tokens (live text display)
    socket.value.on('token', (data: TokenEvent) => {
      tokenHandlers.value.forEach((handler) => handler(data.token))
    })

    // Handle response (includes streaming stats)
    socket.value.on('response', (data: AIResponse) => {
      isThinking.value = false
      isWaiting.value = false
      
      if (data.streamingComplete) {
        isStreaming.value = false
        console.log(`[WS] ✓ Stream complete: ${data.chunkCount} chunks, first@${data.firstChunkMs}ms, total=${data.totalMs}ms`)
      }
      
      responseHandlers.value.forEach((handler) => handler(data))
    })

    // Handle streaming audio chunks (WebSocket fallback)
    socket.value.on('audio_chunk', (data: AudioChunk) => {
      if (data.isLast) {
        console.log(`[WS] Audio complete: ${data.totalSentences} chunks`)
      } else {
        console.log(`[WS] Audio chunk ${data.index}: ${data.sentence?.substring(0, 20)}...`)
      }
      audioChunkHandlers.value.forEach((handler) => handler(data))
    })

    socket.value.on('error', (error: { message: string }) => {
      console.error('[WS] Error:', error.message)
      isThinking.value = false
      isStreaming.value = false
    })
  }

  const disconnect = () => {
    if (socket.value) {
      socket.value.disconnect()
      socket.value = null
    }
    webrtcConnected.value = false
  }

  /**
   * Send a text message with streaming response.
   * 
   * @param text - The text message to send
   * @param preferWebRTC - Whether to use WebRTC for audio delivery
   */
  const sendText = async (text: string, preferWebRTC: boolean = false): Promise<void> => {
    if (!socket.value || !isConnected.value) {
      console.error('[WS] Not connected')
      return
    }

    socket.value.emit('message', {
      type: 'text',
      content: text,
      sessionId: sessionId.value,
      useWebRTC: preferWebRTC && webrtcConnected.value,
    })
  }

  /**
   * Send a voice message with pre-transcribed text.
   * 
   * @param transcript - The transcribed text from frontend STT
   * @param preferWebRTC - Whether to use WebRTC for audio delivery
   */
  const sendVoice = async (transcript: string, preferWebRTC: boolean = false): Promise<void> => {
    if (!socket.value || !isConnected.value) {
      console.error('[WS] Not connected')
      return
    }

    if (!transcript.trim()) {
      console.warn('[WS] Empty transcript, not sending')
      return
    }

    socket.value.emit('message', {
      type: 'voice',
      transcript,
      sessionId: sessionId.value,
      useWebRTC: preferWebRTC && webrtcConnected.value,
    })
  }

  /**
   * Initialize WebRTC connection for low-latency audio.
   * Call this after Socket.IO connects if you want WebRTC audio.
   */
  const initWebRTC = async (): Promise<boolean> => {
    if (!socket.value || !isConnected.value) {
      console.error('[WebRTC] Socket not connected')
      return false
    }
    
    try {
      const { useWebRTC: webrtcComposable } = await import('./useWebRTC')
      const webrtc = webrtcComposable()
      
      const success = await webrtc.initWebRTC({
        socket: socket.value,
        sessionId: sessionId.value,
        onAudioChunk: (audioData, chunkIndex, isLast) => {
          webrtcAudioHandlers.value.forEach((handler) => handler(audioData, chunkIndex, isLast))
        },
        onConnected: () => {
          webrtcConnected.value = true
          console.log('[WebRTC] Connected - using low-latency audio')
        },
        onDisconnected: () => {
          webrtcConnected.value = false
          console.log('[WebRTC] Disconnected - falling back to WebSocket')
        },
        onError: (error) => {
          console.error('[WebRTC] Error:', error)
          webrtcConnected.value = false
        },
      })
      
      return success
    } catch (error) {
      console.error('[WebRTC] Init failed:', error)
      return false
    }
  }

  // Event handler registration
  const onResponse = (handler: ResponseHandler) => {
    responseHandlers.value.push(handler)
  }

  const offResponse = (handler: ResponseHandler) => {
    const index = responseHandlers.value.indexOf(handler)
    if (index > -1) responseHandlers.value.splice(index, 1)
  }

  const onThinking = (handler: ThinkingHandler) => {
    thinkingHandlers.value.push(handler)
  }

  const offThinking = (handler: ThinkingHandler) => {
    const index = thinkingHandlers.value.indexOf(handler)
    if (index > -1) thinkingHandlers.value.splice(index, 1)
  }

  const onAudioChunk = (handler: AudioChunkHandler) => {
    audioChunkHandlers.value.push(handler)
  }

  const offAudioChunk = (handler: AudioChunkHandler) => {
    const index = audioChunkHandlers.value.indexOf(handler)
    if (index > -1) audioChunkHandlers.value.splice(index, 1)
  }

  const onToken = (handler: TokenHandler) => {
    tokenHandlers.value.push(handler)
  }

  const offToken = (handler: TokenHandler) => {
    const index = tokenHandlers.value.indexOf(handler)
    if (index > -1) tokenHandlers.value.splice(index, 1)
  }

  const onWaiting = (handler: WaitingHandler) => {
    waitingHandlers.value.push(handler)
  }

  const offWaiting = (handler: WaitingHandler) => {
    const index = waitingHandlers.value.indexOf(handler)
    if (index > -1) waitingHandlers.value.splice(index, 1)
  }

  const onTakingLong = (handler: TakingLongHandler) => {
    takingLongHandlers.value.push(handler)
  }

  const offTakingLong = (handler: TakingLongHandler) => {
    const index = takingLongHandlers.value.indexOf(handler)
    if (index > -1) takingLongHandlers.value.splice(index, 1)
  }

  const onWebRTCAudio = (handler: WebRTCAudioHandler) => {
    webrtcAudioHandlers.value.push(handler)
  }

  const offWebRTCAudio = (handler: WebRTCAudioHandler) => {
    const index = webrtcAudioHandlers.value.indexOf(handler)
    if (index > -1) webrtcAudioHandlers.value.splice(index, 1)
  }

  /**
   * Get the raw Socket.IO instance for advanced usage.
   */
  const getSocket = (): Socket | null => socket.value

  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    // State
    isConnected,
    isThinking,
    isWaiting,
    isStreaming,
    webrtcConnected,
    sessionId,
    
    // Methods
    sendText,
    sendVoice,
    initWebRTC,
    getSocket,
    connect,
    disconnect,
    
    // Event handlers
    onResponse,
    offResponse,
    onThinking,
    offThinking,
    onAudioChunk,
    offAudioChunk,
    onToken,
    offToken,
    onWaiting,
    offWaiting,
    onTakingLong,
    offTakingLong,
    onWebRTCAudio,
    offWebRTCAudio,
  }
}
