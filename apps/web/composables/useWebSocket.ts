/**
 * WebSocket composable for real-time communication with the API.
 * 
 * Updated to send pre-transcribed text for voice messages instead of
 * raw audio data. The frontend now handles transcription via AWS Transcribe.
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
  responseLength?: 'short' | 'medium' | 'long'  // Response length category
}

interface AudioChunk {
  sentence: string
  audioUrl: string
  index: number
  isLast: boolean
  totalSentences?: number
}

interface ThinkingEvent {
  status: string
  message: string
}

interface WaitingEvent {
  phraseIndex: number  // Index into pre-loaded waiting audio files (0-3)
  message: string
}

interface TakingLongEvent {
  audioUrl: string | null
  message: string
}

type ResponseHandler = (response: AIResponse) => void
type ThinkingHandler = (event: ThinkingEvent) => void
type AudioChunkHandler = (chunk: AudioChunk) => void
type WaitingHandler = (event: WaitingEvent) => void
type TakingLongHandler = (event: TakingLongEvent) => void

export function useWebSocket() {
  const config = useRuntimeConfig()
  const socket = ref<Socket | null>(null)
  const isConnected = ref(false)
  const sessionId = ref<string>('')
  const isThinking = ref(false)
  const isWaiting = ref(false)  // New: waiting for longer response
  
  const responseHandlers = ref<ResponseHandler[]>([])
  const thinkingHandlers = ref<ThinkingHandler[]>([])
  const audioChunkHandlers = ref<AudioChunkHandler[]>([])
  const waitingHandlers = ref<WaitingHandler[]>([])
  const takingLongHandlers = ref<TakingLongHandler[]>([])

  const connect = () => {
    socket.value = io(config.public.wsUrl as string, {
      transports: ['websocket'],
      autoConnect: true,
    })

    socket.value.on('connect', () => {
      isConnected.value = true
      sessionId.value = socket.value?.id || crypto.randomUUID()
      console.log('WebSocket connected:', sessionId.value)
    })

    socket.value.on('disconnect', () => {
      isConnected.value = false
      console.log('WebSocket disconnected')
    })

    // Handle "thinking" event for immediate feedback
    socket.value.on('thinking', (data: ThinkingEvent) => {
      console.log('[WS] Received thinking event:', data)
      isThinking.value = true
      isWaiting.value = false
      thinkingHandlers.value.forEach((handler) => handler(data))
    })

    // Handle "waiting" event for medium/long responses
    // Backend sends phraseIndex, frontend plays pre-loaded audio from /audio/waiting/{index}.mp3
    // Files: 0.mp3="ちょっと待ってね", 1.mp3="えーっと、ちょっと待って", 
    //        2.mp3="うーんと、待ってね", 3.mp3="少し待ってね"
    socket.value.on('waiting', (data: WaitingEvent) => {
      console.log('[WS] Received waiting event:', data.phraseIndex, data.message)
      isWaiting.value = true
      waitingHandlers.value.forEach((handler) => handler(data))
    })

    // Handle "taking_long" event when processing takes > 3 seconds
    // Plays a waiting audio to let user know we're still working
    socket.value.on('taking_long', (data: TakingLongEvent) => {
      console.log('[WS] Received taking_long event:', data.message)
      isWaiting.value = true
      takingLongHandlers.value.forEach((handler) => handler(data))
    })

    socket.value.on('response', (data: AIResponse) => {
      console.log('[WS] Received response:', data, 'length:', data.responseLength)
      isThinking.value = false
      isWaiting.value = false
      responseHandlers.value.forEach((handler) => handler(data))
    })

    // Handle streaming audio chunks
    socket.value.on('audio_chunk', (data: AudioChunk) => {
      console.log('[WS] Received audio chunk:', data.index, data.isLast ? '(last)' : '')
      audioChunkHandlers.value.forEach((handler) => handler(data))
    })

    socket.value.on('error', (error: { message: string }) => {
      console.error('WebSocket error:', error.message)
      isThinking.value = false
    })
  }

  const disconnect = () => {
    if (socket.value) {
      socket.value.disconnect()
      socket.value = null
    }
  }

  /**
   * Send a text message
   */
  const sendText = async (text: string): Promise<void> => {
    if (!socket.value || !isConnected.value) {
      console.error('WebSocket not connected')
      return
    }

    socket.value.emit('message', {
      type: 'text',
      content: text,
      sessionId: sessionId.value,
    })
  }

  /**
   * Send a voice message with pre-transcribed text
   * 
   * @param transcript - The transcribed text from frontend STT
   * @param s3Key - Optional S3 key where the audio was uploaded
   */
  const sendVoice = async (transcript: string, s3Key?: string): Promise<void> => {
    if (!socket.value || !isConnected.value) {
      console.error('WebSocket not connected')
      return
    }

    if (!transcript.trim()) {
      console.warn('Empty transcript, not sending')
      return
    }

    socket.value.emit('message', {
      type: 'voice',
      transcript,
      s3Key,
      sessionId: sessionId.value,
    })
  }

  /**
   * @deprecated Use sendVoice instead. Audio is now transcribed on frontend.
   */
  const sendAudio = async (audioBlob: Blob): Promise<void> => {
    console.warn('sendAudio is deprecated. Use sendVoice with pre-transcribed text.')
    // Convert to base64 for backwards compatibility (if needed)
    const arrayBuffer = await audioBlob.arrayBuffer()
    const base64Audio = btoa(
      new Uint8Array(arrayBuffer).reduce(
        (data, byte) => data + String.fromCharCode(byte),
        ''
      )
    )

    socket.value?.emit('message', {
      type: 'audio',
      content: base64Audio,
      mimeType: audioBlob.type,
      sessionId: sessionId.value,
    })
  }

  const onResponse = (handler: ResponseHandler) => {
    responseHandlers.value.push(handler)
  }

  const offResponse = (handler: ResponseHandler) => {
    const index = responseHandlers.value.indexOf(handler)
    if (index > -1) {
      responseHandlers.value.splice(index, 1)
    }
  }

  const onThinking = (handler: ThinkingHandler) => {
    thinkingHandlers.value.push(handler)
  }

  const offThinking = (handler: ThinkingHandler) => {
    const index = thinkingHandlers.value.indexOf(handler)
    if (index > -1) {
      thinkingHandlers.value.splice(index, 1)
    }
  }

  const onAudioChunk = (handler: AudioChunkHandler) => {
    audioChunkHandlers.value.push(handler)
  }

  const offAudioChunk = (handler: AudioChunkHandler) => {
    const index = audioChunkHandlers.value.indexOf(handler)
    if (index > -1) {
      audioChunkHandlers.value.splice(index, 1)
    }
  }

  const onWaiting = (handler: WaitingHandler) => {
    waitingHandlers.value.push(handler)
  }

  const offWaiting = (handler: WaitingHandler) => {
    const index = waitingHandlers.value.indexOf(handler)
    if (index > -1) {
      waitingHandlers.value.splice(index, 1)
    }
  }

  const onTakingLong = (handler: TakingLongHandler) => {
    takingLongHandlers.value.push(handler)
  }

  const offTakingLong = (handler: TakingLongHandler) => {
    const index = takingLongHandlers.value.indexOf(handler)
    if (index > -1) {
      takingLongHandlers.value.splice(index, 1)
    }
  }

  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    isConnected,
    isThinking,
    isWaiting,
    sessionId,
    sendText,
    sendVoice,
    sendAudio, // Keep for backwards compatibility
    onResponse,
    offResponse,
    onThinking,
    offThinking,
    onAudioChunk,
    offAudioChunk,
    onWaiting,
    offWaiting,
    onTakingLong,
    offTakingLong,
    connect,
    disconnect,
  }
}
