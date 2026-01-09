<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useVoiceRecorder } from '~/composables/useVoiceRecorder'
import { useWebSocket } from '~/composables/useWebSocket'
import { useCharacter } from '~/composables/useCharacter'
import { useAudioQueue } from '~/composables/useAudioQueue'
import { useWaitingAudio } from '~/composables/useWaitingAudio'
import type { CharacterAction } from '~/composables/useCharacter'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  audioUrl?: string
  action?: CharacterAction
  contentType?: string
  timestamp: Date
}

interface AIResponse {
  text: string
  audioUrl?: string
  emotion?: string
  action?: string
  contentType?: string
  audioStreaming?: boolean
  streamingComplete?: boolean  // True when all audio chunks have been sent
}

interface AudioChunk {
  sentence: string
  audioUrl: string
  index: number
  isLast: boolean
  totalSentences?: number
}

const messages = ref<Message[]>([])
const isProcessing = ref(false)
const currentAudioUrl = ref<string | null>(null)
const isPlayingAudio = ref(false)

// Legacy audio queue refs (kept for non-streaming fallback)
const audioQueue = ref<string[]>([])
const isPlayingQueue = ref(false)

// Conversation mode - continuous listening after AI responds
const isConversationMode = ref(false)
const shouldRestartListening = ref(false)

const { 
  isRecording, 
  isSpeaking,
  volumeLevel,
  currentTranscript,
  startRecording, 
  stopRecording 
} = useVoiceRecorder()
const { 
  sendVoice, 
  sendText, 
  isConnected, 
  isThinking, 
  isWaiting,
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
} = useWebSocket()
const { emotion, action, actionConfig, setEmotion, setAction } = useCharacter()

// Pre-load waiting audio files (saves bandwidth - audio is local)
const { playWaitingAudio: playLocalWaitingAudio } = useWaitingAudio()

// Waiting audio element for quick playback (legacy/fallback)
const waitingAudio = ref<HTMLAudioElement | null>(null)

// Smooth audio queue with Web Audio API (gapless playback)
const {
  isPlaying: isAudioQueuePlaying,
  isBuffering: isAudioBuffering,
  addChunk: addAudioChunk,
  markStreamComplete,
  reset: resetAudioQueue,
  initAudioContext,
} = useAudioQueue({
  minBuffer: 1, // Start playing immediately when first chunk arrives
  onPlay: () => {
    console.log('[AudioQueue] Playback started')
    isPlayingAudio.value = true
    setEmotion('speaking')
  },
  onEnded: () => {
    console.log('[AudioQueue] Playback ended')
    handleStreamingAudioEnded()
  },
  onChunkStart: (index) => {
    console.log('[AudioQueue] Playing chunk', index)
  },
})

/**
 * Handle silence detection - auto-stop recording and send message
 */
const handleSilenceDetected = async () => {
  console.log('[Conversation] handleSilenceDetected called, isRecording:', isRecording.value, 'isPlayingAudio:', isPlayingAudio.value)
  
  // Don't process if audio is playing (would pick up AI's voice)
  if (isPlayingAudio.value || isPlayingQueue.value) {
    console.log('[Conversation] Audio is playing, ignoring silence detection')
    return
  }
  
  if (!isRecording.value) {
    console.log('[Conversation] Not recording, skipping')
    return
  }
  
  console.log('[Conversation] Silence detected, stopping recording...')
  
  // Stop recording and get transcript
  const result = await stopRecording(false)
  
  console.log('[Conversation] Recording stopped, transcript:', result.transcript)
  
  if (result.transcript && result.transcript.trim()) {
    // Add user message to chat
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      content: result.transcript,
      timestamp: new Date(),
    })

    isProcessing.value = true
    setEmotion('thinking')
    setAction('thinking')
    
    // Mark that we should restart listening after AI responds
    if (isConversationMode.value) {
      shouldRestartListening.value = true
      console.log('[Conversation] Will restart listening after AI response')
    }
    
    // Send transcribed text to API
    console.log('[Conversation] Sending to API:', result.transcript)
    await sendVoice(result.transcript, result.s3Key)
  } else {
    console.log('[Conversation] No transcript, restarting listening...')
    // No transcript - restart listening in conversation mode
    if (isConversationMode.value) {
      setTimeout(() => {
        if (isConversationMode.value && !isRecording.value && !isProcessing.value) {
          startConversationListening()
        }
      }, 500)
    }
  }
}

/**
 * Start listening with silence detection for conversation mode
 * Uses transcript-based detection: when user finishes speaking (final transcript)
 * and no new speech for silenceDuration, triggers the callback.
 */
const startConversationListening = async () => {
  // CRITICAL: Don't start listening while audio is playing
  if (isPlayingAudio.value || isPlayingQueue.value || isAudioQueuePlaying.value) {
    console.log('[Conversation] Cannot start listening - audio is still playing')
    return
  }
  
  console.log('[Conversation] Starting listening with transcript-based silence detection...')
  try {
    await startRecording({
      silenceDuration: 1000,  // 1.0 seconds after final transcript with no new speech
      onSilenceDetected: handleSilenceDetected,
    })
    setEmotion('listening')
    setAction('listen')
    console.log('[Conversation] Listening started successfully')
  } catch (e) {
    console.error('[Conversation] Failed to start listening:', e)
    isConversationMode.value = false
  }
}

const handleMicClick = async () => {
  // Initialize Web Audio context on user interaction (required by browsers)
  initAudioContext()
  
  try {
    if (isConversationMode.value) {
      // End conversation mode
      console.log('Ending conversation mode...')
      isConversationMode.value = false
      shouldRestartListening.value = false
      
      if (isRecording.value) {
        await stopRecording(false)
      }
      
      setEmotion('idle')
      setAction('idle')
      return
    }
    
    if (isRecording.value) {
      // Stop recording and get transcript (manual stop)
      const result = await stopRecording(false)
      
      console.log('Recording stopped, transcript:', result.transcript)
      
      if (result.transcript) {
        // Add user message to chat
        messages.value.push({
          id: crypto.randomUUID(),
          role: 'user',
          content: result.transcript,
          timestamp: new Date(),
        })

        isProcessing.value = true
        setEmotion('thinking')
        setAction('thinking')
        
        // Send transcribed text to API
        await sendVoice(result.transcript, result.s3Key)
      } else {
        // No transcript - maybe too short or failed
        console.log('No transcript available')
        setEmotion('idle')
        setAction('idle')
      }
    } else {
      // Start conversation mode with silence detection
      console.log('Starting conversation mode...')
      isConversationMode.value = true
      
      try {
        await startConversationListening()
      } catch (e) {
        console.error('Failed to start recording:', e)
        isConversationMode.value = false
        alert('マイクへのアクセスが許可されていません。ブラウザとシステムの設定を確認してください。')
        setEmotion('idle')
        setAction('idle')
      }
    }
  } catch (error) {
    console.error('Mic click error:', error)
    setEmotion('idle')
    setAction('idle')
    isProcessing.value = false
    isConversationMode.value = false
  }
}

const handleTextSubmit = async (text: string) => {
  if (!text.trim()) return

  messages.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    content: text,
    timestamp: new Date(),
  })

  isProcessing.value = true
  setEmotion('thinking')
  setAction('thinking')
  await sendText(text)
}

// Track if we've added the assistant message for current stream
let currentStreamMessageId: string | null = null

// Handle incoming messages from WebSocket
const handleAIResponse = async (response: AIResponse) => {
  console.log('[Conversation] AI Response received:', response)
  
  // Check if this is the final "streaming complete" notification
  if (response.streamingComplete) {
    console.log('[Conversation] Streaming complete, audio playing:', isAudioQueuePlaying.value)
    isProcessing.value = false
    
    // Update the message text if we have the full text now
    if (currentStreamMessageId && response.text) {
      const msg = messages.value.find(m => m.id === currentStreamMessageId)
      if (msg) {
        msg.content = response.text
      }
    }
    currentStreamMessageId = null
    
    // Audio is already playing via audio_chunk events, let it finish naturally
    // The onEnded callback in useAudioQueue will trigger handleStreamingAudioEnded
    return
  }
  
  // CRITICAL: Stop any recording to prevent picking up AI's voice
  if (isRecording.value) {
    console.log('[Conversation] Stopping recording before AI speaks...')
    await stopRecording(false)  // Don't process transcript
  }
  
  // Determine the action from response
  const responseAction = (response.action || 'idle') as CharacterAction
  
  // Check if audio will be streamed separately (initial streaming response)
  if (response.audioStreaming && !response.streamingComplete) {
    console.log('[Conversation] Audio will be streamed, setting up...')
    
    // Add message placeholder (will be updated when streaming completes)
    const msgId = crypto.randomUUID()
    currentStreamMessageId = msgId
    messages.value.push({
      id: msgId,
      role: 'assistant',
      content: response.text || '...',
      audioUrl: response.audioUrl,
      action: responseAction,
      contentType: response.contentType,
      timestamp: new Date(),
    })
    
    isProcessing.value = false
    setAction(responseAction)
    setEmotion('listening') // Show buffering state until audio starts
    isPlayingAudio.value = true
    
    // Reset audio queue for new stream
    resetAudioQueue()
    return
  }
  
  // Non-streaming response - add message normally
  messages.value.push({
    id: crypto.randomUUID(),
    role: 'assistant',
    content: response.text,
    audioUrl: response.audioUrl,
    action: responseAction,
    contentType: response.contentType,
    timestamp: new Date(),
  })
  isProcessing.value = false
  
  // Set the action from the response
  setAction(responseAction)
  
  // Auto-play audio if available (non-streaming mode)
  if (response.audioUrl) {
    currentAudioUrl.value = response.audioUrl
    isPlayingAudio.value = true
    setEmotion('speaking')
  } else {
    // No audio URL - handle conversation mode restart manually
    console.log('[Conversation] No audio URL in response')
    setEmotion(response.emotion || 'idle')
    
    // If in conversation mode, restart listening after a short delay
    if (shouldRestartListening.value && isConversationMode.value) {
      shouldRestartListening.value = false
      console.log('[Conversation] No audio, restarting listening...')
      setTimeout(() => {
        if (isConversationMode.value && !isRecording.value && !isProcessing.value) {
          startConversationListening()
        }
      }, 500)
    }
  }
}

/**
 * Handle streaming audio chunks - add to Web Audio queue with pre-buffering
 */
const handleAudioChunk = (chunk: AudioChunk) => {
  console.log('[Audio Stream] Received chunk:', chunk.index, chunk.isLast ? '(last)' : '')
  
  if (chunk.isLast) {
    // Last chunk received - mark stream as complete
    console.log('[Audio Stream] All', chunk.totalSentences, 'chunks received')
    markStreamComplete(chunk.totalSentences || 0)
    return
  }
  
  // Add to Web Audio queue (will pre-buffer before playing)
  addAudioChunk(chunk.audioUrl, chunk.index)
}

/**
 * Handle when streaming audio playback completes
 */
const handleStreamingAudioEnded = () => {
  console.log('[Conversation] Streaming audio ended, conversation mode:', isConversationMode.value)
  
  isPlayingAudio.value = false
  isPlayingQueue.value = false
  
  // In conversation mode, restart listening after audio ends
  if (isConversationMode.value) {
    console.log('[Conversation] Will restart mic in 500ms...')
    
    setTimeout(() => {
      const canStart = isConversationMode.value && 
                       !isRecording.value && 
                       !isProcessing.value && 
                       !isPlayingAudio.value &&
                       !isAudioQueuePlaying.value
      
      console.log('[Conversation] Restart check:', { 
        canStart, 
        mode: isConversationMode.value, 
        recording: isRecording.value,
        processing: isProcessing.value,
        playing: isPlayingAudio.value,
        queuePlaying: isAudioQueuePlaying.value
      })
      
      if (canStart) {
        console.log('[Conversation] Restarting listening now!')
        startConversationListening()
      } else {
        console.log('[Conversation] Cannot restart - conditions not met')
      }
    }, 500)
  } else {
    setEmotion('idle')
    setAction('smile')
    setTimeout(() => {
      if (!isProcessing.value && !isPlayingAudio.value) {
        setAction('idle')
      }
    }, 2000)
  }
}

// Handle audio playback events
const handleAudioPlay = () => {
  isPlayingAudio.value = true
  setEmotion('speaking')
  // Keep current action but add speaking mouth animation
}

/**
 * Handle legacy single-audio playback ended (non-streaming fallback)
 */
const handleAudioEnded = () => {
  console.log('[Conversation] Legacy audio ended')
  
  currentAudioUrl.value = null
  isPlayingAudio.value = false
  isPlayingQueue.value = false
  shouldRestartListening.value = false
  
  // In conversation mode, restart listening after audio ends
  if (isConversationMode.value) {
    console.log('[Conversation] Will restart mic in 800ms...')
    
    setTimeout(() => {
      const canStart = isConversationMode.value && 
                       !isRecording.value && 
                       !isProcessing.value && 
                       !isPlayingAudio.value &&
                       !isAudioQueuePlaying.value
      
      if (canStart) {
        startConversationListening()
      }
    }, 800)
  } else {
    setEmotion('idle')
    setAction('smile')
    setTimeout(() => {
      if (!isProcessing.value && !isPlayingAudio.value) {
        setAction('idle')
      }
    }, 2000)
  }
}

// Get status message based on current state
const statusMessage = computed(() => {
  if (isRecording.value && isSpeaking.value) return '🎤 聞いています...'
  if (isRecording.value) return '🎤 お話しください...'
  if (isWaiting.value) return '⏳ ちょっと待ってね...'
  if (isThinking.value) return '💭 AIが考え中...'
  if (isProcessing.value) return '💭 処理中...'
  if (isAudioBuffering.value) return '⏳ 音声準備中...'
  if (isPlayingAudio.value || isAudioQueuePlaying.value) return '💬 話しています...'
  if (isConversationMode.value) return '💬 会話モード'
  return actionConfig.value?.labelJa || 'お話しましょう'
})

// Handle thinking event - immediate feedback while AI processes
const handleThinking = () => {
  console.log('[Conversation] Server is processing...')
  // Could add a "typing" indicator or show a quick acknowledgment
  setAction('thinking')
}

/**
 * Handle waiting event - play pre-loaded local audio for medium/long responses
 * Backend sends phraseIndex, frontend plays local audio file (saves bandwidth)
 */
interface WaitingEvent {
  phraseIndex: number  // Index into local audio files (0-3)
  message: string
}

interface TakingLongEvent {
  audioUrl: string | null
  message: string
}

/**
 * Handle waiting event - play local audio file by phraseIndex
 */
const handleWaiting = (event: WaitingEvent) => {
  console.log('[Conversation] Waiting event received:', event.phraseIndex, event.message)
  
  // Set visual feedback
  setEmotion('speaking')
  setAction('smile')
  
  // Play pre-loaded local audio (no streaming needed!)
  playLocalWaitingAudio(event.phraseIndex)
  
  // Reset emotion after approximate audio duration (1.5s)
  setTimeout(() => {
    setEmotion('thinking')
  }, 1500)
}

/**
 * Handle taking_long event - legacy fallback (no longer used)
 * Kept for backwards compatibility
 */
const handleTakingLong = (event: TakingLongEvent) => {
  console.log('[Conversation] TakingLong event received:', event.message)
  
  // Legacy: Play from URL if provided
  if (event.audioUrl) {
    // Clean up previous waiting audio
    if (waitingAudio.value) {
      waitingAudio.value.pause()
      waitingAudio.value = null
    }
    
    const audio = new Audio(event.audioUrl)
    audio.volume = 0.8
    waitingAudio.value = audio
    
    setEmotion('speaking')
    setAction('smile')
    
    audio.play().catch(e => {
      console.warn('[Conversation] Failed to play TakingLong audio:', e)
    })
    
    audio.onended = () => {
      waitingAudio.value = null
      setEmotion('thinking')
    }
  }
}

// Register WebSocket response handlers
onMounted(() => {
  onThinking(handleThinking)
  onWaiting(handleWaiting)
  onTakingLong(handleTakingLong)
  onResponse(handleAIResponse)
  onAudioChunk(handleAudioChunk)
})

onUnmounted(() => {
  offThinking(handleThinking)
  offWaiting(handleWaiting)
  offTakingLong(handleTakingLong)
  offResponse(handleAIResponse)
  offAudioChunk(handleAudioChunk)
  
  // Clean up waiting audio
  if (waitingAudio.value) {
    waitingAudio.value.pause()
    waitingAudio.value = null
  }
})
</script>

<template>
  <div class="flex flex-col h-screen">
    <!-- Header -->
    <header class="p-4 border-b border-white/10">
      <div class="max-w-4xl mx-auto flex items-center justify-between">
        <h1 class="text-2xl font-display font-bold bg-gradient-to-r from-accent-sakura to-primary-400 bg-clip-text text-transparent">
          AI キャラクター
        </h1>
        <div class="flex items-center gap-2">
          <span
            class="w-3 h-3 rounded-full"
            :class="isConnected ? 'bg-green-400' : 'bg-red-400'"
          />
          <span class="text-sm text-white/60">
            {{ isConnected ? '接続中' : '未接続' }}
          </span>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 flex max-w-6xl mx-auto w-full gap-6 p-6 overflow-hidden">
      <!-- Character Avatar Section -->
      <div class="w-1/3 flex flex-col items-center justify-center">
        <div class="glass-panel p-8 w-full aspect-square flex items-center justify-center">
          <CharacterAvatar :action="action" />
        </div>
        <p class="mt-4 text-center text-white/60 text-sm flex items-center gap-2 justify-center">
          {{ statusMessage }}
        </p>
        
        <!-- Auto-play Voice Player -->
        <div v-if="currentAudioUrl" class="mt-4 w-full">
          <VoicePlayer
            :src="currentAudioUrl"
            :autoplay="true"
            @play="handleAudioPlay"
            @ended="handleAudioEnded"
          />
        </div>
      </div>

      <!-- Chat Section -->
      <div class="flex-1 flex flex-col glass-panel">
        <!-- Messages -->
        <ChatWindow :messages="messages" class="flex-1" />

        <!-- Input Area -->
        <div class="p-4 border-t border-white/10">
          <VoiceInput
            :is-recording="isRecording"
            :is-processing="isProcessing"
            :is-conversation-mode="isConversationMode"
            :is-speaking="isSpeaking"
            :volume-level="volumeLevel"
            :current-transcript="currentTranscript"
            @mic-click="handleMicClick"
            @submit="handleTextSubmit"
          />
        </div>
      </div>
    </main>
  </div>
</template>
