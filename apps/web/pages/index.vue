<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useVoiceRecorder } from '~/composables/useVoiceRecorder'
import { useWebSocket } from '~/composables/useWebSocket'
import { useCharacter } from '~/composables/useCharacter'
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
}

const messages = ref<Message[]>([])
const isProcessing = ref(false)
const currentAudioUrl = ref<string | null>(null)
const isPlayingAudio = ref(false)

const { 
  isRecording, 
  currentTranscript,
  startRecording, 
  stopRecording 
} = useVoiceRecorder()
const { sendVoice, sendText, isConnected, onResponse, offResponse } = useWebSocket()
const { emotion, action, actionConfig, setEmotion, setAction } = useCharacter()

const handleMicClick = async () => {
  try {
    if (isRecording.value) {
      // Stop recording and get transcript
      const result = await stopRecording(false) // Don't upload to S3 for simplicity
      
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
      console.log('Starting recording...')
      try {
        await startRecording()
        setEmotion('listening')
        setAction('listen')
        console.log('Recording started, isRecording:', isRecording.value)
      } catch (e) {
        console.error('Failed to start recording:', e)
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

// Handle incoming messages from WebSocket
const handleAIResponse = (response: AIResponse) => {
  console.log('AI Response received:', response)
  
  // Determine the action from response
  const responseAction = (response.action || 'idle') as CharacterAction
  
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
  
  // Auto-play audio if available
  if (response.audioUrl) {
    currentAudioUrl.value = response.audioUrl
    isPlayingAudio.value = true
    setEmotion('speaking')
    // Keep the action during speaking, will return to idle after audio ends
  } else {
    setEmotion(response.emotion || 'idle')
  }
}

// Handle audio playback events
const handleAudioPlay = () => {
  isPlayingAudio.value = true
  setEmotion('speaking')
  // Keep current action but add speaking mouth animation
}

const handleAudioEnded = () => {
  isPlayingAudio.value = false
  currentAudioUrl.value = null
  setEmotion('idle')
  // Return to smile or idle after speaking
  setAction('smile')
  // After a brief smile, return to idle
  setTimeout(() => {
    if (!isProcessing.value && !isPlayingAudio.value) {
      setAction('idle')
    }
  }, 2000)
}

// Get status message based on current state
const statusMessage = computed(() => {
  if (isRecording.value) return '🎤 聞いています...'
  if (isProcessing.value) return '💭 考え中...'
  if (isPlayingAudio.value) return '💬 話しています...'
  return actionConfig.value?.labelJa || 'お話しましょう'
})

// Register WebSocket response handler
onMounted(() => {
  onResponse(handleAIResponse)
})

onUnmounted(() => {
  offResponse(handleAIResponse)
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
            :current-transcript="currentTranscript"
            @mic-click="handleMicClick"
            @submit="handleTextSubmit"
          />
        </div>
      </div>
    </main>
  </div>
</template>
