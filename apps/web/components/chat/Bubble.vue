<script setup lang="ts">
import { ref, computed } from 'vue'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  audioUrl?: string
  timestamp: Date
}

interface Props {
  message: Message
}

const props = defineProps<Props>()

const isPlaying = ref(false)
const audioRef = ref<HTMLAudioElement | null>(null)

const isUser = computed(() => props.message.role === 'user')

const formattedTime = computed(() => {
  return new Intl.DateTimeFormat('ja-JP', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(props.message.timestamp)
})

const playAudio = () => {
  if (!props.message.audioUrl || !audioRef.value) return
  
  if (isPlaying.value) {
    audioRef.value.pause()
    audioRef.value.currentTime = 0
    isPlaying.value = false
  } else {
    audioRef.value.play()
    isPlaying.value = true
  }
}

const onAudioEnded = () => {
  isPlaying.value = false
}
</script>

<template>
  <div
    class="flex"
    :class="isUser ? 'justify-end' : 'justify-start'"
  >
    <div
      class="chat-bubble"
      :class="isUser ? 'chat-bubble-user' : 'chat-bubble-ai'"
    >
      <!-- Message content -->
      <p class="text-white/90 leading-relaxed">
        {{ message.content }}
      </p>

      <!-- Audio player (for AI responses) -->
      <div
        v-if="message.audioUrl"
        class="mt-3 flex items-center gap-2"
      >
        <button
          class="w-8 h-8 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-colors"
          @click="playAudio"
        >
          <svg
            v-if="!isPlaying"
            class="w-4 h-4"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M8 5v14l11-7z" />
          </svg>
          <svg
            v-else
            class="w-4 h-4"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
          </svg>
        </button>
        <span class="text-xs text-white/50">音声再生</span>
        <audio
          ref="audioRef"
          :src="message.audioUrl"
          @ended="onAudioEnded"
        />
      </div>

      <!-- Timestamp -->
      <p class="text-xs text-white/40 mt-2">
        {{ formattedTime }}
      </p>
    </div>
  </div>
</template>

