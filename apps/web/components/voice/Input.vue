<script setup lang="ts">
import { ref } from 'vue'

interface Props {
  isRecording: boolean
  isProcessing: boolean
  currentTranscript?: string
}

interface Emits {
  (e: 'mic-click'): void
  (e: 'submit', text: string): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const textInput = ref('')

const handleSubmit = () => {
  if (textInput.value.trim()) {
    emit('submit', textInput.value)
    textInput.value = ''
  }
}

const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSubmit()
  }
}
</script>

<template>
  <div class="flex flex-col gap-3 w-full">
    <!-- Real-time Transcript Display (when recording) -->
    <div
      v-if="isRecording && currentTranscript"
      class="px-4 py-2 bg-primary-500/10 border border-primary-400/30 rounded-lg"
    >
      <div class="flex items-center gap-2 mb-1">
        <span class="flex h-2 w-2">
          <span class="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-primary-400 opacity-75"></span>
          <span class="relative inline-flex rounded-full h-2 w-2 bg-primary-500"></span>
        </span>
        <span class="text-xs text-primary-400 font-medium">リアルタイム音声認識</span>
      </div>
      <p class="text-white/90 text-sm">{{ currentTranscript }}</p>
    </div>

    <!-- Recording indicator (no transcript yet) -->
    <div
      v-else-if="isRecording"
      class="px-4 py-2 bg-primary-500/10 border border-primary-400/30 rounded-lg"
    >
      <div class="flex items-center gap-2">
        <span class="flex h-2 w-2">
          <span class="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-primary-400 opacity-75"></span>
          <span class="relative inline-flex rounded-full h-2 w-2 bg-primary-500"></span>
        </span>
        <span class="text-sm text-primary-400">音声を聞いています...</span>
      </div>
    </div>

    <!-- Input Area -->
    <div class="flex items-center gap-4 w-full">
      <!-- Text Input -->
      <div class="flex-1 relative">
        <input
          v-model="textInput"
          type="text"
          placeholder="メッセージを入力..."
          class="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-full text-white placeholder-white/40 focus:outline-none focus:border-primary-400/50 focus:ring-2 focus:ring-primary-400/20 transition-all"
          :disabled="isRecording || isProcessing"
          @keydown="handleKeydown"
        >
        
        <!-- Send button -->
        <button
          v-if="textInput.trim() && !isRecording"
          class="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-primary-500 hover:bg-primary-600 flex items-center justify-center transition-colors"
          :disabled="isProcessing"
          @click="handleSubmit"
        >
          <svg
            class="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
            />
          </svg>
        </button>
      </div>

      <!-- Microphone Button -->
      <button
        class="btn-mic"
        :class="{ 'recording': isRecording }"
        :disabled="isProcessing"
        @click="$emit('mic-click')"
      >
        <!-- Microphone Icon -->
        <svg
          v-if="!isRecording && !isProcessing"
          class="w-8 h-8 text-white"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
          />
        </svg>

        <!-- Stop Icon (when recording) -->
        <svg
          v-else-if="isRecording"
          class="w-8 h-8 text-white"
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>

        <!-- Loading Spinner (when processing) -->
        <svg
          v-else
          class="w-8 h-8 text-white animate-spin"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            class="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            stroke-width="4"
          />
          <path
            class="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.btn-mic:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}
</style>
