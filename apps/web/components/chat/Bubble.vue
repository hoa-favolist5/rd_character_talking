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

interface FormattedLine {
  type: 'text' | 'bullet' | 'numbered' | 'header'
  content: string
  number?: number
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

// Parse content into structured lines for beautiful rendering
const formattedContent = computed((): FormattedLine[] => {
  const content = props.message.content
  const lines = content.split('\n')
  const result: FormattedLine[] = []
  
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue
    
    // Check for bullet points (•, -, *, ・, ▪, ▸)
    const bulletMatch = trimmed.match(/^[•\-\*・▪▸]\s*(.+)/)
    if (bulletMatch) {
      result.push({ type: 'bullet', content: bulletMatch[1] })
      continue
    }
    
    // Check for numbered lists (1. 2. etc or ①②③ etc)
    const numberedMatch = trimmed.match(/^(\d+)[.）\)]\s*(.+)/)
    if (numberedMatch) {
      result.push({ type: 'numbered', content: numberedMatch[2], number: parseInt(numberedMatch[1]) })
      continue
    }
    
    // Check for circled numbers (①②③...)
    const circledMatch = trimmed.match(/^([①②③④⑤⑥⑦⑧⑨⑩])\s*(.+)/)
    if (circledMatch) {
      const circledNumbers = '①②③④⑤⑥⑦⑧⑨⑩'
      const num = circledNumbers.indexOf(circledMatch[1]) + 1
      result.push({ type: 'numbered', content: circledMatch[2], number: num })
      continue
    }
    
    // Check for headers/titles (【】or []）
    const headerMatch = trimmed.match(/^[【\[](.+?)[】\]](.*)/)
    if (headerMatch) {
      result.push({ type: 'header', content: headerMatch[1] + (headerMatch[2] ? ': ' + headerMatch[2] : '') })
      continue
    }
    
    // Regular text
    result.push({ type: 'text', content: trimmed })
  }
  
  return result
})

// Check if content has structured formatting
const hasFormatting = computed(() => {
  return formattedContent.value.some(line => line.type !== 'text')
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
      <!-- Message content - Simple text for user or unformatted content -->
      <p v-if="isUser || !hasFormatting" class="text-white/90 leading-relaxed">
        {{ message.content }}
      </p>

      <!-- Formatted content for AI with structure -->
      <div v-else class="text-white/90 leading-relaxed space-y-2">
        <template v-for="(line, index) in formattedContent" :key="index">
          <!-- Header -->
          <div v-if="line.type === 'header'" class="font-semibold text-accent-sakura/90 border-b border-white/10 pb-1 mb-2">
            {{ line.content }}
          </div>
          
          <!-- Bullet point -->
          <div v-else-if="line.type === 'bullet'" class="flex items-start gap-2 pl-1">
            <span class="text-accent-sakura mt-0.5 flex-shrink-0">•</span>
            <span>{{ line.content }}</span>
          </div>
          
          <!-- Numbered list -->
          <div v-else-if="line.type === 'numbered'" class="flex items-start gap-2 pl-1">
            <span class="w-5 h-5 rounded-full bg-primary-500/30 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
              {{ line.number }}
            </span>
            <span>{{ line.content }}</span>
          </div>
          
          <!-- Regular text -->
          <p v-else>{{ line.content }}</p>
        </template>
      </div>

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

