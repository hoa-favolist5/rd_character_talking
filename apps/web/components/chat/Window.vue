<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  audioUrl?: string
  timestamp: Date
}

interface Props {
  messages: Message[]
}

const props = defineProps<Props>()

const chatContainer = ref<HTMLDivElement | null>(null)

// Auto-scroll to bottom when new messages arrive
watch(
  () => props.messages.length,
  async () => {
    await nextTick()
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  }
)
</script>

<template>
  <div
    ref="chatContainer"
    class="flex-1 overflow-y-auto p-4 space-y-4"
  >
    <!-- Empty state -->
    <div
      v-if="messages.length === 0"
      class="h-full flex flex-col items-center justify-center text-white/40"
    >
      <svg
        class="w-16 h-16 mb-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="1.5"
          d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
        />
      </svg>
      <p class="text-lg font-display">話しかけてみてください</p>
      <p class="text-sm mt-2">マイクボタンを押すか、テキストを入力してください</p>
    </div>

    <!-- Messages -->
    <ChatBubble
      v-for="message in messages"
      :key="message.id"
      :message="message"
    />
  </div>
</template>

