<script setup lang="ts">
import { computed } from 'vue'
import type { CharacterEmotion } from '~/composables/useCharacter'

interface Props {
  emotion?: CharacterEmotion
  size?: 'sm' | 'md' | 'lg'
}

const props = withDefaults(defineProps<Props>(), {
  emotion: 'idle',
  size: 'lg',
})

const sizeClasses = computed(() => {
  const sizes = {
    sm: 'w-24 h-24',
    md: 'w-40 h-40',
    lg: 'w-64 h-64',
  }
  return sizes[props.size]
})

const emotionClass = computed(() => {
  return `emotion-${props.emotion}`
})

// Animation states for the character
const isAnimating = computed(() => {
  return ['listening', 'thinking', 'speaking'].includes(props.emotion)
})
</script>

<template>
  <div
    class="character-avatar relative flex items-center justify-center"
    :class="[sizeClasses, emotionClass]"
  >
    <!-- Placeholder avatar - replace with Lottie animation -->
    <div
      class="w-full h-full rounded-full bg-gradient-to-br from-accent-sakura/50 to-primary-500/50 flex items-center justify-center overflow-hidden"
      :class="{ 'animate-pulse-soft': isAnimating }"
    >
      <!-- Character face placeholder -->
      <div class="relative w-3/4 h-3/4">
        <!-- Eyes -->
        <div class="absolute top-1/4 left-1/4 w-1/4 h-1/4 flex gap-4">
          <div
            class="w-6 h-6 rounded-full bg-white shadow-inner transition-all duration-300"
            :class="{
              'animate-pulse': emotion === 'listening',
              'scale-125': emotion === 'surprised',
              'scale-75': emotion === 'happy',
            }"
          >
            <div
              class="w-3 h-3 rounded-full bg-slate-800 mt-1.5 ml-1.5 transition-all duration-300"
              :class="{
                'animate-bounce': emotion === 'thinking',
              }"
            />
          </div>
          <div
            class="w-6 h-6 rounded-full bg-white shadow-inner transition-all duration-300"
            :class="{
              'animate-pulse': emotion === 'listening',
              'scale-125': emotion === 'surprised',
              'scale-75': emotion === 'happy',
            }"
          >
            <div
              class="w-3 h-3 rounded-full bg-slate-800 mt-1.5 ml-1.5 transition-all duration-300"
              :class="{
                'animate-bounce': emotion === 'thinking',
              }"
            />
          </div>
        </div>

        <!-- Mouth -->
        <div
          class="absolute bottom-1/4 left-1/2 -translate-x-1/2 transition-all duration-300"
          :class="{
            'w-8 h-4 rounded-full bg-pink-400': emotion === 'happy',
            'w-4 h-4 rounded-full bg-pink-300': emotion === 'surprised',
            'w-6 h-2 rounded-full bg-pink-300': emotion === 'idle',
            'w-8 h-6 rounded-full bg-pink-300 animate-pulse': emotion === 'speaking',
            'w-4 h-1 rounded-full bg-pink-300': emotion === 'thinking',
            'w-6 h-3 rounded-b-full bg-pink-300': emotion === 'sad',
          }"
        />
      </div>
    </div>

    <!-- Emotion indicator -->
    <div
      v-if="isAnimating"
      class="absolute -bottom-2 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full text-xs font-medium backdrop-blur-sm"
      :class="{
        'bg-blue-500/50': emotion === 'listening',
        'bg-purple-500/50': emotion === 'thinking',
        'bg-green-500/50': emotion === 'speaking',
      }"
    >
      {{
        emotion === 'listening' ? '🎤' :
        emotion === 'thinking' ? '💭' :
        emotion === 'speaking' ? '💬' : ''
      }}
    </div>
  </div>
</template>

<style scoped>
.character-avatar {
  transition: all 0.3s ease;
}

.emotion-listening {
  filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.5));
}

.emotion-thinking {
  filter: drop-shadow(0 0 20px rgba(147, 51, 234, 0.5));
}

.emotion-speaking {
  filter: drop-shadow(0 0 20px rgba(34, 197, 94, 0.5));
}

.emotion-happy {
  filter: drop-shadow(0 0 20px rgba(255, 183, 197, 0.5));
}
</style>

