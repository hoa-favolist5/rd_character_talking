<script setup lang="ts">
import { ref, watch, onUnmounted, nextTick } from 'vue'

interface Props {
  src: string | null
  autoplay?: boolean
}

interface Emits {
  (e: 'play'): void
  (e: 'pause'): void
  (e: 'ended'): void
}

const props = withDefaults(defineProps<Props>(), {
  autoplay: false,
})

const emit = defineEmits<Emits>()

const audioRef = ref<HTMLAudioElement | null>(null)
const isPlaying = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const progress = ref(0)

watch(
  () => props.src,
  async (newSrc) => {
    if (newSrc && props.autoplay) {
      // Wait for next tick to ensure audio element is updated
      await nextTick()
      // Small delay to ensure audio is loaded
      setTimeout(() => {
        play()
      }, 100)
    }
  },
  { immediate: true }
)

const play = async () => {
  if (!audioRef.value) return
  try {
    await audioRef.value.play()
    isPlaying.value = true
    emit('play')
  } catch (error) {
    console.error('Failed to play audio:', error)
  }
}

const pause = () => {
  if (!audioRef.value) return
  audioRef.value.pause()
  isPlaying.value = false
  emit('pause')
}

const togglePlay = () => {
  if (isPlaying.value) {
    pause()
  } else {
    play()
  }
}

const onTimeUpdate = () => {
  if (!audioRef.value) return
  currentTime.value = audioRef.value.currentTime
  progress.value = (currentTime.value / duration.value) * 100
}

const onLoadedMetadata = () => {
  if (!audioRef.value) return
  duration.value = audioRef.value.duration
  
  // Auto-play when metadata is loaded (if autoplay is enabled)
  if (props.autoplay && !isPlaying.value) {
    play()
  }
}

const onEnded = () => {
  isPlaying.value = false
  progress.value = 0
  emit('ended')
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

onUnmounted(() => {
  if (audioRef.value) {
    audioRef.value.pause()
  }
})

defineExpose({ play, pause })
</script>

<template>
  <div
    v-if="src"
    class="flex items-center gap-3 p-3 bg-white/10 rounded-xl"
  >
    <!-- Play/Pause Button -->
    <button
      class="w-10 h-10 rounded-full bg-primary-500 hover:bg-primary-600 flex items-center justify-center transition-colors"
      @click="togglePlay"
    >
      <svg
        v-if="!isPlaying"
        class="w-5 h-5 text-white ml-0.5"
        fill="currentColor"
        viewBox="0 0 24 24"
      >
        <path d="M8 5v14l11-7z" />
      </svg>
      <svg
        v-else
        class="w-5 h-5 text-white"
        fill="currentColor"
        viewBox="0 0 24 24"
      >
        <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
      </svg>
    </button>

    <!-- Progress Bar -->
    <div class="flex-1">
      <div class="h-1 bg-white/20 rounded-full overflow-hidden">
        <div
          class="h-full bg-primary-400 transition-all duration-100"
          :style="{ width: `${progress}%` }"
        />
      </div>
      <div class="flex justify-between mt-1 text-xs text-white/50">
        <span>{{ formatTime(currentTime) }}</span>
        <span>{{ formatTime(duration) }}</span>
      </div>
    </div>

    <!-- Hidden Audio Element -->
    <audio
      ref="audioRef"
      :src="src"
      @timeupdate="onTimeUpdate"
      @loadedmetadata="onLoadedMetadata"
      @ended="onEnded"
    />
  </div>
</template>

