<script setup lang="ts">
import { computed, ref, watch, onMounted } from 'vue'
import type { CharacterAction, ActionConfig } from '~/composables/useCharacter'
import { ACTION_CONFIGS } from '~/composables/useCharacter'
import { DotLottieVue } from '@lottiefiles/dotlottie-vue'

/**
 * Map CharacterAction to lottie file name
 * Available files: idle, happy, talk, listen, thinking, excited, surprised, confused, scared, cry, wave, cheer
 */
const ACTION_TO_LOTTIE: Record<CharacterAction, string> = {
  // Basic expressions
  idle: 'idle',
  smile: 'happy',
  laugh: 'happy',
  grin: 'happy',
  // Sad/Sympathetic
  sad: 'cry',
  cry: 'cry',
  sympathetic: 'cry',
  comfort: 'happy',
  // Curious/Thinking
  curious: 'thinking',
  thinking: 'thinking',
  confused: 'confused',
  wonder: 'surprised',
  // Surprise/Excitement
  surprised: 'surprised',
  shocked: 'surprised',
  excited: 'excited',
  amazed: 'excited',
  // Scared/Nervous
  scared: 'scared',
  nervous: 'scared',
  worried: 'confused',
  // Affection/Romance
  blush: 'happy',
  love: 'happy',
  shy: 'happy',
  wink: 'happy',
  // Agreement/Gestures
  nod: 'happy',
  shake_head: 'confused',
  thumbs_up: 'cheer',
  // Speaking/Listening
  speak: 'talk',
  listen: 'listen',
  explain: 'talk',
  // Special
  wave: 'wave',
  bow: 'idle',
  celebrate: 'cheer',
  cheer: 'cheer',
}

interface Props {
  action?: CharacterAction
  size?: 'sm' | 'md' | 'lg'
}

const props = withDefaults(defineProps<Props>(), {
  action: 'idle',
  size: 'lg',
})

const dotLottieRef = ref<InstanceType<typeof DotLottieVue> | null>(null)

// Get lottie file URL based on current action
const lottieFile = computed(() => {
  const fileName = ACTION_TO_LOTTIE[props.action] || 'idle'
  return `/character/${fileName}.lottie`
})

const sizeClasses = computed(() => {
  const sizes = {
    sm: 'w-32 h-32',
    md: 'w-48 h-48',
    lg: 'w-72 h-72',
  }
  return sizes[props.size]
})

const config = computed<ActionConfig>(() => ACTION_CONFIGS[props.action] || ACTION_CONFIGS.idle)

// Control animation speed based on action
const animationSpeed = computed(() => {
  const speedMap: Partial<Record<CharacterAction, number>> = {
    excited: 1.5,
    celebrate: 1.4,
    cheer: 1.3,
    laugh: 1.2,
    thinking: 0.6,
    sad: 0.5,
    idle: 0.8,
    listen: 0.9,
    speak: 1.1,
  }
  return speedMap[props.action] ?? 1
})

// Control animation based on action state
const shouldPause = computed(() => {
  return props.action === 'bow'
})

// Watch for action changes to control animation
watch(() => props.action, (newAction) => {
  const lottie = dotLottieRef.value?.getDotLottieInstance()
  if (lottie) {
    lottie.setSpeed(animationSpeed.value)
    
    if (shouldPause.value) {
      lottie.pause()
    } else {
      lottie.play()
    }
  }
})

onMounted(() => {
  const lottie = dotLottieRef.value?.getDotLottieInstance()
  if (lottie) {
    lottie.setSpeed(animationSpeed.value)
  }
})

// Glow style based on current action
const glowStyle = computed(() => ({
  filter: `drop-shadow(0 0 30px ${config.value.glowColor}) drop-shadow(0 0 60px ${config.value.glowColor})`,
}))

// Container animation class
const containerAnimationClass = computed(() => {
  const classes: string[] = []
  
  if (props.action === 'excited' || props.action === 'celebrate' || props.action === 'cheer') {
    classes.push('animate-bounce-gentle')
  }
  if (props.action === 'scared' || props.action === 'nervous') {
    classes.push('animate-shake-slow')
  }
  if (props.action === 'nod') {
    classes.push('animate-nod-head')
  }
  if (props.action === 'shake_head') {
    classes.push('animate-shake-head')
  }
  
  return classes.join(' ')
})
</script>

<template>
  <div
    class="character-avatar relative flex items-center justify-center transition-all duration-500"
    :class="sizeClasses"
  >
    <!-- Lottie Animation Container -->
    <div
      class="w-full h-full relative transition-all duration-300"
      :class="containerAnimationClass"
      :style="glowStyle"
    >
      <!-- Background glow effect -->
      <div
        class="absolute inset-0 rounded-full blur-2xl opacity-40 transition-colors duration-500"
        :style="{ backgroundColor: config.glowColor }"
      />
      
      <!-- Lottie Character - uses different animation based on action -->
      <DotLottieVue
        ref="dotLottieRef"
        :src="lottieFile"
        autoplay
        loop
        class="w-full h-full relative z-10"
      />

      <!-- Overlay Effects -->
      <!-- Sparkles effect -->
      <template v-if="config.sparkles">
        <div class="absolute -top-2 -left-2 text-2xl animate-sparkle-1 z-20">✨</div>
        <div class="absolute -top-6 right-6 text-xl animate-sparkle-2 z-20">✨</div>
        <div class="absolute top-2 -right-2 text-2xl animate-sparkle-3 z-20">✨</div>
        <div class="absolute bottom-8 -left-4 text-lg animate-sparkle-2 z-20">✨</div>
      </template>

      <!-- Hearts effect -->
      <template v-if="config.hearts">
        <div class="absolute -top-4 left-1/4 text-2xl animate-float-up z-20">💕</div>
        <div class="absolute -top-2 right-1/4 text-xl animate-float-up delay-300 z-20">💗</div>
        <div class="absolute top-8 -right-4 text-lg animate-float-up delay-500 z-20">💖</div>
      </template>

      <!-- Exclamation effect -->
      <template v-if="config.exclamation">
        <div class="absolute -top-8 left-1/2 -translate-x-1/2 text-4xl animate-bounce font-bold text-yellow-400 z-20">
          ❗
        </div>
      </template>

      <!-- Sweat drop -->
      <div
        v-if="config.sweat"
        class="absolute top-6 right-6 w-4 h-5 rounded-full bg-gradient-to-b from-blue-300 to-blue-400 animate-sweat z-20"
        style="clip-path: polygon(50% 0%, 100% 60%, 50% 100%, 0% 60%);"
      />

      <!-- Tears -->
      <template v-if="config.tears">
        <div class="absolute top-1/3 left-1/4 w-2 h-8 rounded-full bg-blue-400/80 animate-tear-fall z-20" />
        <div class="absolute top-1/3 right-1/4 w-2 h-8 rounded-full bg-blue-400/80 animate-tear-fall delay-200 z-20" />
      </template>

      <!-- Blush effect -->
      <template v-if="config.blush">
        <div class="absolute top-1/2 left-4 w-8 h-6 rounded-full bg-pink-400/40 blur-md animate-pulse-slow z-10" />
        <div class="absolute top-1/2 right-4 w-8 h-6 rounded-full bg-pink-400/40 blur-md animate-pulse-slow z-10" />
      </template>
    </div>
  </div>
</template>

<style scoped>
.character-avatar {
  transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Gentle bounce animation */
@keyframes bounce-gentle {
  0%, 100% { transform: translateY(0) scale(1); }
  50% { transform: translateY(-12px) scale(1.03); }
}

/* Slow shake animation */
@keyframes shake-slow {
  0%, 100% { transform: translateX(0); }
  20% { transform: translateX(-4px) rotate(-1deg); }
  40% { transform: translateX(4px) rotate(1deg); }
  60% { transform: translateX(-3px) rotate(-0.5deg); }
  80% { transform: translateX(3px) rotate(0.5deg); }
}

/* Nod head animation */
@keyframes nod-head {
  0%, 100% { transform: translateY(0) rotate(0deg); }
  30% { transform: translateY(6px) rotate(5deg); }
  60% { transform: translateY(3px) rotate(2deg); }
}

/* Shake head animation */
@keyframes shake-head {
  0%, 100% { transform: rotate(0deg); }
  25% { transform: rotate(-8deg); }
  75% { transform: rotate(8deg); }
}

/* Pulse slow */
@keyframes pulse-slow {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.1); }
}

/* Sparkle animations */
@keyframes sparkle-1 {
  0%, 100% { opacity: 0; transform: scale(0.5) rotate(0deg); }
  50% { opacity: 1; transform: scale(1.2) rotate(180deg); }
}

@keyframes sparkle-2 {
  0%, 100% { opacity: 0; transform: scale(0.5) rotate(0deg); }
  60% { opacity: 1; transform: scale(1.3) rotate(180deg); }
}

@keyframes sparkle-3 {
  0%, 100% { opacity: 0; transform: scale(0.5) rotate(0deg); }
  40% { opacity: 1; transform: scale(1.1) rotate(180deg); }
}

/* Float up animation */
@keyframes float-up {
  0% { transform: translateY(0) scale(1); opacity: 1; }
  100% { transform: translateY(-40px) scale(1.3); opacity: 0; }
}

/* Tear fall animation */
@keyframes tear-fall {
  0% { transform: translateY(0); opacity: 0.8; }
  100% { transform: translateY(30px); opacity: 0; }
}

/* Sweat animation */
@keyframes sweat {
  0% { transform: translateY(0) scale(1); opacity: 0.7; }
  50% { transform: translateY(8px) scale(0.85); opacity: 0.9; }
  100% { transform: translateY(20px) scale(0.6); opacity: 0; }
}

/* Animation classes */
.animate-bounce-gentle { animation: bounce-gentle 1.2s ease-in-out infinite; }
.animate-shake-slow { animation: shake-slow 0.6s ease-in-out infinite; }
.animate-nod-head { animation: nod-head 0.8s ease-in-out infinite; }
.animate-shake-head { animation: shake-head 0.5s ease-in-out infinite; }
.animate-pulse-slow { animation: pulse-slow 3s ease-in-out infinite; }
.animate-sparkle-1 { animation: sparkle-1 1.5s ease-in-out infinite; }
.animate-sparkle-2 { animation: sparkle-2 1.5s ease-in-out infinite 0.4s; }
.animate-sparkle-3 { animation: sparkle-3 1.5s ease-in-out infinite 0.8s; }
.animate-float-up { animation: float-up 2.5s ease-out infinite; }
.animate-tear-fall { animation: tear-fall 1.2s ease-in-out infinite; }
.animate-sweat { animation: sweat 2.5s ease-in-out infinite; }

.delay-200 { animation-delay: 0.2s; }
.delay-300 { animation-delay: 0.3s; }
.delay-500 { animation-delay: 0.5s; }
</style>
