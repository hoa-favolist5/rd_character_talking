<script setup lang="ts">
import { computed } from 'vue'
import type { CharacterAction, ActionConfig } from '~/composables/useCharacter'
import { ACTION_CONFIGS } from '~/composables/useCharacter'

interface Props {
  action?: CharacterAction
  size?: 'sm' | 'md' | 'lg'
}

const props = withDefaults(defineProps<Props>(), {
  action: 'idle',
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

const config = computed<ActionConfig>(() => ACTION_CONFIGS[props.action] || ACTION_CONFIGS.idle)

// Animation states for the character
const isAnimating = computed(() => {
  return ['listen', 'thinking', 'speak', 'explain'].includes(props.action)
})

// Eye styles based on action
const eyeClasses = computed(() => {
  const base = 'transition-all duration-300 rounded-full bg-white shadow-inner'
  const animations: Record<string, string> = {
    'squint': 'scale-y-50',
    'wide': 'scale-125',
    'closed': 'scale-y-10',
    'shake': 'animate-shake',
    'sparkle': 'animate-sparkle',
    'wink': '', // Handled separately
    'heart': '', // Handled separately
    'droop': 'translate-y-1',
    'look-up': '-translate-y-1',
    'look-away': 'translate-x-1',
    'focus': 'scale-110',
    'nod': 'animate-nod',
    'dart': 'animate-dart',
  }
  return `${base} ${animations[config.value.eyeAnimation] || ''}`
})

// Pupil animation
const pupilClasses = computed(() => {
  const base = 'transition-all duration-300 rounded-full bg-slate-800'
  const animations: Record<string, string> = {
    'look-up': '-translate-y-0.5',
    'look-away': 'translate-x-0.5',
    'spiral': 'animate-spin',
    'focus': 'scale-110',
  }
  return `${base} ${animations[config.value.eyeAnimation] || ''}`
})

// Mouth styles
const mouthClasses = computed(() => {
  const styles: Record<string, string> = {
    'smile': 'w-10 h-5 rounded-b-full bg-pink-400',
    'open': 'w-6 h-6 rounded-full bg-pink-400 animate-talk',
    'small': 'w-5 h-2 rounded-full bg-pink-300',
    'wide': 'w-12 h-8 rounded-full bg-pink-400',
    'sad': 'w-8 h-4 rounded-t-full bg-pink-300 rotate-180',
    'o': 'w-6 h-6 rounded-full bg-pink-400',
    'teeth': 'w-10 h-6 rounded-lg bg-pink-400 border-t-4 border-white',
  }
  return styles[config.value.mouthStyle] || styles.small
})

const mouthAnimationClass = computed(() => {
  const animations: Record<string, string> = {
    'bounce': 'animate-bounce-soft',
    'wobble': 'animate-wobble',
    'shake': 'animate-shake',
    'talk': 'animate-talk',
  }
  return animations[config.value.mouthAnimation] || ''
})

// Glow style
const glowStyle = computed(() => ({
  filter: `drop-shadow(0 0 25px ${config.value.glowColor})`,
}))
</script>

<template>
  <div
    class="character-avatar relative flex items-center justify-center transition-all duration-300"
    :class="sizeClasses"
    :style="glowStyle"
  >
    <!-- Main avatar container -->
    <div
      class="w-full h-full rounded-full bg-gradient-to-br from-accent-sakura/60 to-primary-500/60 flex items-center justify-center overflow-hidden relative"
      :class="{ 
        'animate-pulse-soft': isAnimating,
        'animate-bounce-gentle': action === 'excited' || action === 'celebrate',
        'animate-shake-slow': action === 'scared' || action === 'nervous',
      }"
    >
      <!-- Blush effect -->
      <div
        v-if="config.blush"
        class="absolute top-1/2 left-1/4 w-6 h-4 rounded-full bg-pink-400/50 blur-sm animate-pulse-slow"
      />
      <div
        v-if="config.blush"
        class="absolute top-1/2 right-1/4 w-6 h-4 rounded-full bg-pink-400/50 blur-sm animate-pulse-slow"
      />

      <!-- Tears -->
      <div
        v-if="config.tears"
        class="absolute top-1/3 left-1/3 w-2 h-6 rounded-full bg-blue-400/70 animate-tear-fall"
      />
      <div
        v-if="config.tears"
        class="absolute top-1/3 right-1/3 w-2 h-6 rounded-full bg-blue-400/70 animate-tear-fall delay-100"
      />

      <!-- Sweat drop -->
      <div
        v-if="config.sweat"
        class="absolute top-1/4 right-1/4 w-3 h-4 rounded-full bg-blue-300/60 animate-sweat"
      />

      <!-- Character face -->
      <div class="relative w-3/4 h-3/4">
        <!-- Eyes container -->
        <div class="absolute top-1/4 left-0 right-0 flex justify-center gap-6">
          <!-- Left Eye -->
          <div
            class="w-8 h-8"
            :class="eyeClasses"
            :style="{ transform: `scale(${config.eyeScale})` }"
          >
            <!-- Heart eyes -->
            <template v-if="config.eyeAnimation === 'heart'">
              <div class="w-full h-full flex items-center justify-center text-red-500 text-xl animate-pulse">
                ❤️
              </div>
            </template>
            <!-- Star eyes -->
            <template v-else-if="config.eyeAnimation === 'star'">
              <div class="w-full h-full flex items-center justify-center text-yellow-400 text-xl animate-spin-slow">
                ⭐
              </div>
            </template>
            <!-- Wink (left eye) -->
            <template v-else-if="config.eyeAnimation === 'wink'">
              <div class="w-6 h-1 bg-slate-800 rounded-full mt-3 ml-1" />
            </template>
            <!-- Normal pupil -->
            <template v-else>
              <div
                class="w-4 h-4 mt-2 ml-2"
                :class="pupilClasses"
              />
            </template>
          </div>

          <!-- Right Eye -->
          <div
            class="w-8 h-8"
            :class="eyeClasses"
            :style="{ transform: `scale(${config.eyeScale})` }"
          >
            <!-- Heart eyes -->
            <template v-if="config.eyeAnimation === 'heart'">
              <div class="w-full h-full flex items-center justify-center text-red-500 text-xl animate-pulse">
                ❤️
              </div>
            </template>
            <!-- Star eyes -->
            <template v-else-if="config.eyeAnimation === 'star'">
              <div class="w-full h-full flex items-center justify-center text-yellow-400 text-xl animate-spin-slow">
                ⭐
              </div>
            </template>
            <!-- Normal pupil (right eye stays open for wink) -->
            <template v-else>
              <div
                class="w-4 h-4 mt-2 ml-2"
                :class="pupilClasses"
              />
            </template>
          </div>
        </div>

        <!-- Mouth -->
        <div
          class="absolute bottom-1/4 left-1/2 -translate-x-1/2 transition-all duration-300"
          :class="[mouthClasses, mouthAnimationClass]"
        />
      </div>
    </div>

    <!-- Sparkles effect -->
    <template v-if="config.sparkles">
      <div class="absolute -top-2 -left-2 text-2xl animate-sparkle-1">✨</div>
      <div class="absolute -top-4 right-4 text-xl animate-sparkle-2">✨</div>
      <div class="absolute top-0 -right-2 text-2xl animate-sparkle-3">✨</div>
    </template>

    <!-- Hearts effect -->
    <template v-if="config.hearts">
      <div class="absolute -top-4 left-1/4 text-2xl animate-float-up">💕</div>
      <div class="absolute -top-2 right-1/4 text-xl animate-float-up delay-300">💗</div>
    </template>

    <!-- Exclamation effect -->
    <template v-if="config.exclamation">
      <div class="absolute -top-6 left-1/2 -translate-x-1/2 text-3xl animate-bounce font-bold text-yellow-400">
        ❗
      </div>
    </template>

    <!-- Action indicator badge -->
    <div
      class="absolute -bottom-3 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full text-sm font-medium backdrop-blur-md border border-white/20 flex items-center gap-2 shadow-lg"
      :class="config.bgColor"
    >
      <span class="text-lg">{{ config.emoji }}</span>
      <span :class="config.color">{{ config.labelJa }}</span>
    </div>
  </div>
</template>

<style scoped>
.character-avatar {
  transition: all 0.3s ease;
}

/* Soft animations */
@keyframes pulse-soft {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.02); }
}

@keyframes pulse-slow {
  0%, 100% { opacity: 0.5; }
  50% { opacity: 0.8; }
}

@keyframes bounce-soft {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-3px); }
}

@keyframes bounce-gentle {
  0%, 100% { transform: translateY(0) scale(1); }
  50% { transform: translateY(-8px) scale(1.02); }
}

@keyframes shake-slow {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-3px); }
  75% { transform: translateX(3px); }
}

@keyframes wobble {
  0%, 100% { transform: rotate(0deg); }
  25% { transform: rotate(-3deg); }
  75% { transform: rotate(3deg); }
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  10%, 30%, 50%, 70%, 90% { transform: translateX(-2px); }
  20%, 40%, 60%, 80% { transform: translateX(2px); }
}

@keyframes talk {
  0%, 100% { transform: scaleY(1); }
  50% { transform: scaleY(0.7); }
}

@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes tear-fall {
  0% { transform: translateY(0); opacity: 0.7; }
  100% { transform: translateY(20px); opacity: 0; }
}

@keyframes sweat {
  0% { transform: translateY(0) scale(1); opacity: 0.6; }
  50% { transform: translateY(5px) scale(0.8); opacity: 0.8; }
  100% { transform: translateY(15px) scale(0.5); opacity: 0; }
}

@keyframes sparkle-1 {
  0%, 100% { opacity: 0; transform: scale(0.5) rotate(0deg); }
  50% { opacity: 1; transform: scale(1) rotate(180deg); }
}

@keyframes sparkle-2 {
  0%, 100% { opacity: 0; transform: scale(0.5) rotate(0deg); }
  60% { opacity: 1; transform: scale(1.2) rotate(180deg); }
}

@keyframes sparkle-3 {
  0%, 100% { opacity: 0; transform: scale(0.5) rotate(0deg); }
  40% { opacity: 1; transform: scale(0.9) rotate(180deg); }
}

@keyframes float-up {
  0% { transform: translateY(0) scale(1); opacity: 1; }
  100% { transform: translateY(-30px) scale(1.2); opacity: 0; }
}

@keyframes dart {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(2px); }
  50% { transform: translateX(-2px); }
  75% { transform: translateX(1px); }
}

@keyframes nod {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(3px); }
}

/* Animation classes */
.animate-pulse-soft { animation: pulse-soft 2s ease-in-out infinite; }
.animate-pulse-slow { animation: pulse-slow 3s ease-in-out infinite; }
.animate-bounce-soft { animation: bounce-soft 0.5s ease-in-out infinite; }
.animate-bounce-gentle { animation: bounce-gentle 1s ease-in-out infinite; }
.animate-shake-slow { animation: shake-slow 0.5s ease-in-out infinite; }
.animate-wobble { animation: wobble 0.3s ease-in-out infinite; }
.animate-shake { animation: shake 0.5s ease-in-out; }
.animate-talk { animation: talk 0.3s ease-in-out infinite; }
.animate-spin-slow { animation: spin-slow 3s linear infinite; }
.animate-tear-fall { animation: tear-fall 1s ease-in-out infinite; }
.animate-sweat { animation: sweat 2s ease-in-out infinite; }
.animate-sparkle-1 { animation: sparkle-1 1.5s ease-in-out infinite; }
.animate-sparkle-2 { animation: sparkle-2 1.5s ease-in-out infinite 0.3s; }
.animate-sparkle-3 { animation: sparkle-3 1.5s ease-in-out infinite 0.6s; }
.animate-float-up { animation: float-up 2s ease-out infinite; }
.animate-dart { animation: dart 0.5s ease-in-out infinite; }
.animate-nod { animation: nod 0.5s ease-in-out infinite; }
.animate-sparkle { animation: sparkle-1 1s ease-in-out infinite; }

.delay-100 { animation-delay: 0.1s; }
.delay-300 { animation-delay: 0.3s; }
</style>
