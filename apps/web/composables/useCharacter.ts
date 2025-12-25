import { ref, computed } from 'vue'

/**
 * Character emotion states (basic emotions)
 */
export type CharacterEmotion = 
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'happy'
  | 'surprised'
  | 'confused'
  | 'sad'
  | 'excited'
  | 'calm'

/**
 * Character actions for avatar animation
 * These map to specific visual animations/expressions
 */
export type CharacterAction =
  // Basic expressions
  | 'idle'
  | 'smile'
  | 'laugh'
  | 'grin'
  // Sad/Sympathetic
  | 'sad'
  | 'cry'
  | 'sympathetic'
  | 'comfort'
  // Curious/Thinking
  | 'curious'
  | 'thinking'
  | 'confused'
  | 'wonder'
  // Surprise/Excitement
  | 'surprised'
  | 'shocked'
  | 'excited'
  | 'amazed'
  // Scared/Nervous
  | 'scared'
  | 'nervous'
  | 'worried'
  // Affection/Romance
  | 'blush'
  | 'love'
  | 'shy'
  | 'wink'
  // Agreement/Gestures
  | 'nod'
  | 'shake_head'
  | 'thumbs_up'
  // Speaking/Listening
  | 'speak'
  | 'listen'
  | 'explain'
  // Special
  | 'wave'
  | 'bow'
  | 'celebrate'
  | 'cheer'

/**
 * Action configuration for avatar display
 */
export interface ActionConfig {
  emoji: string
  label: string
  labelJa: string
  color: string
  bgColor: string
  glowColor: string
  eyeScale: number
  eyeAnimation: string
  mouthStyle: 'smile' | 'open' | 'small' | 'wide' | 'sad' | 'o' | 'teeth'
  mouthAnimation: string
  blush: boolean
  tears: boolean
  sparkles: boolean
  hearts: boolean
  sweat: boolean
  exclamation: boolean
}

/**
 * Configuration for each action
 */
export const ACTION_CONFIGS: Record<CharacterAction, ActionConfig> = {
  // Basic expressions
  idle: {
    emoji: '😊',
    label: 'Idle',
    labelJa: '待機中',
    color: 'text-gray-400',
    bgColor: 'bg-gray-500/30',
    glowColor: 'rgba(156, 163, 175, 0.3)',
    eyeScale: 1,
    eyeAnimation: '',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  smile: {
    emoji: '😊',
    label: 'Smile',
    labelJa: '笑顔',
    color: 'text-pink-400',
    bgColor: 'bg-pink-500/30',
    glowColor: 'rgba(244, 114, 182, 0.5)',
    eyeScale: 0.8,
    eyeAnimation: 'squint',
    mouthStyle: 'smile',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  laugh: {
    emoji: '😄',
    label: 'Laugh',
    labelJa: '笑い',
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-500/30',
    glowColor: 'rgba(250, 204, 21, 0.5)',
    eyeScale: 0.6,
    eyeAnimation: 'squint',
    mouthStyle: 'wide',
    mouthAnimation: 'bounce',
    blush: true,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  grin: {
    emoji: '😁',
    label: 'Grin',
    labelJa: 'にっこり',
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/30',
    glowColor: 'rgba(251, 146, 60, 0.5)',
    eyeScale: 0.7,
    eyeAnimation: 'squint',
    mouthStyle: 'teeth',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  // Sad/Sympathetic
  sad: {
    emoji: '😢',
    label: 'Sad',
    labelJa: '悲しい',
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/30',
    glowColor: 'rgba(96, 165, 250, 0.5)',
    eyeScale: 1.1,
    eyeAnimation: 'droop',
    mouthStyle: 'sad',
    mouthAnimation: '',
    blush: false,
    tears: true,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  cry: {
    emoji: '😭',
    label: 'Cry',
    labelJa: '泣き',
    color: 'text-blue-500',
    bgColor: 'bg-blue-600/30',
    glowColor: 'rgba(59, 130, 246, 0.5)',
    eyeScale: 0.7,
    eyeAnimation: 'closed',
    mouthStyle: 'wide',
    mouthAnimation: 'wobble',
    blush: false,
    tears: true,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  sympathetic: {
    emoji: '🥺',
    label: 'Sympathetic',
    labelJa: '共感',
    color: 'text-indigo-400',
    bgColor: 'bg-indigo-500/30',
    glowColor: 'rgba(129, 140, 248, 0.5)',
    eyeScale: 1.2,
    eyeAnimation: 'soft',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  comfort: {
    emoji: '🤗',
    label: 'Comfort',
    labelJa: '慰め',
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/30',
    glowColor: 'rgba(251, 191, 36, 0.5)',
    eyeScale: 0.9,
    eyeAnimation: 'gentle',
    mouthStyle: 'smile',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: false,
    hearts: true,
    sweat: false,
    exclamation: false,
  },
  // Curious/Thinking
  curious: {
    emoji: '🤔',
    label: 'Curious',
    labelJa: '興味津々',
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-500/30',
    glowColor: 'rgba(34, 211, 238, 0.5)',
    eyeScale: 1.1,
    eyeAnimation: 'tilt',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  thinking: {
    emoji: '💭',
    label: 'Thinking',
    labelJa: '考え中',
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/30',
    glowColor: 'rgba(192, 132, 252, 0.5)',
    eyeScale: 1,
    eyeAnimation: 'look-up',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  confused: {
    emoji: '😕',
    label: 'Confused',
    labelJa: '困惑',
    color: 'text-violet-400',
    bgColor: 'bg-violet-500/30',
    glowColor: 'rgba(167, 139, 250, 0.5)',
    eyeScale: 1.1,
    eyeAnimation: 'spiral',
    mouthStyle: 'small',
    mouthAnimation: 'wobble',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: true,
    exclamation: false,
  },
  wonder: {
    emoji: '✨',
    label: 'Wonder',
    labelJa: '感嘆',
    color: 'text-teal-400',
    bgColor: 'bg-teal-500/30',
    glowColor: 'rgba(45, 212, 191, 0.5)',
    eyeScale: 1.3,
    eyeAnimation: 'sparkle',
    mouthStyle: 'o',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  // Surprise/Excitement
  surprised: {
    emoji: '😲',
    label: 'Surprised',
    labelJa: '驚き',
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/30',
    glowColor: 'rgba(251, 191, 36, 0.5)',
    eyeScale: 1.4,
    eyeAnimation: 'wide',
    mouthStyle: 'o',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: true,
  },
  shocked: {
    emoji: '😱',
    label: 'Shocked',
    labelJa: 'ショック',
    color: 'text-red-400',
    bgColor: 'bg-red-500/30',
    glowColor: 'rgba(248, 113, 113, 0.5)',
    eyeScale: 1.5,
    eyeAnimation: 'shake',
    mouthStyle: 'wide',
    mouthAnimation: 'shake',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: true,
    exclamation: true,
  },
  excited: {
    emoji: '🤩',
    label: 'Excited',
    labelJa: '興奮',
    color: 'text-rose-400',
    bgColor: 'bg-rose-500/30',
    glowColor: 'rgba(251, 113, 133, 0.5)',
    eyeScale: 1.2,
    eyeAnimation: 'star',
    mouthStyle: 'wide',
    mouthAnimation: 'bounce',
    blush: true,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: true,
  },
  amazed: {
    emoji: '🌟',
    label: 'Amazed',
    labelJa: '感動',
    color: 'text-yellow-300',
    bgColor: 'bg-yellow-400/30',
    glowColor: 'rgba(253, 224, 71, 0.5)',
    eyeScale: 1.3,
    eyeAnimation: 'sparkle',
    mouthStyle: 'smile',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  // Scared/Nervous
  scared: {
    emoji: '😨',
    label: 'Scared',
    labelJa: '怖い',
    color: 'text-slate-400',
    bgColor: 'bg-slate-500/30',
    glowColor: 'rgba(148, 163, 184, 0.5)',
    eyeScale: 1.4,
    eyeAnimation: 'shake',
    mouthStyle: 'small',
    mouthAnimation: 'wobble',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: true,
    exclamation: false,
  },
  nervous: {
    emoji: '😰',
    label: 'Nervous',
    labelJa: '緊張',
    color: 'text-sky-400',
    bgColor: 'bg-sky-500/30',
    glowColor: 'rgba(56, 189, 248, 0.5)',
    eyeScale: 1.1,
    eyeAnimation: 'dart',
    mouthStyle: 'small',
    mouthAnimation: 'wobble',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: true,
    exclamation: false,
  },
  worried: {
    emoji: '😟',
    label: 'Worried',
    labelJa: '心配',
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/30',
    glowColor: 'rgba(52, 211, 153, 0.5)',
    eyeScale: 1.1,
    eyeAnimation: 'droop',
    mouthStyle: 'sad',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: true,
    exclamation: false,
  },
  // Affection/Romance
  blush: {
    emoji: '😳',
    label: 'Blush',
    labelJa: '照れ',
    color: 'text-pink-400',
    bgColor: 'bg-pink-500/30',
    glowColor: 'rgba(244, 114, 182, 0.5)',
    eyeScale: 0.9,
    eyeAnimation: 'shy',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: false,
    hearts: true,
    sweat: false,
    exclamation: false,
  },
  love: {
    emoji: '😍',
    label: 'Love',
    labelJa: 'ラブ',
    color: 'text-red-400',
    bgColor: 'bg-red-500/30',
    glowColor: 'rgba(248, 113, 113, 0.5)',
    eyeScale: 1,
    eyeAnimation: 'heart',
    mouthStyle: 'smile',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: false,
    hearts: true,
    sweat: false,
    exclamation: false,
  },
  shy: {
    emoji: '🙈',
    label: 'Shy',
    labelJa: 'シャイ',
    color: 'text-rose-300',
    bgColor: 'bg-rose-400/30',
    glowColor: 'rgba(253, 164, 175, 0.5)',
    eyeScale: 0.7,
    eyeAnimation: 'look-away',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  wink: {
    emoji: '😉',
    label: 'Wink',
    labelJa: 'ウィンク',
    color: 'text-fuchsia-400',
    bgColor: 'bg-fuchsia-500/30',
    glowColor: 'rgba(232, 121, 249, 0.5)',
    eyeScale: 1,
    eyeAnimation: 'wink',
    mouthStyle: 'smile',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  // Agreement/Gestures
  nod: {
    emoji: '👍',
    label: 'Nod',
    labelJa: 'うなずき',
    color: 'text-green-400',
    bgColor: 'bg-green-500/30',
    glowColor: 'rgba(74, 222, 128, 0.5)',
    eyeScale: 0.9,
    eyeAnimation: 'nod',
    mouthStyle: 'smile',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  shake_head: {
    emoji: '🙅',
    label: 'Shake Head',
    labelJa: '首振り',
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/30',
    glowColor: 'rgba(251, 146, 60, 0.5)',
    eyeScale: 1,
    eyeAnimation: 'shake',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  thumbs_up: {
    emoji: '👍',
    label: 'Thumbs Up',
    labelJa: 'いいね',
    color: 'text-lime-400',
    bgColor: 'bg-lime-500/30',
    glowColor: 'rgba(163, 230, 53, 0.5)',
    eyeScale: 0.8,
    eyeAnimation: 'squint',
    mouthStyle: 'teeth',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  // Speaking/Listening
  speak: {
    emoji: '💬',
    label: 'Speaking',
    labelJa: '話し中',
    color: 'text-green-400',
    bgColor: 'bg-green-500/30',
    glowColor: 'rgba(74, 222, 128, 0.5)',
    eyeScale: 1,
    eyeAnimation: '',
    mouthStyle: 'open',
    mouthAnimation: 'talk',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  listen: {
    emoji: '🎤',
    label: 'Listening',
    labelJa: '聞いています',
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/30',
    glowColor: 'rgba(96, 165, 250, 0.5)',
    eyeScale: 1.1,
    eyeAnimation: 'focus',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  explain: {
    emoji: '📚',
    label: 'Explain',
    labelJa: '説明',
    color: 'text-indigo-400',
    bgColor: 'bg-indigo-500/30',
    glowColor: 'rgba(129, 140, 248, 0.5)',
    eyeScale: 1,
    eyeAnimation: '',
    mouthStyle: 'open',
    mouthAnimation: 'talk',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  // Special
  wave: {
    emoji: '👋',
    label: 'Wave',
    labelJa: '手を振る',
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-500/30',
    glowColor: 'rgba(250, 204, 21, 0.5)',
    eyeScale: 0.8,
    eyeAnimation: 'squint',
    mouthStyle: 'smile',
    mouthAnimation: '',
    blush: true,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  bow: {
    emoji: '🙇',
    label: 'Bow',
    labelJa: 'お辞儀',
    color: 'text-slate-400',
    bgColor: 'bg-slate-500/30',
    glowColor: 'rgba(148, 163, 184, 0.5)',
    eyeScale: 0.5,
    eyeAnimation: 'closed',
    mouthStyle: 'small',
    mouthAnimation: '',
    blush: false,
    tears: false,
    sparkles: false,
    hearts: false,
    sweat: false,
    exclamation: false,
  },
  celebrate: {
    emoji: '🎉',
    label: 'Celebrate',
    labelJa: 'お祝い',
    color: 'text-fuchsia-400',
    bgColor: 'bg-fuchsia-500/30',
    glowColor: 'rgba(232, 121, 249, 0.5)',
    eyeScale: 0.7,
    eyeAnimation: 'squint',
    mouthStyle: 'wide',
    mouthAnimation: 'bounce',
    blush: true,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: true,
  },
  cheer: {
    emoji: '🙌',
    label: 'Cheer',
    labelJa: '応援',
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/30',
    glowColor: 'rgba(251, 146, 60, 0.5)',
    eyeScale: 0.8,
    eyeAnimation: 'squint',
    mouthStyle: 'teeth',
    mouthAnimation: 'bounce',
    blush: true,
    tears: false,
    sparkles: true,
    hearts: false,
    sweat: false,
    exclamation: true,
  },
}

export interface CharacterState {
  emotion: CharacterEmotion
  action: CharacterAction
  isSpeaking: boolean
  currentAudioUrl: string | null
}

export function useCharacter() {
  const emotion = ref<CharacterEmotion>('idle')
  const action = ref<CharacterAction>('idle')
  const isSpeaking = ref(false)
  const currentAudioUrl = ref<string | null>(null)
  const audioElement = ref<HTMLAudioElement | null>(null)

  const validEmotions: CharacterEmotion[] = [
    'idle', 'listening', 'thinking', 'speaking',
    'happy', 'surprised', 'confused', 'sad', 'excited', 'calm'
  ]

  const validActions: CharacterAction[] = Object.keys(ACTION_CONFIGS) as CharacterAction[]

  const setEmotion = (newEmotion: CharacterEmotion | string) => {
    if (validEmotions.includes(newEmotion as CharacterEmotion)) {
      emotion.value = newEmotion as CharacterEmotion
    } else {
      emotion.value = 'idle'
    }
  }

  const setAction = (newAction: CharacterAction | string) => {
    if (validActions.includes(newAction as CharacterAction)) {
      action.value = newAction as CharacterAction
    } else {
      // Try to map emotion to action if action is invalid
      const emotionToAction: Record<string, CharacterAction> = {
        happy: 'smile',
        sad: 'sad',
        surprised: 'surprised',
        confused: 'confused',
        excited: 'excited',
        listening: 'listen',
        thinking: 'thinking',
        speaking: 'speak',
      }
      action.value = emotionToAction[newAction] || 'idle'
    }
  }

  const actionConfig = computed(() => ACTION_CONFIGS[action.value])

  const playAudio = async (audioUrl: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (audioElement.value) {
        audioElement.value.pause()
      }

      audioElement.value = new Audio(audioUrl)
      currentAudioUrl.value = audioUrl
      isSpeaking.value = true
      setEmotion('speaking')
      setAction('speak')

      audioElement.value.onended = () => {
        isSpeaking.value = false
        setEmotion('idle')
        setAction('idle')
        resolve()
      }

      audioElement.value.onerror = (error) => {
        isSpeaking.value = false
        setEmotion('idle')
        setAction('idle')
        reject(error)
      }

      audioElement.value.play().catch(reject)
    })
  }

  const stopAudio = () => {
    if (audioElement.value) {
      audioElement.value.pause()
      audioElement.value.currentTime = 0
    }
    isSpeaking.value = false
    setEmotion('idle')
    setAction('idle')
  }

  const animationName = computed(() => {
    return `character-${action.value}`
  })

  return {
    emotion,
    action,
    actionConfig,
    isSpeaking,
    currentAudioUrl,
    animationName,
    setEmotion,
    setAction,
    playAudio,
    stopAudio,
  }
}
