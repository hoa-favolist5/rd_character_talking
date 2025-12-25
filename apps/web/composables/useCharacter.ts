import { ref, computed } from 'vue'

export type CharacterEmotion = 
  | 'idle'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'happy'
  | 'surprised'
  | 'confused'
  | 'sad'

export interface CharacterState {
  emotion: CharacterEmotion
  isSpeaking: boolean
  currentAudioUrl: string | null
}

export function useCharacter() {
  const emotion = ref<CharacterEmotion>('idle')
  const isSpeaking = ref(false)
  const currentAudioUrl = ref<string | null>(null)
  const audioElement = ref<HTMLAudioElement | null>(null)

  const setEmotion = (newEmotion: CharacterEmotion | string) => {
    const validEmotions: CharacterEmotion[] = [
      'idle', 'listening', 'thinking', 'speaking',
      'happy', 'surprised', 'confused', 'sad'
    ]
    
    if (validEmotions.includes(newEmotion as CharacterEmotion)) {
      emotion.value = newEmotion as CharacterEmotion
    } else {
      emotion.value = 'idle'
    }
  }

  const playAudio = async (audioUrl: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (audioElement.value) {
        audioElement.value.pause()
      }

      audioElement.value = new Audio(audioUrl)
      currentAudioUrl.value = audioUrl
      isSpeaking.value = true
      setEmotion('speaking')

      audioElement.value.onended = () => {
        isSpeaking.value = false
        setEmotion('idle')
        resolve()
      }

      audioElement.value.onerror = (error) => {
        isSpeaking.value = false
        setEmotion('idle')
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
  }

  const animationName = computed(() => {
    const emotionToAnimation: Record<CharacterEmotion, string> = {
      idle: 'character-idle',
      listening: 'character-listening',
      thinking: 'character-thinking',
      speaking: 'character-speaking',
      happy: 'character-happy',
      surprised: 'character-surprised',
      confused: 'character-confused',
      sad: 'character-sad',
    }
    return emotionToAnimation[emotion.value]
  })

  return {
    emotion,
    isSpeaking,
    currentAudioUrl,
    animationName,
    setEmotion,
    playAudio,
    stopAudio,
  }
}

