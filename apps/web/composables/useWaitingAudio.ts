/**
 * Pre-loaded waiting audio composable
 * 
 * Waiting audio files are stored locally at /audio/waiting/{index}.mp3
 * This saves bandwidth by not streaming audio from the backend.
 * 
 * Phrases (matching backend WAITING_PHRASES):
 * - 0: "ちょっと待ってね"
 * - 1: "えーっと、ちょっと待って"
 * - 2: "うーんと、待ってね"
 * - 3: "少し待ってね"
 */

import { ref, onMounted } from 'vue'

// Pre-loaded audio elements for instant playback
const waitingAudioCache: HTMLAudioElement[] = []
const isPreloaded = ref(false)

/**
 * Pre-load all waiting audio files on mount
 */
export function preloadWaitingAudio() {
  if (isPreloaded.value) return

  const phrases = [0, 1, 2, 3]
  
  for (const index of phrases) {
    const audio = new Audio(`/audio/waiting/${index}.mp3`)
    audio.preload = 'auto'
    
    // Trigger load
    audio.load()
    
    waitingAudioCache[index] = audio
  }
  
  isPreloaded.value = true
  console.log('[WaitingAudio] Pre-loaded', phrases.length, 'waiting audio files')
}

/**
 * Play waiting audio by phrase index
 */
export function playWaitingAudio(phraseIndex: number): void {
  const audio = waitingAudioCache[phraseIndex]
  
  if (audio) {
    // Reset and play
    audio.currentTime = 0
    audio.play().catch((err) => {
      console.error('[WaitingAudio] Failed to play:', err)
    })
    console.log('[WaitingAudio] Playing phrase', phraseIndex)
  } else {
    // Fallback: create new audio element
    console.warn('[WaitingAudio] Cache miss, creating new audio for phrase', phraseIndex)
    const fallback = new Audio(`/audio/waiting/${phraseIndex}.mp3`)
    fallback.play().catch((err) => {
      console.error('[WaitingAudio] Failed to play fallback:', err)
    })
  }
}

/**
 * Composable for waiting audio
 */
export function useWaitingAudio() {
  onMounted(() => {
    preloadWaitingAudio()
  })

  return {
    isPreloaded,
    playWaitingAudio,
  }
}

