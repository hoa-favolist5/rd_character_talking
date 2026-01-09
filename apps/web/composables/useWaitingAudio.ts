/**
 * Pre-loaded waiting audio composable
 * 
 * Waiting audio files are stored locally at /audio/waiting/{index}.mp3
 * This saves bandwidth by not streaming audio from the backend.
 * 
 * Arita's waiting phrases (20 total, matching backend CharacterCrew):
 * - 0: "了解。ちょっと待ってね。"
 * - 1: "うん、わかった。今確認するから待っててね。"
 * - 2: "なるほど。少し調べてみるから待ってて。"
 * - ... (20 phrases total)
 */

import { ref, onMounted } from 'vue'

// Pre-loaded audio elements for instant playback
const waitingAudioCache: HTMLAudioElement[] = []
const isPreloaded = ref(false)

// Total number of waiting phrases (0-19)
const TOTAL_WAITING_PHRASES = 20

/**
 * Pre-load all waiting audio files on mount
 */
export function preloadWaitingAudio() {
  if (isPreloaded.value) return

  // Preload all 20 phrases (0-19)
  for (let index = 0; index < TOTAL_WAITING_PHRASES; index++) {
    const audio = new Audio(`/audio/waiting/${index}.mp3`)
    audio.preload = 'auto'
    
    // Trigger load
    audio.load()
    
    waitingAudioCache[index] = audio
  }
  
  isPreloaded.value = true
  console.log('[WaitingAudio] Pre-loaded', TOTAL_WAITING_PHRASES, 'waiting audio files')
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

