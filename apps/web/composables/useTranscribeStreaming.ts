/**
 * Speech recognition composable using Web Speech API for real-time STT.
 * 
 * Uses the browser's built-in speech recognition for immediate feedback,
 * which works without AWS credentials and provides real-time transcription.
 */

import { ref, computed } from 'vue'

// Extend Window interface for SpeechRecognition
declare global {
  interface Window {
    SpeechRecognition: typeof SpeechRecognition
    webkitSpeechRecognition: typeof SpeechRecognition
  }
}

export function useTranscribeStreaming() {
  const isTranscribing = ref(false)
  const currentTranscript = ref('')
  const finalTranscript = ref('')
  const partialTranscript = ref('')
  const error = ref<string | null>(null)
  
  let recognition: SpeechRecognition | null = null
  
  // Combined transcript (final + partial)
  const transcript = computed(() => {
    return (finalTranscript.value + ' ' + partialTranscript.value).trim()
  })

  /**
   * Check if speech recognition is supported
   */
  const isSupported = (): boolean => {
    return !!(window.SpeechRecognition || window.webkitSpeechRecognition)
  }

  /**
   * Start transcription with the given media stream
   * Note: We don't actually need the stream for Web Speech API,
   * but we keep the same interface for compatibility
   */
  const startTranscription = async (_mediaStream?: MediaStream): Promise<void> => {
    if (isTranscribing.value) {
      console.warn('Transcription already in progress')
      return
    }

    if (!isSupported()) {
      error.value = 'Speech recognition not supported in this browser'
      console.error(error.value)
      return
    }

    try {
      error.value = null
      finalTranscript.value = ''
      partialTranscript.value = ''
      currentTranscript.value = ''

      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      recognition = new SpeechRecognition()
      
      // Configure recognition
      recognition.lang = 'ja-JP'  // Japanese
      recognition.continuous = true
      recognition.interimResults = true
      recognition.maxAlternatives = 1

      recognition.onstart = () => {
        console.log('Speech recognition started')
        isTranscribing.value = true
      }

      recognition.onresult = (event) => {
        let interim = ''
        let final = ''

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i]
          if (result.isFinal) {
            final += result[0].transcript
          } else {
            interim += result[0].transcript
          }
        }

        if (final) {
          finalTranscript.value += (finalTranscript.value ? ' ' : '') + final
        }
        partialTranscript.value = interim
        currentTranscript.value = transcript.value
        
        console.log('Transcript update:', { final: finalTranscript.value, partial: partialTranscript.value })
      }

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error)
        if (event.error !== 'aborted' && event.error !== 'no-speech') {
          error.value = `Speech recognition error: ${event.error}`
        }
      }

      recognition.onend = () => {
        console.log('Speech recognition ended')
        // If still supposed to be transcribing, restart (for continuous recognition)
        if (isTranscribing.value && recognition) {
          try {
            recognition.start()
          } catch (e) {
            console.log('Recognition restart skipped')
          }
        }
      }

      recognition.start()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to start speech recognition'
      console.error('Speech recognition error:', e)
      isTranscribing.value = false
    }
  }

  /**
   * Stop transcription and return the final transcript
   */
  const stopTranscription = (): string => {
    isTranscribing.value = false
    
    if (recognition) {
      try {
        recognition.stop()
      } catch (e) {
        // Ignore stop errors
      }
      recognition = null
    }
    
    // Combine final and any remaining partial
    const result = transcript.value
    
    // Keep the result for display but clear partials
    partialTranscript.value = ''
    
    console.log('Final transcript:', result)
    return result
  }

  /**
   * Reset all state
   */
  const reset = () => {
    isTranscribing.value = false
    currentTranscript.value = ''
    finalTranscript.value = ''
    partialTranscript.value = ''
    error.value = null
    
    if (recognition) {
      try {
        recognition.stop()
      } catch (e) {
        // Ignore
      }
      recognition = null
    }
  }

  return {
    isTranscribing,
    transcript,
    currentTranscript,
    finalTranscript,
    partialTranscript,
    error,
    isSupported,
    startTranscription,
    stopTranscription,
    reset,
  }
}
