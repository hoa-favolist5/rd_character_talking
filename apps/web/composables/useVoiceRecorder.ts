/**
 * Voice recorder composable with real-time transcription and silence detection.
 * 
 * Records audio while simultaneously streaming to AWS Transcribe for
 * real-time speech-to-text, with automatic silence detection for
 * hands-free conversation mode.
 * 
 * Silence detection is based on transcript activity - when we have a final
 * transcript and no new speech for a period, we trigger the callback.
 */

import { ref, computed, watch } from 'vue'
import { useTranscribeStreaming } from './useTranscribeStreaming'

interface VoiceRecorderResult {
  transcript: string
  audioBlob: Blob | null
  s3Key?: string
}

interface SilenceDetectionOptions {
  silenceThreshold?: number  // Not used anymore, kept for compatibility
  silenceDuration?: number   // How long no speech before triggering (ms), default 1500
  onSilenceDetected?: () => void  // Callback when silence is detected
}

export function useVoiceRecorder() {
  const config = useRuntimeConfig()
  
  const isRecording = ref(false)
  const mediaRecorder = ref<MediaRecorder | null>(null)
  const audioChunks = ref<Blob[]>([])
  const mediaStream = ref<MediaStream | null>(null)
  
  // Silence detection based on transcript activity
  const silenceTimer = ref<ReturnType<typeof setTimeout> | null>(null)
  const isSpeaking = ref(false)
  const volumeLevel = ref(0)
  const silenceCallback = ref<(() => void) | null>(null)
  const silenceDurationMs = ref(1500)
  const hasFinalTranscript = ref(false)
  
  // Transcription
  const {
    isTranscribing,
    transcript,
    currentTranscript,
    partialTranscript,
    finalTranscript,
    error: transcribeError,
    startTranscription,
    stopTranscription,
    reset: resetTranscription,
  } = useTranscribeStreaming()

  // Combined state
  const isActive = computed(() => isRecording.value || isTranscribing.value)
  
  // Watch for transcript changes to detect speech end
  watch([finalTranscript, partialTranscript], ([newFinal, newPartial], [oldFinal, oldPartial]) => {
    if (!isRecording.value || !silenceCallback.value) return
    
    console.log('[Silence Detection] Transcript changed:', { newFinal, newPartial })
    
    // If we have new speech (partial or final), user is speaking
    if (newPartial || (newFinal && newFinal !== oldFinal)) {
      isSpeaking.value = true
      
      // Clear any pending silence timer
      if (silenceTimer.value) {
        clearTimeout(silenceTimer.value)
        silenceTimer.value = null
      }
      
      // Mark that we have received a final transcript
      if (newFinal && newFinal.trim()) {
        hasFinalTranscript.value = true
      }
    }
    
    // If no partial and we have a final transcript, start silence countdown
    if (!newPartial && hasFinalTranscript.value && newFinal && newFinal.trim()) {
      isSpeaking.value = false
      
      // Start countdown if not already started
      if (!silenceTimer.value) {
        console.log('[Silence Detection] Starting silence countdown:', silenceDurationMs.value, 'ms')
        silenceTimer.value = setTimeout(() => {
          console.log('[Silence Detection] Silence timeout reached, triggering callback')
          if (isRecording.value && silenceCallback.value) {
            const callback = silenceCallback.value
            silenceCallback.value = null  // Prevent double trigger
            callback()
          }
        }, silenceDurationMs.value)
      }
    }
  })

  /**
   * Get pre-signed URL for S3 upload
   */
  const getUploadUrl = async (filename: string): Promise<{ uploadUrl: string; s3Key: string }> => {
    const response = await fetch(`${config.public.apiUrl}/credentials/s3-upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename,
        content_type: 'audio/webm',
      }),
    })
    
    if (!response.ok) {
      throw new Error('Failed to get S3 upload URL')
    }
    
    return response.json()
  }

  /**
   * Upload audio blob to S3
   */
  const uploadToS3 = async (blob: Blob): Promise<string | undefined> => {
    try {
      const filename = `recording-${Date.now()}.webm`
      const { uploadUrl, s3Key } = await getUploadUrl(filename)
      
      const uploadResponse = await fetch(uploadUrl, {
        method: 'PUT',
        headers: { 'Content-Type': 'audio/webm' },
        body: blob,
      })
      
      if (!uploadResponse.ok) {
        console.error('S3 upload failed:', uploadResponse.statusText)
        return undefined
      }
      
      return s3Key
    } catch (e) {
      console.error('S3 upload error:', e)
      return undefined
    }
  }

  /**
   * Setup silence detection based on transcript activity
   */
  const setupSilenceDetection = (options: SilenceDetectionOptions = {}): void => {
    const { silenceDuration = 1500, onSilenceDetected } = options
    
    silenceDurationMs.value = silenceDuration
    silenceCallback.value = onSilenceDetected || null
    hasFinalTranscript.value = false
    
    console.log('[Silence Detection] Setup with duration:', silenceDuration, 'ms')
  }

  /**
   * Cleanup silence detection resources
   */
  const cleanupSilenceDetection = (): void => {
    if (silenceTimer.value) {
      clearTimeout(silenceTimer.value)
      silenceTimer.value = null
    }
    silenceCallback.value = null
    hasFinalTranscript.value = false
    isSpeaking.value = false
    volumeLevel.value = 0
    console.log('[Silence Detection] Cleaned up')
  }

  /**
   * Start recording with real-time transcription
   * @param silenceOptions - Optional silence detection configuration
   */
  const startRecording = async (silenceOptions?: SilenceDetectionOptions): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })

      mediaStream.value = stream

      // Start MediaRecorder for audio capture
      mediaRecorder.value = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      })

      audioChunks.value = []

      mediaRecorder.value.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.value.push(event.data)
        }
      }

      mediaRecorder.value.start(100) // Collect data every 100ms
      isRecording.value = true

      // Setup silence detection if callback provided (uses transcript, not audio levels)
      if (silenceOptions?.onSilenceDetected) {
        setupSilenceDetection(silenceOptions)
      }

      // Start transcription in parallel (don't await, let it run)
      startTranscription(stream).catch((e) => {
        console.error('Transcription start error:', e)
      })
      
      console.log('Recording started successfully')
    } catch (error) {
      console.error('Failed to start recording:', error)
      isRecording.value = false
      throw error
    }
  }

  /**
   * Stop recording and return result with transcript
   */
  const stopRecording = async (uploadAudio: boolean = false): Promise<VoiceRecorderResult> => {
    // Cleanup silence detection first
    cleanupSilenceDetection()
    
    return new Promise(async (resolve) => {
      if (!mediaRecorder.value || mediaRecorder.value.state === 'inactive') {
        resolve({ transcript: '', audioBlob: null })
        return
      }

      // Stop transcription and get final transcript
      const finalTranscript = stopTranscription()

      mediaRecorder.value.onstop = async () => {
        const audioBlob = new Blob(audioChunks.value, { type: 'audio/webm' })
        
        // Stop all tracks
        if (mediaStream.value) {
          mediaStream.value.getTracks().forEach((track) => track.stop())
          mediaStream.value = null
        }
        
        isRecording.value = false

        // Optionally upload to S3
        let s3Key: string | undefined
        if (uploadAudio && audioBlob.size > 0) {
          s3Key = await uploadToS3(audioBlob)
        }

        resolve({
          transcript: finalTranscript,
          audioBlob,
          s3Key,
        })
      }

      mediaRecorder.value.stop()
    })
  }

  /**
   * Cancel recording without saving
   */
  const cancelRecording = (): void => {
    // Cleanup silence detection
    cleanupSilenceDetection()
    
    if (mediaRecorder.value && mediaRecorder.value.state !== 'inactive') {
      mediaRecorder.value.stop()
    }
    
    if (mediaStream.value) {
      mediaStream.value.getTracks().forEach((track) => track.stop())
      mediaStream.value = null
    }
    
    isRecording.value = false
    audioChunks.value = []
    resetTranscription()
  }

  return {
    isRecording,
    isTranscribing,
    isActive,
    isSpeaking,
    volumeLevel,
    transcript,
    currentTranscript,
    partialTranscript,
    transcribeError,
    startRecording,
    stopRecording,
    cancelRecording,
  }
}
