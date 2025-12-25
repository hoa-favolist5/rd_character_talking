/**
 * Voice recorder composable with real-time transcription.
 * 
 * Records audio while simultaneously streaming to AWS Transcribe for
 * real-time speech-to-text, and optionally uploads the final audio to S3.
 */

import { ref, computed } from 'vue'
import { useTranscribeStreaming } from './useTranscribeStreaming'

interface VoiceRecorderResult {
  transcript: string
  audioBlob: Blob | null
  s3Key?: string
}

export function useVoiceRecorder() {
  const config = useRuntimeConfig()
  
  const isRecording = ref(false)
  const mediaRecorder = ref<MediaRecorder | null>(null)
  const audioChunks = ref<Blob[]>([])
  const mediaStream = ref<MediaStream | null>(null)
  
  // Transcription
  const {
    isTranscribing,
    transcript,
    currentTranscript,
    partialTranscript,
    error: transcribeError,
    startTranscription,
    stopTranscription,
    reset: resetTranscription,
  } = useTranscribeStreaming()

  // Combined state
  const isActive = computed(() => isRecording.value || isTranscribing.value)

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
   * Start recording with real-time transcription
   */
  const startRecording = async (): Promise<void> => {
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
    transcript,
    currentTranscript,
    partialTranscript,
    transcribeError,
    startRecording,
    stopRecording,
    cancelRecording,
  }
}
