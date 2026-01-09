/**
 * Audio Queue Composable - Smooth streaming audio playback
 * 
 * Uses Web Audio API for gapless playback.
 * Strategy: Schedule audio as soon as it arrives, Web Audio handles timing.
 * 
 * Supports both:
 * - WebSocket base64 data URLs (addChunk)
 * - WebRTC raw binary audio (addBinaryChunk)
 */

import { ref, onUnmounted } from 'vue'

interface AudioQueueOptions {
  /** Minimum chunks to buffer before starting playback (default: 1) */
  minBuffer?: number
  /** Called when playback starts */
  onPlay?: () => void
  /** Called when all audio finishes */
  onEnded?: () => void
  /** Called when a chunk starts playing */
  onChunkStart?: (index: number) => void
}

export function useAudioQueue(options: AudioQueueOptions = {}) {
  const {
    minBuffer = 1,
    onPlay,
    onEnded,
    onChunkStart,
  } = options

  // State
  const isPlaying = ref(false)
  const isBuffering = ref(false)
  const queueLength = ref(0)
  const currentChunkIndex = ref(0)
  const totalChunks = ref(0)

  // Web Audio context (lazy initialized)
  let audioContext: AudioContext | null = null
  
  // Track all scheduled sources so we can stop them
  const scheduledSources: AudioBufferSourceNode[] = []
  
  // Queue of decoded audio buffers waiting to be scheduled
  const pendingBuffers: AudioBuffer[] = []
  
  // Scheduling state
  let nextScheduleTime = 0
  let isStreamComplete = false
  let hasStartedPlaying = false
  let chunksScheduled = 0
  let chunksFinished = 0

  /**
   * Initialize Web Audio context (must be called from user interaction)
   */
  const initAudioContext = () => {
    if (!audioContext) {
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    }
    // Resume context if suspended (browser autoplay policy)
    if (audioContext.state === 'suspended') {
      audioContext.resume()
    }
    return audioContext
  }

  /**
   * Decode base64 audio data URL to AudioBuffer
   */
  const decodeAudioData = async (dataUrl: string): Promise<AudioBuffer> => {
    const ctx = initAudioContext()
    
    // Use fetch for efficient decoding
    const response = await fetch(dataUrl)
    const arrayBuffer = await response.arrayBuffer()
    
    return await ctx.decodeAudioData(arrayBuffer)
  }

  /**
   * Schedule an AudioBuffer for playback at the specified time
   */
  const scheduleBuffer = (buffer: AudioBuffer, startTime: number): number => {
    if (!audioContext) return 0
    
    const source = audioContext.createBufferSource()
    source.buffer = buffer
    source.connect(audioContext.destination)
    
    const chunkIndex = chunksScheduled
    chunksScheduled++
    
    source.onended = () => {
      chunksFinished++
      
      // Remove from scheduled list
      const idx = scheduledSources.indexOf(source)
      if (idx > -1) scheduledSources.splice(idx, 1)
      
      // Update current playing index
      currentChunkIndex.value = chunksFinished
      onChunkStart?.(chunksFinished)
      
      // Check if all done
      if (isStreamComplete && chunksFinished >= totalChunks.value) {
        finishPlayback()
      }
    }
    
    source.start(startTime)
    scheduledSources.push(source)
    
    console.log(`[AudioQueue] Scheduled chunk ${chunkIndex + 1} at t=${startTime.toFixed(3)}`)
    
    return buffer.duration
  }

  /**
   * Process pending buffers - schedule them for playback
   */
  const schedulePendingBuffers = () => {
    if (!audioContext) return
    
    while (pendingBuffers.length > 0) {
      const buffer = pendingBuffers.shift()!
      
      // If we're behind, catch up to current time
      if (nextScheduleTime < audioContext.currentTime) {
        nextScheduleTime = audioContext.currentTime + 0.01 // Small offset to avoid glitches
      }
      
      const duration = scheduleBuffer(buffer, nextScheduleTime)
      nextScheduleTime += duration // Gapless: next starts immediately after this ends
    }
    
    queueLength.value = pendingBuffers.length
  }

  /**
   * Start playback if we have enough buffered
   */
  const tryStartPlayback = () => {
    if (hasStartedPlaying) return
    
    const shouldStart = 
      (pendingBuffers.length >= minBuffer) || 
      (isStreamComplete && pendingBuffers.length > 0)
    
    if (shouldStart) {
      hasStartedPlaying = true
      isBuffering.value = false
      isPlaying.value = true
      
      console.log('[AudioQueue] Starting playback with', pendingBuffers.length, 'chunks buffered')
      
      const ctx = initAudioContext()
      nextScheduleTime = ctx.currentTime + 0.01 // Start almost immediately
      
      onPlay?.()
      onChunkStart?.(0)
      
      schedulePendingBuffers()
    }
  }

  /**
   * Add audio chunk to queue (base64 data URL from WebSocket)
   */
  const addChunk = async (audioUrl: string, index: number) => {
    console.log('[AudioQueue] Decoding chunk', index)
    
    try {
      // Decode audio
      const buffer = await decodeAudioData(audioUrl)
      
      // Add to pending buffers
      pendingBuffers.push(buffer)
      queueLength.value = pendingBuffers.length
      totalChunks.value = Math.max(totalChunks.value, index)
      
      console.log('[AudioQueue] Chunk', index, 'decoded, duration:', buffer.duration.toFixed(2) + 's')
      
      // Try to start playback or schedule immediately if already playing
      if (!hasStartedPlaying) {
        isBuffering.value = true
        tryStartPlayback()
      } else {
        // Already playing - schedule immediately
        schedulePendingBuffers()
      }
    } catch (error) {
      console.error('[AudioQueue] Failed to decode chunk', index, error)
    }
  }

  /**
   * Add binary audio chunk to queue (raw bytes from WebRTC)
   * 
   * This is more efficient than base64 as it skips encoding/decoding.
   */
  const addBinaryChunk = async (audioData: Uint8Array, index: number) => {
    console.log('[AudioQueue] Decoding binary chunk', index, ':', audioData.length, 'bytes')
    
    try {
      const ctx = initAudioContext()
      
      // Create ArrayBuffer from Uint8Array
      const arrayBuffer = audioData.buffer.slice(
        audioData.byteOffset,
        audioData.byteOffset + audioData.byteLength
      )
      
      // Decode audio data directly
      const buffer = await ctx.decodeAudioData(arrayBuffer)
      
      // Add to pending buffers
      pendingBuffers.push(buffer)
      queueLength.value = pendingBuffers.length
      totalChunks.value = Math.max(totalChunks.value, index)
      
      console.log('[AudioQueue] Binary chunk', index, 'decoded, duration:', buffer.duration.toFixed(2) + 's')
      
      // Try to start playback or schedule immediately if already playing
      if (!hasStartedPlaying) {
        isBuffering.value = true
        tryStartPlayback()
      } else {
        // Already playing - schedule immediately
        schedulePendingBuffers()
      }
    } catch (error) {
      console.error('[AudioQueue] Failed to decode binary chunk', index, error)
    }
  }

  /**
   * Mark stream as complete (no more chunks coming)
   */
  const markStreamComplete = (total: number) => {
    console.log('[AudioQueue] Stream complete, total:', total)
    isStreamComplete = true
    totalChunks.value = total
    
    // If we haven't started yet, start now with whatever we have
    if (!hasStartedPlaying && pendingBuffers.length > 0) {
      tryStartPlayback()
    }
    
    // If no chunks were scheduled and stream is complete, finish
    if (total === 0 || (hasStartedPlaying && chunksFinished >= total)) {
      finishPlayback()
    }
  }

  /**
   * Finish playback and clean up
   */
  const finishPlayback = () => {
    console.log('[AudioQueue] Playback finished')
    isPlaying.value = false
    onEnded?.()
  }

  /**
   * Stop playback and clear queue
   */
  const stop = () => {
    console.log('[AudioQueue] Stopping')
    
    // Stop ALL scheduled sources
    for (const source of scheduledSources) {
      try {
        source.stop()
      } catch (e) {
        // Ignore - might already be stopped
      }
    }
    scheduledSources.length = 0
    
    // Clear pending buffers
    pendingBuffers.length = 0
    queueLength.value = 0
    
    // Reset state
    isPlaying.value = false
    isBuffering.value = false
    currentChunkIndex.value = 0
    totalChunks.value = 0
    hasStartedPlaying = false
    isStreamComplete = false
    nextScheduleTime = 0
    chunksScheduled = 0
    chunksFinished = 0
  }

  /**
   * Reset for new audio stream
   */
  const reset = () => {
    stop()
    console.log('[AudioQueue] Reset for new stream')
  }

  // Cleanup on unmount
  onUnmounted(() => {
    stop()
    if (audioContext) {
      audioContext.close()
      audioContext = null
    }
  })

  return {
    // State
    isPlaying,
    isBuffering,
    queueLength,
    currentChunkIndex,
    totalChunks,
    
    // Methods
    addChunk,           // For WebSocket base64 data URLs
    addBinaryChunk,     // For WebRTC raw binary audio
    markStreamComplete,
    stop,
    reset,
    initAudioContext,
  }
}
