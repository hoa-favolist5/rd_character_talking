"""Voice Emotion Analysis Service using Librosa.

Extracts prosodic features from audio to infer emotional state:
- Pitch (fundamental frequency): High = excited/happy, Low = sad/calm
- Pitch variation: High variability = emotional/expressive
- Speaking rate: Fast = excited/nervous, Slow = sad/thoughtful
- Energy/Volume: High = strong emotion, Low = subdued
- Silence ratio: High = hesitant/uncertain
"""

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass
class VoiceFeatures:
    """Extracted voice features for emotion analysis."""
    
    # Pitch features
    pitch_mean: float  # Average fundamental frequency (Hz)
    pitch_std: float   # Pitch variability (Hz)
    pitch_range: float # Max - Min pitch (Hz)
    
    # Energy features
    energy_mean: float     # Average RMS energy
    energy_std: float      # Energy variability
    energy_max: float      # Peak energy
    
    # Temporal features
    speaking_rate: float   # Estimated syllables per second
    silence_ratio: float   # Ratio of silent frames
    duration: float        # Total duration in seconds
    
    # Spectral features (for voice quality)
    spectral_centroid: float  # Brightness of sound
    spectral_rolloff: float   # High frequency content
    
    # Inferred emotion hints
    emotion_hints: list[str]
    
    def to_context_string(self) -> str:
        """Format features as context string for LLM."""
        hints_str = ", ".join(self.emotion_hints) if self.emotion_hints else "neutral tone"
        
        return f"""[Voice Analysis Results]
The user's voice characteristics:
- Voice tone: {hints_str}
- Pitch: {self.pitch_mean:.1f}Hz (variation: {self.pitch_std:.1f}Hz)
- Energy level: {self.energy_mean:.4f} (peak: {self.energy_max:.4f})
- Speaking rate: {self.speaking_rate:.1f} syllables/sec
- Pause ratio: {self.silence_ratio:.1%}
- Voice brightness: {self.spectral_centroid:.1f}Hz"""


class VoiceEmotionService:
    """Service for extracting emotional features from voice audio."""
    
    # Thresholds for emotion inference (tuned for Japanese speech)
    PITCH_HIGH_THRESHOLD = 220      # Hz - above this = excited/happy
    PITCH_LOW_THRESHOLD = 140       # Hz - below this = sad/calm
    PITCH_VAR_HIGH_THRESHOLD = 40   # Hz - high variability = emotional
    ENERGY_HIGH_THRESHOLD = 0.08    # RMS - high energy = strong emotion
    ENERGY_LOW_THRESHOLD = 0.02     # RMS - low energy = subdued
    RATE_FAST_THRESHOLD = 5.0       # syllables/sec - fast speech
    RATE_SLOW_THRESHOLD = 2.5       # syllables/sec - slow speech
    SILENCE_HIGH_THRESHOLD = 0.35   # ratio - many pauses
    
    def __init__(self) -> None:
        """Initialize the voice emotion service."""
        self._sample_rate = 22050  # Standard librosa sample rate
    
    async def analyze_audio(
        self, 
        audio_data: bytes, 
        mime_type: str = "audio/webm"
    ) -> VoiceFeatures:
        """
        Analyze audio data and extract emotion-related features.
        
        Args:
            audio_data: Raw audio bytes
            mime_type: Audio MIME type (webm, wav, mp3, etc.)
            
        Returns:
            VoiceFeatures with extracted prosodic features and emotion hints
        """
        # Convert audio bytes to numpy array
        y, sr = self._load_audio(audio_data, mime_type)
        
        # Resample if needed
        if sr != self._sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self._sample_rate)
            sr = self._sample_rate
        
        # Extract all features
        pitch_features = self._extract_pitch_features(y, sr)
        energy_features = self._extract_energy_features(y)
        temporal_features = self._extract_temporal_features(y, sr)
        spectral_features = self._extract_spectral_features(y, sr)
        
        # Combine all features
        features = VoiceFeatures(
            pitch_mean=pitch_features["mean"],
            pitch_std=pitch_features["std"],
            pitch_range=pitch_features["range"],
            energy_mean=energy_features["mean"],
            energy_std=energy_features["std"],
            energy_max=energy_features["max"],
            speaking_rate=temporal_features["speaking_rate"],
            silence_ratio=temporal_features["silence_ratio"],
            duration=temporal_features["duration"],
            spectral_centroid=spectral_features["centroid"],
            spectral_rolloff=spectral_features["rolloff"],
            emotion_hints=[],  # Will be filled by inference
        )
        
        # Infer emotion hints from features
        features.emotion_hints = self._infer_emotion_hints(features)
        
        return features
    
    def _load_audio(self, audio_data: bytes, mime_type: str) -> tuple[np.ndarray, int]:
        """Load audio data from bytes into numpy array."""
        # Determine file extension from mime type
        ext_map = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/wav": ".wav",
            "audio/wave": ".wav",
            "audio/mp3": ".mp3",
            "audio/mpeg": ".mp3",
            "audio/flac": ".flac",
        }
        ext = ext_map.get(mime_type, ".webm")
        
        # Write to temp file for librosa (some formats need file access)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = Path(tmp.name)
        
        try:
            # Load with librosa
            y, sr = librosa.load(str(tmp_path), sr=None, mono=True)
            return y, sr
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)
    
    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract pitch (fundamental frequency) features."""
        try:
            # Use YIN algorithm for pitch detection
            f0 = librosa.yin(
                y, 
                fmin=50,    # Minimum expected pitch
                fmax=500,   # Maximum expected pitch
                sr=sr
            )
            
            # Filter out unvoiced frames (0 or very low values)
            voiced_f0 = f0[f0 > 50]
            
            if len(voiced_f0) > 0:
                return {
                    "mean": float(np.mean(voiced_f0)),
                    "std": float(np.std(voiced_f0)),
                    "range": float(np.max(voiced_f0) - np.min(voiced_f0)),
                }
            else:
                # Default values if no voiced frames detected
                return {"mean": 150.0, "std": 0.0, "range": 0.0}
                
        except Exception:
            # Fallback to safe defaults
            return {"mean": 150.0, "std": 0.0, "range": 0.0}
    
    def _extract_energy_features(self, y: np.ndarray) -> dict:
        """Extract energy/loudness features."""
        # RMS energy per frame
        rms = librosa.feature.rms(y=y)[0]
        
        return {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms)),
            "max": float(np.max(rms)),
        }
    
    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract temporal features like speaking rate and pauses."""
        duration = len(y) / sr
        
        # Estimate speaking rate using onset detection
        # More onsets = more syllables (approximation)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        num_onsets = len(onset_frames)
        speaking_rate = num_onsets / duration if duration > 0 else 0.0
        
        # Calculate silence ratio using energy threshold
        rms = librosa.feature.rms(y=y)[0]
        silence_threshold = 0.01  # RMS below this = silence
        silence_frames = np.sum(rms < silence_threshold)
        silence_ratio = silence_frames / len(rms) if len(rms) > 0 else 0.0
        
        return {
            "speaking_rate": speaking_rate,
            "silence_ratio": float(silence_ratio),
            "duration": duration,
        }
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract spectral features for voice quality analysis."""
        # Spectral centroid - brightness of sound
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff - high frequency content
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        return {
            "centroid": float(np.mean(centroid)),
            "rolloff": float(np.mean(rolloff)),
        }
    
    def _infer_emotion_hints(self, features: VoiceFeatures) -> list[str]:
        """
        Infer emotion hints from extracted audio features.
        
        This uses rule-based inference based on prosodic research:
        - High pitch + high energy = excited/happy
        - Low pitch + low energy = sad/calm
        - Fast speech = excited/nervous
        - Many pauses = uncertain/thoughtful
        """
        hints = []
        
        # Pitch-based inference
        if features.pitch_mean > self.PITCH_HIGH_THRESHOLD:
            if features.energy_mean > self.ENERGY_HIGH_THRESHOLD:
                hints.append("excited or enthusiastic tone")
            else:
                hints.append("bright or cheerful tone")
        elif features.pitch_mean < self.PITCH_LOW_THRESHOLD:
            if features.energy_mean < self.ENERGY_LOW_THRESHOLD:
                hints.append("sad or melancholic tone")
            else:
                hints.append("calm or serious tone")
        
        # Pitch variability
        if features.pitch_std > self.PITCH_VAR_HIGH_THRESHOLD:
            hints.append("emotionally expressive voice")
        elif features.pitch_std < 10:
            hints.append("monotone or flat delivery")
        
        # Speaking rate
        if features.speaking_rate > self.RATE_FAST_THRESHOLD:
            hints.append("speaking quickly (possibly excited or nervous)")
        elif features.speaking_rate < self.RATE_SLOW_THRESHOLD:
            hints.append("speaking slowly (possibly thoughtful or sad)")
        
        # Silence/pause patterns
        if features.silence_ratio > self.SILENCE_HIGH_THRESHOLD:
            hints.append("many pauses (possibly uncertain or thinking)")
        
        # Energy patterns
        if features.energy_max > 0.15:
            hints.append("strong vocal emphasis detected")
        elif features.energy_mean < self.ENERGY_LOW_THRESHOLD:
            hints.append("soft or quiet voice")
        
        # Voice brightness (spectral centroid)
        if features.spectral_centroid > 2500:
            hints.append("bright/sharp voice quality")
        elif features.spectral_centroid < 1000:
            hints.append("dark/mellow voice quality")
        
        # Default if no specific hints
        if not hints:
            hints.append("neutral speaking tone")
        
        return hints


# Global instance
voice_emotion_service = VoiceEmotionService()


def get_voice_emotion_service() -> VoiceEmotionService:
    """Get voice emotion service instance."""
    return voice_emotion_service

