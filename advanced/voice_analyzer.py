"""
Voice tone analyzer – multimodal input alongside facial emotion.
Provides a stub and optional real implementation (e.g. volume/pitch or sentiment).
"""

import threading
import queue
import time
import numpy as np

# Optional: use sounddevice + numpy for mic; try import
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class VoiceAnalyzer:
    """
    Analyzes voice tone from microphone for stress/calm/neutral.
    Stub mode: returns 'neutral' when no real analysis is available.
    """

    def __init__(self, sample_rate=16000, chunk_seconds=1.0, use_mic=True):
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_seconds)
        self.use_mic = use_mic and HAS_SOUNDDEVICE
        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self._last_tone = "neutral"
        self._last_confidence = 0.5

    def _analyze_chunk(self, audio: np.ndarray) -> tuple:
        """Return (tone_str, confidence). tone: 'neutral' | 'positive' | 'negative' | 'stressed'."""
        if audio is None or len(audio) < 100:
            return "neutral", 0.5
        if not HAS_LIBROSA:
            # Stub: very simple energy-based guess
            energy = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
            if energy > 0.15:
                return "stressed", 0.4
            return "neutral", 0.5
        try:
            # Simple features: RMS energy, zero-crossing (rough agitation)
            rms = np.sqrt(np.mean(librosa.feature.rms(y=audio.astype(np.float32) / 32768.0)[0]))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio.astype(np.float32) / 32768.0)[0])
            if rms > 0.08 and zcr > 0.08:
                return "stressed", min(0.8, rms + zcr)
            if rms < 0.02:
                return "neutral", 0.6
            return "positive" if rms < 0.06 else "negative", 0.5
        except Exception:
            return "neutral", 0.5

    def _record_loop(self):
        if not self.use_mic:
            return
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize=self.chunk_frames, dtype=np.int16) as stream:
                while self._running:
                    chunk, _ = stream.read(self.chunk_frames)
                    if chunk is not None and len(chunk):
                        tone, conf = self._analyze_chunk(chunk.flatten())
                        self._last_tone = tone
                        self._last_confidence = conf
                        self._queue.put((tone, conf))
        except Exception:
            self._last_tone = "neutral"
            self._last_confidence = 0.5

    def start(self):
        """Start background recording (if mic available)."""
        self._running = True
        if self.use_mic:
            self._thread = threading.Thread(target=self._record_loop, daemon=True)
            self._thread.start()
        else:
            self._last_tone = "neutral"
            self._last_confidence = 0.5

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_latest(self):
        """Non-blocking: return last (tone, confidence)."""
        while not self._queue.empty():
            try:
                self._last_tone, self._last_confidence = self._queue.get_nowait()
            except queue.Empty:
                break
        return self._last_tone, self._last_confidence

    def analyze_audio(self, audio_chunk=None):
        """
        If audio_chunk provided (numpy int16), analyze it.
        Otherwise return last cached (tone, confidence).
        """
        if audio_chunk is not None:
            tone, conf = self._analyze_chunk(np.array(audio_chunk, dtype=np.int16))
            self._last_tone = tone
            self._last_confidence = conf
            return {"tone": tone, "confidence": conf}
        t, c = self.get_latest()
        return {"tone": t, "confidence": c}
