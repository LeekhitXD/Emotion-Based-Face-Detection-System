"""
Session Manager – Emotion history tracking per session.
Stores timestamps and emotions for real-time visualization and feedback.
"""

import json
import time
from collections import defaultdict
from pathlib import Path


class SessionManager:
    """Tracks emotion history for the current session and exports to JSON."""

    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, max_history_seconds=300, sample_interval=1.0):
        """
        Args:
            max_history_seconds: Keep at most this many seconds of history (default 5 min).
            sample_interval: Minimum seconds between recorded samples (throttle).
        """
        self.max_history_seconds = max_history_seconds
        self.sample_interval = sample_interval
        self._history = []  # list of {"time": ts, "emotion": str, "confidence": float, "voice": str or None}
        self._start_time = None
        self._last_record_time = 0.0

    def start_session(self):
        """Start a new session (clears previous history)."""
        self._history.clear()
        self._start_time = time.time()
        self._last_record_time = 0.0

    def record(self, emotion, confidence, voice_tone=None):
        """Record one emotion sample (throttled by sample_interval)."""
        now = time.time()
        if self._start_time is None:
            self.start_session()
        if now - self._last_record_time < self.sample_interval:
            return
        self._last_record_time = now
        elapsed = now - self._start_time
        self._history.append({
            "time": round(elapsed, 2),
            "emotion": emotion,
            "confidence": round(float(confidence), 4),
            "voice": voice_tone,
        })
        # Trim to max_history_seconds
        cutoff = now - self.max_history_seconds
        self._history = [h for h in self._history if (self._start_time + h["time"]) >= cutoff]

    def get_history(self):
        """Return full history list (each item: time, emotion, confidence, voice)."""
        return list(self._history)

    def get_recent(self, seconds=30):
        """Return history for the last `seconds` seconds."""
        if not self._start_time:
            return []
        now = time.time()
        cutoff = now - seconds
        return [h for h in self._history if (self._start_time + h["time"]) >= cutoff]

    def get_emotion_distribution(self, last_seconds=None):
        """Return counts per emotion (for last_seconds or full session)."""
        hist = self.get_recent(last_seconds) if last_seconds else self.get_history()
        counts = defaultdict(int)
        for h in hist:
            counts[h["emotion"]] += 1
        return dict(counts)

    def get_dominant_emotion(self, last_seconds=30):
        """Dominant emotion in the last N seconds."""
        dist = self.get_emotion_distribution(last_seconds)
        if not dist:
            return None, 0
        dominant = max(dist, key=dist.get)
        total = sum(dist.values())
        return dominant, dist[dominant] / total if total else 0

    def get_stress_score(self, last_seconds=60):
        """
        Simple stress indicator: higher if negative emotions dominate.
        Returns 0–1 (0=calm, 1=stressed).
        """
        dist = self.get_emotion_distribution(last_seconds)
        if not dist:
            return 0.0
        negative = sum(dist.get(e, 0) for e in ['Angry', 'Sad', 'Fear', 'Disgust'])
        total = sum(dist.values())
        if total == 0:
            return 0.0
        return min(1.0, negative / total * 1.2)  # slight scale

    def end_session(self):
        """Mark session as ended (no-op; use save_to_json to persist)."""
        pass

    def save_to_json(self, filepath=None):
        """Save session history to JSON. Returns path used."""
        if filepath is None:
            filepath = Path("sessions") / f"session_{int(time.time())}.json"
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "start_time": self._start_time,
            "duration_seconds": (time.time() - self._start_time) if self._start_time else 0,
            "samples": self.get_history(),
            "summary": self.get_emotion_distribution(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return str(filepath)
