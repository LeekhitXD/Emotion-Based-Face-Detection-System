"""
Feedback Engine – Smart feedback based on detected emotions.
Motivational messages, calming suggestions, and stress-aware tips.
"""

import random
from .session_manager import SessionManager


class FeedbackEngine:
    """Generates contextual feedback from emotion history and current state."""

    CALMING = [
        "Take a slow breath. You're doing okay.",
        "It's okay to take a moment. Try a few deep breaths.",
        "Things can feel heavy sometimes. Be kind to yourself.",
        "A short break might help. Stretch or look away for a bit.",
        "Remember: this moment will pass. You've got this.",
    ]
    MOTIVATIONAL = [
        "You're doing great. Keep going.",
        "Your effort matters. Stay with it.",
        "Nice focus. Keep that energy.",
        "You're on track. Believe in your progress.",
        "One step at a time. You've got this.",
    ]
    NEUTRAL_POSITIVE = [
        "Steady and calm. Good place to be.",
        "You seem focused. Keep it up.",
        "Balanced state. Well done.",
    ]
    STRESS_RELIEF = [
        "Consider a quick break or a glass of water.",
        "If you can, step away for 1–2 minutes. It helps.",
        "Stress is normal. Small pauses can reset your mood.",
    ]

    def __init__(self, session: SessionManager):
        self.session = session
        self._last_feedback_time = 0
        self._min_interval = 8.0  # don't spam feedback
        self._last_message = None

    def get_feedback(self, current_emotion, current_confidence, stress_score=None, voice_tone=None):
        """
        Return one short feedback message based on session and current state.
        Uses session history and optional stress_score (0–1) and voice_tone.
        """
        import time
        now = time.time()
        if stress_score is None:
            stress_score = self.session.get_stress_score(last_seconds=60)
        dominant, dom_ratio = self.session.get_dominant_emotion(last_seconds=30)
        recent = self.session.get_recent(20)

        # Decide category
        negative_emotions = {'Angry', 'Sad', 'Fear', 'Disgust'}
        positive_emotions = {'Happy', 'Surprise'}
        is_negative = current_emotion in negative_emotions or (dominant and dominant in negative_emotions)
        is_positive = current_emotion in positive_emotions or (dominant and dominant in positive_emotions)
        high_stress = stress_score >= 0.5

        # Simple voice-aware adjustment: stressed tone increases perceived stress,
        # positive tone nudges towards motivational feedback.
        if voice_tone:
            vt = voice_tone.lower()
            if vt in {"stressed", "negative"}:
                high_stress = True
            elif vt == "positive":
                is_positive = True

        if high_stress or (is_negative and dom_ratio >= 0.4):
            pool = self.CALMING + self.STRESS_RELIEF
        elif is_positive:
            pool = self.MOTIVATIONAL
        else:
            pool = self.NEUTRAL_POSITIVE

        if now - self._last_feedback_time < self._min_interval and self._last_message:
            return self._last_message
        msg = random.choice(pool)
        self._last_feedback_time = now
        self._last_message = msg
        return msg
