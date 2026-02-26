"""
Real-time emotion visualization: timeline and bar chart.
Draws on OpenCV mats for in-app display.
"""

import numpy as np
import cv2
from .session_manager import SessionManager


# BGR colors per emotion (for timeline and bars)
EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 128),
    "Fear": (128, 128, 128),
    "Happy": (0, 255, 0),
    "Sad": (255, 0, 0),
    "Surprise": (0, 255, 255),
    "Neutral": (200, 200, 200),
}


def draw_timeline(panel, session: SessionManager, width, height, last_seconds=60):
    """
    Draw a horizontal timeline: each segment = one sample, color = emotion.
    panel: BGR image (e.g. 400x120)
    """
    hist = session.get_recent(last_seconds)
    if not hist:
        cv2.putText(panel, "No data yet", (10, height // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return
    n = len(hist)
    segment_w = max(2, (width - 20) // n)
    y1, y2 = 10, height - 25
    for i, h in enumerate(hist):
        x1 = 10 + i * segment_w
        x2 = min(x1 + segment_w, width - 10)
        color = EMOTION_COLORS.get(h["emotion"], (150, 150, 150))
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
    cv2.putText(panel, "Emotion timeline (recent)", (10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_emotion_bars(panel, session: SessionManager, width, height, last_seconds=60):
    """
    Draw horizontal bar chart: one bar per emotion (count in last_seconds).
    panel: BGR image
    """
    dist = session.get_emotion_distribution(last_seconds)
    emotions = list(EMOTION_COLORS.keys())
    max_count = max(dist.values()) if dist else 1
    bar_h = max(12, (height - 40) // 7)
    y = 20
    for em in emotions:
        count = dist.get(em, 0)
        bar_w = int((width - 80) * (count / max_count)) if max_count else 0
        color = EMOTION_COLORS.get(em, (150, 150, 150))
        cv2.putText(panel, em[:6], (5, y + bar_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        cv2.rectangle(panel, (70, y), (70 + bar_w, y + bar_h - 2), color, -1)
        cv2.putText(panel, str(count), (72 + bar_w, y + bar_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += bar_h
    cv2.putText(panel, "Distribution (last 60s)", (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_stress_gauge(panel, stress_score, width, height):
    """Draw a simple 0–100% stress gauge (horizontal bar)."""
    cv2.putText(panel, "Stress", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    bar_w = int((width - 20) * stress_score)
    cv2.rectangle(panel, (10, 18), (width - 10, 28), (60, 60, 60), -1)
    cv2.rectangle(panel, (10, 18), (10 + bar_w, 28), (0, 0, 255), -1)
    cv2.putText(panel, f"{int(stress_score*100)}%", (width - 35, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def build_dashboard_panel(session: SessionManager, width=400, height=280):
    """
    Build one BGR image containing timeline + emotion bars + stress gauge.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    # Timeline (top)
    th = 80
    draw_timeline(panel[:th, :], session, width, th, last_seconds=60)
    # Stress (small strip)
    stress = session.get_stress_score(60)
    strip = panel[th:th+40, :]
    strip[:] = (35, 35, 35)
    draw_stress_gauge(strip, stress, width, 40)
    # Bars (rest)
    bh = height - th - 40
    draw_emotion_bars(panel[th+40:, :], session, width, bh, last_seconds=60)
    return panel
