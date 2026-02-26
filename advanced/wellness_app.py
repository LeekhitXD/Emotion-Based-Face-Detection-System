"""
Wellness & Mood Monitor – Advanced real-world application.

Use case: Mental health / stress & mood monitoring with:
- Real-time emotion visualization (timeline + distribution chart + stress gauge)
- Session emotion history and export to JSON
- Smart feedback (motivational messages, calming suggestions)
- Optional voice tone analysis (multimodal)

Run from project root: python -m advanced.wellness_app [--weights path]
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Ensure project root is on path when run as script or -m
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main import EmotionRecognitionSystem

try:
    from .session_manager import SessionManager
    from .feedback_engine import FeedbackEngine
    from .visualization import build_dashboard_panel
    from .voice_analyzer import VoiceAnalyzer
except ImportError:
    from advanced.session_manager import SessionManager
    from advanced.feedback_engine import FeedbackEngine
    from advanced.visualization import build_dashboard_panel
    from advanced.voice_analyzer import VoiceAnalyzer


def run_wellness_app(camera_index=0, weights_path=None, use_voice=False):
    session = SessionManager(max_history_seconds=300, sample_interval=1.0)
    session.start_session()
    feedback_engine = FeedbackEngine(session)
    voice = VoiceAnalyzer(use_mic=use_voice)
    voice.start()

    emotion_system = EmotionRecognitionSystem(
        camera_index=camera_index,
        model_weights_path=weights_path,
        use_tta=False,
        emotion_every_n=2,
        smooth_frames=2,
        temperature=2.5,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    panel_w, panel_h = 420, 300
    window_name = "Wellness & Mood Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Wellness & Mood Monitor")
    print("Press 'q' to quit. Session will be saved to sessions/")
    print("Feedback is shown below the camera based on your detected mood.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, faces, emotions = emotion_system.process_frame(frame)
            h, w = frame.shape[:2]

            # Primary emotion for session and feedback (first face)
            if emotions:
                em, conf = emotions[0]
                voice_tone = voice.get_latest()[0] if use_voice else None
                session.record(em, conf, voice_tone=voice_tone)
                stress = session.get_stress_score(60)
                msg = feedback_engine.get_feedback(em, conf, stress_score=stress, voice_tone=voice_tone)
                # Draw feedback text on camera frame (wrapped)
                y0 = h - 70
                for i, line in enumerate(msg.split("\n")[:2] or [msg]):
                    cv2.putText(frame, line[:50], (10, y0 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
            else:
                session.record("Neutral", 0.5)  # no face = neutral for timeline continuity

            # Build dashboard and show side-by-side
            panel = build_dashboard_panel(session, width=panel_w, height=panel_h)
            combined = np.hstack([frame, panel])
            cv2.imshow(window_name, combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        voice.stop()
        out_path = session.save_to_json()
        print(f"Session saved to {out_path}")
        cv2.destroyAllWindows()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Wellness & Mood Monitor – emotion tracking and feedback")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--weights", type=str, default=None, help="Path to emotion model weights (.h5)")
    p.add_argument("--voice", action="store_true", help="Enable voice tone analysis (requires sounddevice)")
    args = p.parse_args()
    run_wellness_app(camera_index=args.camera, weights_path=args.weights, use_voice=args.voice)


if __name__ == "__main__":
    main()
