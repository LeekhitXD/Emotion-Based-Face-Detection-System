# Advanced Wellness & Mood Monitor

Real-world application built on the emotion recognition system: **mental health / stress & mood monitoring** with real-time visualization, session history, smart feedback, and optional voice analysis.

## Features

| Feature | Description |
|--------|-------------|
| **Real-time emotion visualization** | Timeline (color-coded segments) + bar chart of emotion distribution + stress gauge (0–100%). |
| **Session history** | Every few seconds the current emotion is recorded. Full history is exported to JSON when you quit (`sessions/session_<timestamp>.json`). |
| **Smart feedback** | Contextual messages: calming suggestions when stress or negative emotions dominate, motivational when positive, neutral when balanced. |
| **Multimodal input** | Optional **voice tone** analysis (run with `--voice`). Uses microphone + simple energy/agitation cues; install `sounddevice` and optionally `librosa` for basic tone (neutral / stressed / positive). |
| **Application focus** | **Wellness & stress monitoring**: track mood over a session, get gentle feedback, and review session summaries later. |

## Quick start

From the **project root** (`face_detection_ntcc`):

```bash
# With trained weights (recommended)
python -m advanced.wellness_app --weights emotion_model_weights.h5

# With voice tone analysis (optional)
python -m advanced.wellness_app --weights emotion_model_weights.h5 --voice
```

Or run as a script from project root:

```bash
python advanced/wellness_app.py --weights emotion_model_weights.h5
```

**Controls:** Press **q** to quit. Session is saved automatically to `sessions/`.

## Layout

- **Left:** Camera feed with face bounding boxes and emotion labels (same as main app) and a short feedback line at the bottom.
- **Right:** Dashboard panel:
  - **Top:** Emotion timeline (recent 60 s), one color per emotion.
  - **Middle:** Stress gauge (0–100%) from proportion of negative emotions.
  - **Bottom:** Bar chart of emotion counts over the last 60 seconds.

## Session export

After quitting, a JSON file is written to `sessions/session_<unix_time>.json`:

- `start_time`, `duration_seconds`
- `samples`: list of `{ time, emotion, confidence, voice }`
- `summary`: emotion counts over the full session

You can use this for simple analytics, logging, or integration with other tools.

## Optional: voice analysis

- Install: `pip install sounddevice` (and optionally `librosa` for slightly better tone cues).
- Run with `--voice` to enable. The last voice tone is included in session samples and can influence feedback when combined with face emotion.

## Module overview

- **session_manager.py** – In-memory emotion history, throttled recording, stress score, JSON export.
- **feedback_engine.py** – Rule-based feedback from recent emotions and stress level.
- **visualization.py** – OpenCV-based timeline, bar chart, and stress gauge.
- **voice_analyzer.py** – Optional mic capture and simple tone classification (stub if no deps).
- **wellness_app.py** – Main loop: camera + `EmotionRecognitionSystem.process_frame()`, session recording, feedback, dashboard, and export.

## Other possible applications

The same building blocks can be reused for:

- **Classroom engagement** – Track dominant emotion over time per student (with consent).
- **Interview analysis** – Record emotion timeline and stress for post-interview review.
- **Stress detection** – Use stress gauge + feedback as a lightweight check-in tool.

To adapt, change the feedback messages in `feedback_engine.py` and the session export path or format in `session_manager.py`.
