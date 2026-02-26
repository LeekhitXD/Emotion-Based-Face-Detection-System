# Real-time Facial Emotion Recognition System

A deep learning-based system that captures live camera or video input, detects faces, and classifies emotions in real-time using CNN models. The system is optimized for accuracy and low latency and includes an advanced wellness dashboard.

## Features

- **Real-time face detection** using OpenCV's Haar Cascade classifier
- **CNN-based emotion classification** for 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Low latency processing** optimized for real-time performance
- **Live FPS monitoring** to track system performance
- **Configurable confidence threshold** and optional “Unknown” label for low-confidence predictions
- **Camera or video-file input**, optional recording of annotated video, and snapshots on keypress
- **Advanced wellness app** with emotion timeline, stress gauge, session export, and optional voice tone analysis

## Requirements

- Python 3.7+
- Webcam or camera device (for live mode)
- TensorFlow 2.13.0
- OpenCV 4.8.1
- NumPy 1.24.3

## Installation

1. Clone or navigate to the project directory:
```bash
cd face_detection_ntcc
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage – Main Real-time App

### Running with Pre-trained Model (recommended)

If you have a pre-trained model weights file:

```bash
python main.py --weights emotion_model_weights.h5
```

### Running without Pre-trained Model

The system will run but with random predictions (for testing purposes):

```bash
python main.py
```

### Camera vs Video Input

- Live camera (default):
  ```bash
  python main.py --camera 0 --weights emotion_model_weights.h5
  ```

- Video file:
  ```bash
  python main.py --video path/to/video.mp4 --weights emotion_model_weights.h5
  ```

### Recording & Snapshots

- Save annotated video:
  ```bash
  python main.py --weights emotion_model_weights.h5 --save-video output.mp4
  ```
- Save snapshots (press `s` while running):
  ```bash
  python main.py --weights emotion_model_weights.h5 --snapshot-dir snapshots/
  ```

### Confidence Threshold & Privacy Blur

- Minimum confidence for showing a non-Unknown label:
  ```bash
  python main.py --weights emotion_model_weights.h5 --min-confidence 0.6
  ```

- Blur faces before display and saving:
  ```bash
  python main.py --weights emotion_model_weights.h5 --privacy-blur
  ```

### Important Controls

- Press `q` to quit the application.
- Press `s` (when `--snapshot-dir` is set) to save a snapshot.

## Advanced Wellness & Mood Monitor

For a richer wellness-oriented dashboard with session history, stress gauge, and optional voice tone analysis:

```bash
python -m advanced.wellness_app --weights emotion_model_weights.h5
```

With voice tone analysis (requires `sounddevice` and optionally `librosa`):

```bash
python -m advanced.wellness_app --weights emotion_model_weights.h5 --voice
```

## Training Your Own Model

To train the model on the FER2013 dataset:

1. Download the FER2013 dataset from Kaggle.
2. Place the `fer2013.csv` file in the project directory.
3. Run the training script:

```bash
python train_model.py --epochs 50 --batch_size 64 --data fer2013.csv --save emotion_model_weights.h5
```

You can also train from image folders under `data/train`, `data/val`, and `data/test` (class folders `0..6`).

## Model Architecture

The CNN model consists of:
- Multiple convolutional blocks with batch normalization and dropout
- Max pooling layers for dimensionality reduction
- Dense layers with dropout for regularization
- Softmax output layer for 7 emotion classes
- L2 regularization and label smoothing for better generalization

## Project Structure

```text
face_detection_ntcc/
├── main.py                 # Main real-time application (CLI: camera/video, recording, snapshots)
├── face_detector.py        # Face detection module
├── emotion_model.py        # CNN model for emotion classification
├── train_model.py          # Training script
├── evaluate_model.py       # Quick per-class evaluation on test data
├── advanced/               # Wellness & mood monitor (dashboard, sessions, voice)
├── sessions/               # Saved wellness sessions (JSON)
├── data/                   # Optional image datasets (train/val/test)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Troubleshooting

**Camera not opening:**
- Check if the camera index is correct (try `--camera 1` or `--camera 2`).
- Ensure no other application is using the camera.

**Low FPS:**
- Reduce input frame size in `main.py`.
- Use a GPU-enabled TensorFlow installation for faster inference.
- Close other resource-intensive applications.

**Model not loading:**
- Ensure the weights file path is correct.
- Check that the model architecture matches the weights file.

## License

This project is open source and available for educational and research purposes.
