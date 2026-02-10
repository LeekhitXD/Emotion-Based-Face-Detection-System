# Quick Start Guide

## The Issue You Encountered

If you tried to run `face_detector.py` directly and got an error (or nothing happened), that's because it's a **module file**, not a standalone script. You need to run `main.py` instead.

## Step-by-Step Setup

### 1. Install Dependencies

First, make sure all required packages are installed:

```bash
pip install -r requirements.txt
```

Or if you're using pip3:

```bash
pip3 install -r requirements.txt
```

### 2. Run the Main Application

**DO NOT run `face_detector.py` or `emotion_model.py` directly.** These are module files.

Instead, run:

```bash
python main.py
```

Or:

```bash
python3 main.py
```

### 3. Using with Pre-trained Model (Recommended)

For accurate predictions, you'll need a trained model. Run:

```bash
python main.py --weights emotion_model_weights.h5
```

### 4. Check Available Options

To see all available command-line options:

```bash
python main.py --help
```

## Common Issues

### Issue: "No module named 'cv2'"
**Solution:** Install OpenCV:
```bash
pip install opencv-python
```

### Issue: "No module named 'tensorflow'"
**Solution:** Install TensorFlow:
```bash
pip install tensorflow
```

### Issue: Camera not opening
**Solution:** Try a different camera index:
```bash
python main.py --camera 1
```

### Issue: Model weights not found
**Solution:** The system will still run but with random predictions. Train a model first using `train_model.py` or download pre-trained weights.

## What Each File Does

- **`main.py`** - **RUN THIS FILE** - Main application
- **`face_detector.py`** - Module for face detection (don't run directly)
- **`emotion_model.py`** - Module for emotion classification (don't run directly)
- **`train_model.py`** - Script to train the model (run separately when needed)

## Testing the System

1. Make sure your camera is connected and working
2. Run: `python main.py`
3. You should see a window with your camera feed
4. The system will detect faces and show emotion labels
5. Press `q` to quit
