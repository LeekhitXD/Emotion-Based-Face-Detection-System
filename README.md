<<<<<<< HEAD
# Real-time Facial Emotion Recognition System

A deep learning-based system that captures live camera input, detects faces, and classifies emotions in real-time using CNN models. The system is optimized for accuracy and low latency.

## Features

- **Real-time face detection** using OpenCV's Haar Cascade classifier
- **CNN-based emotion classification** for 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Low latency processing** optimized for real-time performance
- **Live FPS monitoring** to track system performance
- **Easy to use** with simple command-line interface

## Requirements

- Python 3.7+
- Webcam or camera device
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

## Usage

### Running with Pre-trained Model

If you have a pre-trained model weights file:

```bash
python main.py --weights emotion_model_weights.h5
```

### Running without Pre-trained Model

The system will run but with random predictions (for testing purposes):

```bash
python main.py
```

### Command-line Options

- `--camera`: Camera device index (default: 0)
- `--weights`: Path to pre-trained model weights file

Example:
```bash
python main.py --camera 0 --weights emotion_model_weights.h5
```

### Controls

- Press `q` to quit the application

## Training Your Own Model

To train the model on the FER2013 dataset:

1. Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Place the `fer2013.csv` file in the project directory
3. Run the training script:

```bash
python train_model.py --epochs 50 --batch_size 64 --data fer2013.csv --save emotion_model_weights.h5
```

### Training Options

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 64)
- `--data`: Path to FER2013 CSV file (default: fer2013.csv)
- `--save`: Path to save model weights (default: emotion_model_weights.h5)

## Model Architecture

The CNN model consists of:
- 3 convolutional blocks with batch normalization and dropout
- Max pooling layers for dimensionality reduction
- Fully connected layers with dropout for regularization
- Softmax output layer for 7 emotion classes

## Performance Optimization

The system is optimized for low latency through:
- Efficient face detection with optimized Haar Cascade parameters
- Batch processing of predictions
- Frame size optimization (640x480)
- FPS monitoring for performance tracking

## Project Structure

```
face_detection_ntcc/
├── main.py                 # Main application
├── face_detector.py        # Face detection module
├── emotion_model.py        # CNN model for emotion classification
├── train_model.py          # Training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Notes

- The system requires a trained model for accurate predictions. Without pre-trained weights, predictions will be random.
- For best results, ensure good lighting and face the camera directly.
- The system works best with a single face in the frame, though it can handle multiple faces.

## Troubleshooting

**Camera not opening:**
- Check if the camera index is correct (try `--camera 1` or `--camera 2`)
- Ensure no other application is using the camera

**Low FPS:**
- Reduce input frame size in `main.py`
- Use a GPU-enabled TensorFlow installation for faster inference
- Close other resource-intensive applications

**Model not loading:**
- Ensure the weights file path is correct
- Check that the model architecture matches the weights file

## License

This project is open source and available for educational and research purposes.
=======
# Emotion-Based-Face-Detection-System
tym pass
>>>>>>> c2b076ba60f741d5887ce344bc7b686d7e6b5a58
