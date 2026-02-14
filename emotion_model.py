"""
CNN Model for Emotion Classification
This module contains the CNN architecture for facial emotion recognition.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# L2 regularization for better generalization
L2_REG = 1e-4


class EmotionModel:
    """CNN model for emotion classification"""
    
    # Emotion labels
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build the CNN architecture for emotion recognition (deeper + L2 for higher accuracy)"""
        reg = regularizers.l2(L2_REG)
        model = Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg, input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=reg),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Fourth convolutional block (extra capacity)
            layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=reg),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # Global average pooling (reduces overfitting, improves generalization)
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu', kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_weights(self, weights_path):
        """Load pre-trained weights"""
        if self.model is None:
            self.build_model()
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    
    def predict(self, face_image):
        """Predict emotion from a face image"""
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load_weights() first.")
        
        # Convert to numpy array if needed
        if isinstance(face_image, tf.Tensor):
            face_image = face_image.numpy()
        
        # Handle different input formats
        if len(face_image.shape) == 3:
            if face_image.shape[2] == 3:
                # Convert BGR to grayscale (OpenCV uses BGR)
                face_image = np.dot(face_image[...,:3], [0.2989, 0.5870, 0.1140])
            elif face_image.shape[2] == 1:
                face_image = face_image[:, :, 0]
        
        # Ensure it's 2D
        if len(face_image.shape) != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {face_image.shape}")
        
        # Resize to model input size
        face_image = cv2.resize(face_image, (self.input_shape[0], self.input_shape[1]))
        # CLAHE (match training preprocessing for better accuracy)
        if face_image.dtype != np.uint8:
            face_image = (np.clip(face_image, 0, 1) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_image = clahe.apply(face_image)
        # Normalize to [0, 1]
        face_image = face_image.astype(np.float32) / 255.0
        
        # Reshape to add channel and batch dimensions: (1, 48, 48, 1)
        face_batch = face_image.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        
        # Test-time augmentation: average predictions on original + horizontally flipped (boosts accuracy)
        pred_orig = self.model.predict(face_batch, verbose=0)[0]
        face_flip = np.flip(face_batch, axis=2)  # flip width
        pred_flip = self.model.predict(face_flip, verbose=0)[0]
        predictions = (np.array(pred_orig) + np.array(pred_flip)) / 2.0
        
        emotion_idx = int(np.argmax(predictions))
        confidence = float(predictions[emotion_idx])
        return self.EMOTIONS[emotion_idx], confidence, predictions
