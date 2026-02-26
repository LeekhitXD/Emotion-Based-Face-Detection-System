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
        self._clahe = None  # cached for FPS

    def _get_clahe(self):
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return self._clahe
        
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
            # Fourth convolutional block (padding='same' so 2x2 spatial size is valid for 3x3 conv)
            layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=reg, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=reg, padding='same'),
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
    
    def predict(self, face_image, use_tta=False, temperature=1.0):
        """Predict emotion from a face image.
        use_tta: if True, average with flipped image (better accuracy, ~2x slower).
        temperature: >1 softens predictions so other emotions can show (reduces one-class dominance).
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Call build_model() or load_weights() first.")
        
        if isinstance(face_image, tf.Tensor):
            face_image = face_image.numpy()
        
        if len(face_image.shape) == 3:
            if face_image.shape[2] == 3:
                face_image = np.dot(face_image[...,:3], [0.2989, 0.5870, 0.1140])
            elif face_image.shape[2] == 1:
                face_image = face_image[:, :, 0]
        
        if len(face_image.shape) != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {face_image.shape}")
        
        face_image = cv2.resize(face_image, (self.input_shape[0], self.input_shape[1]))
        if face_image.dtype != np.uint8:
            face_image = (np.clip(face_image, 0, 1) * 255).astype(np.uint8)
        face_image = self._get_clahe().apply(face_image)
        face_image = face_image.astype(np.float32) / 255.0
        face_batch = face_image.reshape(1, self.input_shape[0], self.input_shape[1], 1)
        
        if use_tta:
            pred_orig = self.model.predict(face_batch, verbose=0)[0]
            pred_flip = self.model.predict(np.flip(face_batch, axis=2), verbose=0)[0]
            predictions = (np.array(pred_orig) + np.array(pred_flip)) / 2.0
        else:
            predictions = np.array(self.model.predict(face_batch, verbose=0)[0], dtype=np.float64)
        
        # Temperature scaling: T>1 softens distribution so other emotions can appear (fixes one-class dominance)
        if temperature != 1.0 and temperature > 0:
            eps = 1e-8
            predictions = np.clip(predictions, eps, 1.0)
            predictions = np.power(predictions, 1.0 / temperature)
            predictions = predictions / predictions.sum()
        
        emotion_idx = int(np.argmax(predictions))
        confidence = float(predictions[emotion_idx])
        return self.EMOTIONS[emotion_idx], confidence, predictions
