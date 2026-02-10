"""
Training script for the emotion recognition CNN model
This script trains the model on the FER2013 dataset or custom data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from emotion_model import EmotionModel
import os


def load_fer2013_data(csv_path='fer2013.csv'):
    """
    Load FER2013 dataset from CSV file
    
    The CSV should have format: emotion,pixels,usage
    where emotion is 0-6, pixels is space-separated pixel values, usage is Train/PublicTest/PrivateTest
    """
    if not os.path.exists(csv_path):
        print(f"FER2013 CSV file not found at {csv_path}")
        print("Please download the FER2013 dataset from:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        return None, None, None, None, None, None
    
    print("Loading FER2013 dataset...")
    data = []
    labels = []
    
    with open(csv_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            emotion, pixels, usage = line.strip().split(',')
            emotion = int(emotion)
            pixels = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
            pixels = pixels.reshape(48, 48, 1)
            
            data.append(pixels)
            labels.append(emotion)
    
    data = np.array(data, dtype=np.float32) / 255.0
    labels = np.array(labels)
    
    # Split into train, validation, and test sets
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    test_data = []
    test_labels = []
    
    with open(csv_path, 'r') as f:
        next(f)  # Skip header
        idx = 0
        for line in f:
            emotion, pixels, usage = line.strip().split(',')
            emotion = int(emotion)
            pixels = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
            pixels = pixels.reshape(48, 48, 1)
            
            if usage == 'Training':
                train_data.append(pixels)
                train_labels.append(emotion)
            elif usage == 'PublicTest':
                val_data.append(pixels)
                val_labels.append(emotion)
            else:  # PrivateTest
                test_data.append(pixels)
                test_labels.append(emotion)
    
    train_data = np.array(train_data, dtype=np.float32) / 255.0
    train_labels = keras.utils.to_categorical(np.array(train_labels), 7)
    val_data = np.array(val_data, dtype=np.float32) / 255.0
    val_labels = keras.utils.to_categorical(np.array(val_labels), 7)
    test_data = np.array(test_data, dtype=np.float32) / 255.0
    test_labels = keras.utils.to_categorical(np.array(test_labels), 7)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def train_model(epochs=50, batch_size=64, data_path='fer2013.csv', save_path='emotion_model_weights.h5'):
    """Train the emotion recognition model"""
    
    # Load data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_fer2013_data(data_path)
    
    if train_data is None:
        print("Cannot train without data. Please provide FER2013 dataset.")
        return
    
    # Build model
    model = EmotionModel()
    model.build_model()
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.model.fit(
        datagen.flow(train_data, train_labels, batch_size=batch_size),
        steps_per_epoch=len(train_data) // batch_size,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.model.evaluate(test_data, test_labels, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    print(f"\nModel saved to {save_path}")
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data', type=str, default='fer2013.csv', help='Path to FER2013 CSV file')
    parser.add_argument('--save', type=str, default='emotion_model_weights.h5', help='Path to save model weights')
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data,
        save_path=args.save
    )
