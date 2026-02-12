"""
Training script for the emotion recognition CNN model
This script trains the model on the FER2013 dataset or custom data.
Supports: (1) FER2013 CSV file, (2) Image folders: data/train/0..6, data/val/0..6, data/test/0..6
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from emotion_model import EmotionModel
import os
import cv2

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.utils.class_weight import compute_class_weight
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# CLAHE for consistent preprocessing (improves accuracy)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(img):
    """Apply CLAHE to a single grayscale image (uint8 or float in [0,1])."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return _CLAHE.apply(img)


def load_data_from_dirs(data_dir, img_size=(48, 48), use_clahe=True):
    """
    Load dataset from directory structure: data_dir/train/0..6, data_dir/val/0..6, data_dir/test/0..6
    Class folders 0-6 must match EmotionModel.EMOTIONS order: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"Expected {train_dir} and {val_dir} to exist.")
        return None, None, None, None, None, None

    def load_split(split_dir):
        images, labels = [], []
        for class_id in range(7):
            class_dir = os.path.join(split_dir, str(class_id))
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                path = os.path.join(class_dir, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                if use_clahe:
                    img = _apply_clahe(img)
                images.append(img)
                labels.append(class_id)
        return np.array(images, dtype=np.float32) / 255.0, np.array(labels)

    print("Loading dataset from directories...")
    train_x, train_y = load_split(train_dir)
    val_x, val_y = load_split(val_dir)
    if os.path.isdir(test_dir):
        test_x, test_y = load_split(test_dir)
    else:
        test_x, test_y = val_x, val_y

    # Add channel dimension (48, 48) -> (48, 48, 1)
    train_x = np.expand_dims(train_x, axis=-1)
    val_x = np.expand_dims(val_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)
    train_y = keras.utils.to_categorical(train_y, 7)
    val_y = keras.utils.to_categorical(val_y, 7)
    test_y = keras.utils.to_categorical(test_y, 7)

    print(f"Training samples: {len(train_x)}")
    print(f"Validation samples: {len(val_x)}")
    print(f"Test samples: {len(test_x)}")
    return train_x, train_y, val_x, val_y, test_x, test_y


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
            pixels = pixels.reshape(48, 48)
            pixels = _apply_clahe(pixels).reshape(48, 48, 1)
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
    
    # Load data: directory (data/train, data/val, data/test) or FER2013 CSV
    if os.path.isdir(data_path):
        train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data_from_dirs(data_path)
    else:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = load_fer2013_data(data_path)
    
    if train_data is None:
        print("Cannot train without data. Use --data path/to/fer2013.csv or --data data (folder with train/val/test).")
        return

    # Class weights for imbalanced FER (improves accuracy on rare emotions)
    class_weight_dict = None
    if HAS_SKLEARN:
        train_labels_int = np.argmax(train_labels, axis=1)
        classes = np.unique(train_labels_int)
        weights = compute_class_weight('balanced', classes=classes, y=train_labels_int)
        class_weight_dict = dict(zip(classes, weights))
        print("Using balanced class weights for imbalanced data.")
    else:
        print("Install scikit-learn for class weights (pip install scikit-learn).")
    
    # Build model
    model = EmotionModel()
    model.build_model()
    
    # Data augmentation (optional; requires scipy for some transforms)
    use_augmentation = HAS_SCIPY
    if use_augmentation:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
    else:
        datagen = None
        print("Training without augmentation (install scipy for augmentation).")
    
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
    fit_kw = dict(
        epochs=epochs,
        validation_data=(val_data, val_labels),
        callbacks=callbacks,
        verbose=1,
    )
    if class_weight_dict is not None:
        fit_kw['class_weight'] = class_weight_dict
    if use_augmentation and datagen is not None:
        steps = max(1, len(train_data) // batch_size)
        fit_kw['steps_per_epoch'] = steps
        history = model.model.fit(
            datagen.flow(train_data, train_labels, batch_size=batch_size),
            **fit_kw
        )
    else:
        fit_kw['batch_size'] = batch_size
        fit_kw['x'] = train_data
        fit_kw['y'] = train_labels
        history = model.model.fit(**fit_kw)
    
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
    parser.add_argument('--data', type=str, default='data', help='Path to data: FER2013 CSV file or directory with train/val/test and class folders 0-6')
    parser.add_argument('--save', type=str, default='emotion_model_weights.h5', help='Path to save model weights')
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data,
        save_path=args.save
    )
