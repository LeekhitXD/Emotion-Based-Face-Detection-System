"""
Quick evaluation: run the model on test data and print per-class accuracy.
Use this to see if the model predicts all 7 emotions or is stuck on one/two.
Run from project root: python evaluate_model.py --weights emotion_model_weights.h5 [--data data]
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_model import EmotionModel
import cv2


def load_test_images(data_dir, max_per_class=100):
    """Load images from data/test/0..6."""
    from train_model import _apply_clahe
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_dir):
        return None, None
    X, y = [], []
    for class_id in range(7):
        class_dir = os.path.join(test_dir, str(class_id))
        if not os.path.isdir(class_dir):
            continue
        count = 0
        for fname in os.listdir(class_dir):
            if count >= max_per_class:
                break
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(class_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            img = _apply_clahe(img)
            X.append(img)
            y.append(class_id)
            count += 1
    if not X:
        return None, None
    X = np.expand_dims(np.array(X, dtype=np.float32) / 255.0, axis=-1)
    y = np.array(y)
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="emotion_model_weights.h5", help="Path to .h5 weights")
    p.add_argument("--data", type=str, default="data", help="Data dir with test/0..6")
    p.add_argument("--max", type=int, default=80, help="Max test samples per class")
    args = p.parse_args()

    model = EmotionModel()
    model.build_model()
    if not os.path.isfile(args.weights):
        print(f"Weights not found: {args.weights}")
        return
    model.load_weights(args.weights)

    X, y = load_test_images(args.data, max_per_class=args.max)
    if X is None:
        print("No test data found. Use --data path/to/data (with data/test/0..6)")
        return

    preds = model.model.predict(X, verbose=0)
    pred_idx = np.argmax(preds, axis=1)
    correct = (pred_idx == y)

    print("\n--- Per-class accuracy (model should not be 0% or 100% for all) ---\n")
    for c in range(7):
        mask = y == c
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean() * 100
        total = mask.sum()
        right = correct[mask].sum()
        print(f"  {model.EMOTIONS[c]:12s}: {right:3d}/{total} = {acc:5.1f}%")
    print(f"\n  Overall: {correct.mean()*100:.1f}%")
    print("\nIf one/two classes have very high % and others 0%, retrain with more epochs and --data data.\n")


if __name__ == "__main__":
    main()
