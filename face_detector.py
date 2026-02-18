"""
Face Detection Module
Handles real-time face detection from video frames using OpenCV's Haar Cascade.
"""

import cv2
import numpy as np


class FaceDetector:
    """Face detector using Haar Cascade classifier"""
    
    def __init__(self, cascade_path=None):
        """
        Initialize face detector
        
        Args:
            cascade_path: Path to Haar Cascade XML file. If None, uses default.
        """
        if cascade_path is None:
            # Use OpenCV's built-in Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Could not load cascade from {cascade_path}")
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces: slightly coarser scaleFactor for speed, minSize to skip tiny faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def extract_face(self, frame, bbox):
        """
        Extract face region from frame
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Extracted face image
        """
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        return face


if __name__ == '__main__':
    print("=" * 60)
    print("Face Detector Module")
    print("=" * 60)
    print("\nThis is a module file, not meant to be run directly.")
    print("\nTo run the emotion recognition system, use:")
    print("  python main.py")
    print("\nOr with model weights:")
    print("  python main.py --weights emotion_model_weights.h5")
    print("\nFor help:")
    print("  python main.py --help")
    print("=" * 60)
