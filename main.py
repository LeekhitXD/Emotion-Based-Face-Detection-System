import cv2
import numpy as np
import time
from collections import deque
from face_detector import FaceDetector
from emotion_model import EmotionModel


def _padded_face_roi(frame, x, y, w, h, pad_ratio=0.15):
    """Expand face bbox by pad_ratio and clamp to frame for better model input."""
    H, W = frame.shape[:2]
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)
    return frame[y1:y2, x1:x2]


class EmotionRecognitionSystem:
    """Main system for real-time emotion recognition"""
    
    def __init__(self, camera_index=0, model_weights_path=None, use_tta=False,
                 emotion_every_n=2, smooth_frames=5):
        """
        Initialize the emotion recognition system.

        Args:
            camera_index: Camera device index (default: 0)
            model_weights_path: Path to pre-trained model weights (optional)
            use_tta: Use test-time augmentation (2x slower, slightly more accurate)
            emotion_every_n: Run emotion model every N frames (1=every frame, 2=half cost)
            smooth_frames: Number of frames to average for stable predictions (0=off)
        """
        self.camera_index = camera_index
        self.face_detector = FaceDetector()
        self.emotion_model = EmotionModel()
        self.use_tta = use_tta
        self.emotion_every_n = max(1, emotion_every_n)
        self.smooth_frames = max(0, smooth_frames)
        self._frame_counter = 0
        self._last_emotions = []  # [(emotion, confidence), ...] reused when skipping
        self._prediction_history = []  # list of deques of prediction vectors per face slot
        
        # Initialize model
        if model_weights_path:
            self.emotion_model.build_model()
            try:
                self.emotion_model.load_weights(model_weights_path)
            except Exception as e:
                print(f"Warning: Could not load weights from {model_weights_path}: {e}")
                print("Using untrained model. Please train the model first.")
        else:
            self.emotion_model.build_model()
            print("Warning: No model weights provided. Using untrained model.")
            print("The system will run but predictions will be random.")
        
        # Performance tracking
        self.fps_history = []
        self.last_fps_time = time.time()
        self.frame_count = 0
        
    def draw_results(self, frame, faces, emotions):
        """
        Draw bounding boxes and emotion labels on frame
        
        Args:
            frame: Input frame
            faces: List of face bounding boxes
            emotions: List of (emotion, confidence) tuples
        """
        for (x, y, w, h), (emotion, confidence) in zip(faces, emotions):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare label text
            label = f"{emotion}: {confidence:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x, y - text_height - baseline - 10),
                (x + text_width, y),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x, y - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
    
    def calculate_fps(self):
        """Calculate and return current FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:  # Update FPS every second
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            
            self.frame_count = 0
            self.last_fps_time = current_time
            return fps
        
        self.frame_count += 1
        if self.fps_history:
            return self.fps_history[-1]
        return 0
    
    def run(self):
        """Run the real-time emotion recognition system"""
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {self.camera_index}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting emotion recognition system...")
        print("Press 'q' to quit")
        
        # Create window explicitly (helps on macOS so the camera window appears)
        cv2.namedWindow('Emotion Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Emotion Recognition', 640, 480)
        if hasattr(cv2, 'startWindowThread'):
            cv2.startWindowThread()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                frame = cv2.flip(frame, 1)
                faces = self.face_detector.detect_faces(frame)
                
                # Run emotion every N frames to boost FPS; reuse last result otherwise
                run_emotion = (self._frame_counter % self.emotion_every_n == 0)
                self._frame_counter += 1
                
                if run_emotion and len(faces) > 0:
                    # Ensure we have a smoothing deque per face slot
                    while len(self._prediction_history) < len(faces):
                        self._prediction_history.append(deque(maxlen=self.smooth_frames))
                    self._prediction_history = self._prediction_history[:len(faces)]
                    
                    emotions = []
                    for i, (x, y, w, h) in enumerate(faces):
                        face_roi = _padded_face_roi(frame, x, y, w, h)
                        if face_roi.size == 0:
                            emotions.append(("Unknown", 0.0))
                            continue
                        try:
                            emotion, confidence, pred_vec = self.emotion_model.predict(
                                face_roi, use_tta=self.use_tta
                            )
                            if self.smooth_frames > 0 and i < len(self._prediction_history):
                                self._prediction_history[i].append(pred_vec)
                                hist = list(self._prediction_history[i])
                                avg_pred = np.mean(hist, axis=0)
                                emotion_idx = int(np.argmax(avg_pred))
                                emotion = self.emotion_model.EMOTIONS[emotion_idx]
                                confidence = float(avg_pred[emotion_idx])
                            emotions.append((emotion, confidence))
                        except Exception as e:
                            print(f"Error predicting emotion: {e}")
                            emotions.append(("Unknown", 0.0))
                    self._last_emotions = emotions
                else:
                    emotions = self._last_emotions if self._last_emotions else []
                    # If face count changed, reuse what we can or pad
                    if len(emotions) != len(faces):
                        emotions = emotions[:len(faces)]
                        while len(emotions) < len(faces):
                            emotions.append(("Unknown", 0.0))
                
                if faces is not None and len(faces) > 0:
                    self.draw_results(frame, faces, emotions)
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Display frame
                cv2.imshow('Emotion Recognition', frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print performance stats
            if self.fps_history:
                avg_fps = np.mean(self.fps_history)
                print(f"\nAverage FPS: {avg_fps:.2f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Facial Emotion Recognition')
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to pre-trained model weights file'
    )
    parser.add_argument(
        '--emotion-every',
        type=int,
        default=2,
        metavar='N',
        help='Run emotion model every N frames (default: 2, higher FPS)'
    )
    parser.add_argument(
        '--smooth',
        type=int,
        default=5,
        metavar='K',
        help='Smooth predictions over last K frames (default: 5, more stable labels)'
    )
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Use test-time augmentation (better accuracy, lower FPS)'
    )
    args = parser.parse_args()
    
    system = EmotionRecognitionSystem(
        camera_index=args.camera,
        model_weights_path=args.weights,
        use_tta=args.tta,
        emotion_every_n=args.emotion_every,
        smooth_frames=args.smooth
    )
    
    system.run()


if __name__ == '__main__':
    main()
