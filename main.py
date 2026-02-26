import cv2
import numpy as np
import os
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
                 emotion_every_n=2, smooth_frames=2, temperature=1.0, enable_variety=False,
                 min_confidence=0.0, unknown_label="Unknown"):
        """
        Initialize the emotion recognition system.

        Args:
            camera_index: Camera device index (default: 0)
            model_weights_path: Path to pre-trained model weights (optional)
            use_tta: Use test-time augmentation (2x slower, slightly more accurate)
            emotion_every_n: Run emotion model every N frames (1=every frame, 2=half cost)
            smooth_frames: Frames to average (0=off, 2=light so labels can change)
            temperature: Softmax temperature (1.0 = normal argmax). Values >1 make output less peaky.
            enable_variety: If True, cycles through top emotions when stuck (demo mode, less accurate).
        """
        self.camera_index = camera_index
        self.face_detector = FaceDetector()
        self.emotion_model = EmotionModel()
        self.use_tta = use_tta
        self.emotion_every_n = max(1, emotion_every_n)
        self.smooth_frames = max(0, smooth_frames)
        self.temperature = max(0.1, min(4.0, float(temperature)))
        self.enable_variety = bool(enable_variety)
        self.min_confidence = float(min_confidence)
        self.unknown_label = str(unknown_label)
        self._frame_counter = 0
        self._last_emotions = []
        self._prediction_history = []
        self._last_shown_emotion_idx = []
        self._stuck_count = []
        self._stuck_cycle_offset = []  # used only when enable_variety=True
        
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

    def process_frame(self, frame):
        """
        Process one frame: detect faces, predict emotions, draw on frame.
        Returns (frame_with_drawings, faces, emotions) for use by advanced apps.
        """
        frame = cv2.flip(frame, 1)
        faces = self.face_detector.detect_faces(frame)
        run_emotion = (self._frame_counter % self.emotion_every_n == 0)
        self._frame_counter += 1
        if run_emotion and len(faces) > 0:
            while len(self._prediction_history) < len(faces):
                self._prediction_history.append(deque(maxlen=self.smooth_frames))
            self._prediction_history = self._prediction_history[:len(faces)]
            while len(self._last_shown_emotion_idx) < len(faces):
                self._last_shown_emotion_idx.append(-1)
                self._stuck_count.append(0)
                self._stuck_cycle_offset.append(0)
            self._last_shown_emotion_idx = self._last_shown_emotion_idx[:len(faces)]
            self._stuck_count = self._stuck_count[:len(faces)]
            self._stuck_cycle_offset = self._stuck_cycle_offset[:len(faces)]
            emotions = []
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = _padded_face_roi(frame, x, y, w, h)
                if face_roi.size == 0:
                    emotions.append(("Unknown", 0.0))
                    continue
                try:
                    emotion, confidence, pred_vec = self.emotion_model.predict(
                        face_roi, use_tta=self.use_tta, temperature=self.temperature
                    )
                    pred_vec = np.array(pred_vec, dtype=np.float64)
                    probs = pred_vec.copy()
                    if self.smooth_frames > 0 and i < len(self._prediction_history):
                        hist = self._prediction_history[i]
                        current_argmax = int(np.argmax(pred_vec))
                        if len(hist) > 0:
                            avg_so_far = np.mean(list(hist), axis=0)
                            smoothed_argmax = int(np.argmax(avg_so_far))
                            if current_argmax != smoothed_argmax:
                                hist.clear()
                        hist.append(pred_vec.copy())
                        avg_pred = np.mean(list(hist), axis=0)
                        probs = avg_pred
                        emotion_idx = int(np.argmax(probs))
                        confidence = float(probs[emotion_idx])
                    else:
                        emotion_idx = int(np.argmax(probs))
                        confidence = float(probs[emotion_idx])
                    last_idx = self._last_shown_emotion_idx[i]
                    if emotion_idx == last_idx:
                        self._stuck_count[i] = self._stuck_count[i] + 1
                    else:
                        self._stuck_count[i] = 0
                        self._stuck_cycle_offset[i] = 0
                    self._last_shown_emotion_idx[i] = emotion_idx
                    # Demo/variety mode only: cycle through top emotions when stuck.
                    # For accurate predictions, keep enable_variety=False (default) and use pure argmax.
                    if self.enable_variety and self._stuck_count[i] >= 4:
                        order = np.argsort(probs)[::-1]  # best first
                        top_k = order[:4]
                        self._stuck_cycle_offset[i] = (self._stuck_cycle_offset[i] + 1) % 4
                        emotion_idx = int(top_k[self._stuck_cycle_offset[i]])
                        confidence = float(probs[emotion_idx])
                        self._last_shown_emotion_idx[i] = emotion_idx
                        self._stuck_count[i] = 0
                    emotion = self.emotion_model.EMOTIONS[emotion_idx]
                    if confidence < self.min_confidence:
                        emotion = self.unknown_label
                    emotions.append((emotion, confidence))
                except Exception:
                    emotions.append(("Unknown", 0.0))
            self._last_emotions = emotions
        else:
            emotions = self._last_emotions if self._last_emotions else []
            if len(emotions) != len(faces):
                emotions = emotions[:len(faces)]
                while len(emotions) < len(faces):
                    emotions.append(("Unknown", 0.0))
        if faces is not None and len(faces) > 0:
            self.draw_results(frame, faces, emotions)
        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame, faces, emotions
    
    def run(self, video_path=None, save_video_path=None, snapshot_dir=None, privacy_blur=False):
        """Run the real-time emotion recognition system.
        
        Args:
            video_path: Optional path to a video file. If None, use live camera.
            save_video_path: Optional path to save an annotated video. If None, don't save.
            snapshot_dir: Optional directory to save snapshots when 's' is pressed.
            privacy_blur: If True, blur detected faces before display/save.
        """
        # Initialize capture source
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {video_path}")
        else:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise ValueError(f"Could not open camera {self.camera_index}")
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting emotion recognition system...")
        print("Press 'q' to quit, 's' to save a snapshot.")
        
        # Create window explicitly (helps on macOS so the camera window appears)
        window_name = 'Emotion Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        if hasattr(cv2, 'startWindowThread'):
            cv2.startWindowThread()
        
        # Optional video writer (initialized lazily after first frame so it works with cameras too)
        video_writer = None
        
        # Ensure snapshot directory exists if requested
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or failed to grab frame")
                    break
                frame, faces, emotions = self.process_frame(frame)
                # Lazily initialize video writer once we know frame size
                if video_writer is None and save_video_path:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(save_video_path, fourcc, 20.0, (w, h))
                if privacy_blur and faces is not None and len(faces) > 0:
                    for (x, y, w, h) in faces:
                        roi = frame[y:y+h, x:x+w]
                        if roi.size == 0:
                            continue
                        roi_blur = cv2.GaussianBlur(roi, (31, 31), 0)
                        frame[y:y+h, x:x+w] = roi_blur
                cv2.imshow(window_name, frame)
                
                if video_writer is not None:
                    video_writer.write(frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s') and snapshot_dir:
                    ts = int(time.time())
                    snapshot_path = os.path.join(snapshot_dir, f"snapshot_{ts}.png")
                    cv2.imwrite(snapshot_path, frame)
                    print(f"Saved snapshot to {snapshot_path}")
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
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
        default=2,
        metavar='K',
        help='Smooth over last K frames (default: 2; use 0 for no smoothing, more emotion changes)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        metavar='T',
        help='Softmax temperature (default: 1.0). >1 makes output less peaky but can look random.'
    )
    parser.add_argument(
        '--variety',
        action='store_true',
        help='Demo mode: cycle through top emotions when stuck (less accurate).'
    )
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Use test-time augmentation (better accuracy, lower FPS)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.0,
        metavar='C',
        help='Minimum confidence for showing a non-Unknown label (default: 0.0)'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Path to a video file instead of live camera'
    )
    parser.add_argument(
        '--save-video',
        type=str,
        default=None,
        help='Path to save annotated video (e.g., output.mp4)'
    )
    parser.add_argument(
        '--snapshot-dir',
        type=str,
        default=None,
        help='Directory to save snapshots when pressing "s"'
    )
    parser.add_argument(
        '--privacy-blur',
        action='store_true',
        help='Blur detected faces before display and saving'
    )
    args = parser.parse_args()
    
    system = EmotionRecognitionSystem(
        camera_index=args.camera,
        model_weights_path=args.weights,
        use_tta=args.tta,
        emotion_every_n=args.emotion_every,
        smooth_frames=args.smooth,
        temperature=args.temperature,
        enable_variety=args.variety,
        min_confidence=args.min_confidence
    )
    
    system.run(
        video_path=args.video,
        save_video_path=args.save_video,
        snapshot_dir=args.snapshot_dir,
        privacy_blur=args.privacy_blur,
    )


if __name__ == '__main__':
    main()
