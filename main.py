import cv2
import numpy as np
import time
from face_detector import FaceDetector
from emotion_model import EmotionModel


class EmotionRecognitionSystem:
    """Main system for real-time emotion recognition"""
    
    def __init__(self, camera_index=0, model_weights_path=None):
        """
        Initialize the emotion recognition system
        
        Args:
            camera_index: Camera device index (default: 0)
            model_weights_path: Path to pre-trained model weights (optional)
        """
        self.camera_index = camera_index
        self.face_detector = FaceDetector()
        self.emotion_model = EmotionModel()
        
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
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                # Process each detected face
                emotions = []
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    try:
                        emotion, confidence, _ = self.emotion_model.predict(face_roi)
                        emotions.append((emotion, confidence))
                    except Exception as e:
                        print(f"Error predicting emotion: {e}")
                        emotions.append(("Unknown", 0.0))
                
                # Draw results
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
    
    args = parser.parse_args()
    
    # Create and run system
    system = EmotionRecognitionSystem(
        camera_index=args.camera,
        model_weights_path=args.weights
    )
    
    system.run()


if __name__ == '__main__':
    main()
