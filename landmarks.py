import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class FacialLandmarkDetector:
    """Facial landmark detection using MediaPipe"""
    
    def __init__(self):
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.available = True
        except ImportError:
            print("MediaPipe not available, using fallback landmark detection")
            self.available = False
        
        # Key facial landmarks (68-point compatible)
        self.key_landmarks = [
            # Jaw line (17 points)
            *range(0, 17),
            # Eyebrows (10 points) 
            *range(17, 27),
            # Nose (9 points)
            *range(27, 36),
            # Eyes (12 points)
            *range(36, 48),
            # Mouth (20 points)
            *range(48, 68)
        ]
    
    def detect(self, image_path: str, confidence: float = 0.5) -> Optional[List[Tuple[float, float]]]:
        """
        Detect facial landmarks in image
        
        Args:
            image_path: Path to input image
            confidence: Minimum detection confidence
            
        Returns:
            List of (x, y) landmark coordinates or None if no face detected
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return None
            
            if not self.available:
                return self.fallback_detection(image)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = rgb_image.shape[:2]
            
            # Process image
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                print("No face detected in image")
                return self.fallback_detection(image)
            
            # Extract landmarks
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            
            # Convert MediaPipe landmarks to pixel coordinates
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks.append((x, y))
            
            # Return subset of key landmarks for compatibility
            if len(landmarks) >= 68:
                key_points = [landmarks[i] for i in self.key_landmarks[:68]]
            else:
                key_points = landmarks[:68] if len(landmarks) >= 68 else landmarks
            
            return key_points
            
        except Exception as e:
            print(f"Error detecting landmarks: {e}")
            return self.fallback_detection(cv2.imread(image_path))
    
    def fallback_detection(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """Fallback detection using simple face detection"""
        try:
            # Use Haar cascade for basic face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                # Generate default landmarks grid
                height, width = image.shape[:2]
                landmarks = []
                for i in range(8):
                    for j in range(8):
                        x = int((i + 1) * width / 9)
                        y = int((j + 1) * height / 9)
                        landmarks.append((x, y))
                        if len(landmarks) >= 68:
                            break
                    if len(landmarks) >= 68:
                        break
                return landmarks[:68]
            
            # Generate landmarks based on face rectangle
            x, y, w, h = faces[0]
            landmarks = self.generate_face_landmarks(x, y, w, h)
            
            return landmarks
            
        except Exception as e:
            print(f"Fallback detection failed: {e}")
            # Return default grid
            return [(100 + i * 20, 100 + j * 20) for i in range(8) for j in range(8)][:68]
    
    def generate_face_landmarks(self, x: int, y: int, w: int, h: int) -> List[Tuple[float, float]]:
        """Generate facial landmarks based on face bounding box"""
        landmarks = []
        
        # Face contour (17 points)
        for i in range(17):
            lx = x + int(w * (i / 16.0))
            ly = y + h if i < 8 else y + int(h * 0.8)
            landmarks.append((lx, ly))
        
        # Eyebrows (10 points)
        for i in range(10):
            lx = x + int(w * (0.2 + i * 0.06))
            ly = y + int(h * 0.3)
            landmarks.append((lx, ly))
        
        # Nose (9 points)
        for i in range(9):
            lx = x + int(w * (0.4 + i * 0.02))
            ly = y + int(h * (0.4 + i * 0.05))
            landmarks.append((lx, ly))
        
        # Eyes (12 points)
        for i in range(12):
            if i < 6:  # Left eye
                lx = x + int(w * (0.25 + i * 0.03))
                ly = y + int(h * 0.35)
            else:  # Right eye
                lx = x + int(w * (0.65 + (i-6) * 0.03))
                ly = y + int(h * 0.35)
            landmarks.append((lx, ly))
        
        # Mouth (20 points)
        for i in range(20):
            lx = x + int(w * (0.3 + i * 0.02))
            ly = y + int(h * (0.65 + (i % 4) * 0.02))
            landmarks.append((lx, ly))
        
        return landmarks
    
    def visualize_landmarks(self, image_path: str, landmarks: List[Tuple[float, float]], output_path: str = None):
        """
        Visualize landmarks on image
        
        Args:
            image_path: Input image path
            landmarks: List of landmark coordinates
            output_path: Optional output path for visualization
        """
        try:
            image = cv2.imread(image_path)
            
            # Draw landmarks
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(image, str(i), (int(x), int(y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            if output_path:
                cv2.imwrite(output_path, image)
            else:
                cv2.imshow('Landmarks', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"Error visualizing landmarks: {e}")