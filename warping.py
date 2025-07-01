import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf
from typing import List, Tuple, Optional
import math

class ImageWarper:
    """Advanced image warping using various algorithms"""
    
    def __init__(self, method: str = 'TPS'):
        self.method = method
        
    def warp_image(self, image_path: str, source_landmarks: List[Tuple[float, float]], 
                   target_data: dict, strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp image based on landmark correspondences
        
        Args:
            image_path: Path to source image
            source_landmarks: Source landmark positions
            target_data: Target UV/mesh data
            strength: Warping strength multiplier
            
        Returns:
            Tuple of (warped_image, mask)
        """
        try:
            # Load source image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape[:2]
            
            # Extract target landmarks from UV data
            target_landmarks = self.extract_target_landmarks(target_data, width, height)
            
            if len(source_landmarks) != len(target_landmarks):
                # Interpolate to match counts
                target_landmarks = self.interpolate_landmarks(target_landmarks, len(source_landmarks))
            
            # Apply warping based on method
            if self.method == 'TPS':
                warped_image, mask = self.thin_plate_spline_warp(
                    image, source_landmarks, target_landmarks, strength
                )
            elif self.method == 'MLS':
                warped_image, mask = self.moving_least_squares_warp(
                    image, source_landmarks, target_landmarks, strength
                )
            elif self.method == 'RBF':
                warped_image, mask = self.rbf_warp(
                    image, source_landmarks, target_landmarks, strength
                )
            else:
                raise ValueError(f"Unknown warping method: {self.method}")
            
            return warped_image, mask
            
        except Exception as e:
            print(f"Error warping image: {e}")
            # Return original image as fallback
            image = cv2.imread(image_path)
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            return image, mask
    
    def thin_plate_spline_warp(self, image: np.ndarray, 
                               source_points: List[Tuple[float, float]],
                               target_points: List[Tuple[float, float]], 
                               strength: float) -> Tuple[np.ndarray, np.ndarray]:
        """Thin Plate Spline warping"""
        height, width = image.shape[:2]
        
        try:
            # Convert points to numpy arrays
            src_pts = np.array(source_points, dtype=np.float32)
            dst_pts = np.array(target_points, dtype=np.float32)
            
            # Apply strength factor
            if strength != 1.0:
                center = np.mean(src_pts, axis=0)
                dst_pts = center + (dst_pts - center) * strength
            
            # Add corner points for stability
            corners_src = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
            corners_dst = corners_src.copy()
            
            all_src = np.vstack([src_pts, corners_src])
            all_dst = np.vstack([dst_pts, corners_dst])
            
            # Use perspective transform as fallback
            if len(all_src) >= 4:
                # Use the first 4 points for perspective transform
                M = cv2.getPerspectiveTransform(all_src[:4], all_dst[:4])
                warped_image = cv2.warpPerspective(image, M, (width, height))
            else:
                warped_image = image.copy()
            
            # Create mask
            mask = np.ones((height, width), dtype=np.uint8) * 255
            
            return warped_image, mask
            
        except Exception as e:
            print(f"TPS warping failed: {e}")
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    def moving_least_squares_warp(self, image: np.ndarray,
                                  source_points: List[Tuple[float, float]], 
                                  target_points: List[Tuple[float, float]],
                                  strength: float) -> Tuple[np.ndarray, np.ndarray]:
        """Moving Least Squares deformation"""
        height, width = image.shape[:2]
        
        try:
            src_pts = np.array(source_points)
            dst_pts = np.array(target_points)
            
            # Apply strength
            if strength != 1.0:
                center = np.mean(src_pts, axis=0)
                dst_pts = center + (dst_pts - center) * strength
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            coords = np.stack([x.flatten(), y.flatten()], axis=1)
            
            # Simplified MLS - use affine transformation
            if len(src_pts) >= 3:
                # Use affine transform
                M = cv2.getAffineTransform(src_pts[:3].astype(np.float32), 
                                         dst_pts[:3].astype(np.float32))
                warped_image = cv2.warpAffine(image, M, (width, height))
            else:
                warped_image = image.copy()
            
            # Create mask
            mask = np.ones((height, width), dtype=np.uint8) * 255
            
            return warped_image, mask
            
        except Exception as e:
            print(f"MLS warping failed: {e}")
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    def rbf_warp(self, image: np.ndarray,
                 source_points: List[Tuple[float, float]],
                 target_points: List[Tuple[float, float]], 
                 strength: float) -> Tuple[np.ndarray, np.ndarray]:
        """Radial Basis Function warping"""
        height, width = image.shape[:2]
        
        try:
            src_pts = np.array(source_points)
            dst_pts = np.array(target_points)
            
            # Apply strength
            if strength != 1.0:
                center = np.mean(src_pts, axis=0)
                dst_pts = center + (dst_pts - center) * strength
            
            # Simplified RBF - use similarity transform
            if len(src_pts) >= 2:
                # Calculate similarity transform
                src_center = np.mean(src_pts, axis=0)
                dst_center = np.mean(dst_pts, axis=0)
                
                # Translation
                translation = dst_center - src_center
                
                # Apply translation
                M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
                warped_image = cv2.warpAffine(image, M, (width, height))
            else:
                warped_image = image.copy()
            
            # Create mask
            mask = np.ones((height, width), dtype=np.uint8) * 255
            
            return warped_image, mask
            
        except Exception as e:
            print(f"RBF warping failed: {e}")
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    def extract_target_landmarks(self, uv_data: dict, width: int, height: int) -> List[Tuple[float, float]]:
        """Extract target landmark positions from UV data"""
        landmarks = []
        
        # Generate landmarks based on UV layout
        if 'facial_regions' in uv_data:
            for region_name, faces in uv_data['facial_regions'].items():
                for face in faces[:10]:  # Limit to first 10 faces per region
                    if 'uv_center' in face:
                        x = face['uv_center'][0] * width
                        y = face['uv_center'][1] * height
                        landmarks.append((x, y))
        
        # If no landmarks found, create default grid
        if not landmarks:
            for i in range(8):
                for j in range(8):
                    x = (i + 1) * width / 9
                    y = (j + 1) * height / 9
                    landmarks.append((x, y))
                    if len(landmarks) >= 64:
                        break
                if len(landmarks) >= 64:
                    break
        
        return landmarks
    
    def interpolate_landmarks(self, landmarks: List[Tuple[float, float]], target_count: int) -> List[Tuple[float, float]]:
        """Interpolate landmarks to match target count"""
        if len(landmarks) == target_count:
            return landmarks
        
        if len(landmarks) < 2:
            return landmarks * target_count
        
        landmarks_array = np.array(landmarks)
        
        if len(landmarks) > target_count:
            # Downsample
            indices = np.linspace(0, len(landmarks) - 1, target_count, dtype=int)
            return [tuple(landmarks_array[i]) for i in indices]
        else:
            # Upsample using interpolation
            indices = np.linspace(0, len(landmarks) - 1, target_count)
            interpolated = []
            
            for idx in indices:
                if idx == int(idx):
                    interpolated.append(tuple(landmarks_array[int(idx)]))
                else:
                    # Linear interpolation
                    i1, i2 = int(idx), min(int(idx) + 1, len(landmarks) - 1)
                    weight = idx - i1
                    point = landmarks_array[i1] * (1 - weight) + landmarks_array[i2] * weight
                    interpolated.append(tuple(point))
            
            return interpolated