import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional
import os

class ImageProcessor:
    """Utility class for image processing operations"""
    
    def __init__(self):
        self.current_image = None
        self.image_path = None
    
    def load_image(self, image_path: str) -> bool:
        """Load image from file path"""
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return False
            
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                print(f"Failed to load image: {image_path}")
                return False
            
            self.image_path = image_path
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def generate_normal_map(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Generate normal map from height information"""
        try:
            # Convert to grayscale for height map
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to smooth the heightmap
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Normalize gradients
            grad_x = grad_x / 255.0 * strength
            grad_y = grad_y / 255.0 * strength
            
            # Calculate normal vectors
            # X component (red channel)
            normal_x = grad_x
            # Y component (green channel) 
            normal_y = -grad_y  # Flip Y for correct normal mapping
            # Z component (blue channel)
            normal_z = np.ones_like(gray)
            
            # Normalize the normal vectors
            length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
            normal_x /= length
            normal_y /= length
            normal_z /= length
            
            # Convert to 0-255 range
            normal_x = ((normal_x + 1) * 127.5).astype(np.uint8)
            normal_y = ((normal_y + 1) * 127.5).astype(np.uint8)
            normal_z = ((normal_z + 1) * 127.5).astype(np.uint8)
            
            # Combine channels (BGR format)
            normal_map = cv2.merge([normal_z, normal_y, normal_x])
            
            return normal_map
            
        except Exception as e:
            print(f"Error generating normal map: {e}")
            return image
    
    def generate_roughness_map(self, image: np.ndarray) -> np.ndarray:
        """Generate roughness map from image details"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Calculate local variance as roughness indicator
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
            
            # Normalize variance to 0-255 range
            roughness = cv2.normalize(local_variance, None, 0, 255, cv2.NORM_MINMAX)
            roughness = roughness.astype(np.uint8)
            
            # Invert so smooth areas are dark (low roughness)
            roughness = 255 - roughness
            
            # Apply some smoothing
            roughness = cv2.GaussianBlur(roughness, (3, 3), 0)
            
            # Convert to 3-channel image
            roughness_map = cv2.cvtColor(roughness, cv2.COLOR_GRAY2BGR)
            
            return roughness_map
            
        except Exception as e:
            print(f"Error generating roughness map: {e}")
            return image
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance overall image quality"""
        try:
            # Convert to PIL for better enhancement tools
            pil_image = self.cv2_to_pil(image)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            enhanced = enhancer.enhance(1.2)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Enhance color saturation
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            # Apply subtle unsharp mask
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            # Convert back to OpenCV format
            result = self.pil_to_cv2(enhanced)
            
            return result
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image
    
    def normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """Normalize lighting conditions in the image"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            
            # Convert back to BGR
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return result
            
        except Exception as e:
            print(f"Error normalizing lighting: {e}")
            return image
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        try:
            # Use Non-local Means Denoising
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            print(f"Error removing noise: {e}")
            return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                     preserve_aspect: bool = True) -> np.ndarray:
        """Resize image to target size"""
        try:
            height, width = image.shape[:2]
            target_width, target_height = target_size
            
            if preserve_aspect:
                # Calculate scaling factor to fit within target size
                scale = min(target_width / width, target_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize image
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Create padded image if needed
                if new_width != target_width or new_height != target_height:
                    # Create blank image with target size
                    if len(image.shape) == 3:
                        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    else:
                        padded = np.zeros((target_height, target_width), dtype=np.uint8)
                    
                    # Center the resized image
                    y_offset = (target_height - new_height) // 2
                    x_offset = (target_width - new_width) // 2
                    
                    if len(image.shape) == 3:
                        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
                    else:
                        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
                    
                    return padded
                else:
                    return resized
            else:
                # Direct resize without preserving aspect ratio
                return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
                
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image
    
    def cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL Image"""
        if len(cv2_image.shape) == 3:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        else:
            return Image.fromarray(cv2_image)
    
    def pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV image"""
        cv2_image = np.array(pil_image)
        if len(cv2_image.shape) == 3:
            # Convert RGB to BGR
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        return cv2_image
    
    def create_edge_padding(self, image: np.ndarray, padding_size: int = 4) -> np.ndarray:
        """Add edge padding for seamless texture mapping"""
        try:
            height, width = image.shape[:2]
            
            if len(image.shape) == 3:
                padded = np.zeros((height + padding_size * 2, width + padding_size * 2, 3), dtype=np.uint8)
                padded[padding_size:height+padding_size, padding_size:width+padding_size] = image
                
                # Extend edges
                # Top edge
                padded[:padding_size, padding_size:width+padding_size] = image[0:1, :]
                # Bottom edge
                padded[height+padding_size:, padding_size:width+padding_size] = image[-1:, :]
                # Left edge
                padded[padding_size:height+padding_size, :padding_size] = image[:, 0:1]
                # Right edge
                padded[padding_size:height+padding_size, width+padding_size:] = image[:, -1:]
                
                # Corners
                padded[:padding_size, :padding_size] = image[0, 0]
                padded[:padding_size, width+padding_size:] = image[0, -1]
                padded[height+padding_size:, :padding_size] = image[-1, 0]
                padded[height+padding_size:, width+padding_size:] = image[-1, -1]
                
            else:
                padded = np.zeros((height + padding_size * 2, width + padding_size * 2), dtype=np.uint8)
                padded[padding_size:height+padding_size, padding_size:width+padding_size] = image
                
                # Extend edges for grayscale
                padded[:padding_size, padding_size:width+padding_size] = image[0:1, :]
                padded[height+padding_size:, padding_size:width+padding_size] = image[-1:, :]
                padded[padding_size:height+padding_size, :padding_size] = image[:, 0:1]
                padded[padding_size:height+padding_size, width+padding_size:] = image[:, -1:]
                
                # Corners
                padded[:padding_size, :padding_size] = image[0, 0]
                padded[:padding_size, width+padding_size:] = image[0, -1]
                padded[height+padding_size:, :padding_size] = image[-1, 0]
                padded[height+padding_size:, width+padding_size:] = image[-1, -1]
            
            return padded
            
        except Exception as e:
            print(f"Error creating edge padding: {e}")
            return image