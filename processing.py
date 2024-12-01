# preprocessing.py
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from scipy.ndimage import rotate
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

class PreprocessingPipeline:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
        self.base_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def remove_background(self, image):
        """Enhanced background removal with adaptive thresholding"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        thresh = threshold_otsu(gray)
        binary = gray < thresh
        return Image.fromarray((binary * 255).astype(np.uint8))
    
    def correct_slant(self, image):
        """Correct writing slant using Hough transform"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = [line[0][1] for line in lines]
            median_angle = np.median(angles)
            rotation_angle = np.degrees(median_angle) - 90
            rotated = rotate(image, rotation_angle)
            return Image.fromarray((rotated * 255).astype(np.uint8))
        return Image.fromarray(image)
    
    def extract_features(self, image):
        """Extract basic handwriting features"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Skeletonize for stroke analysis
        skeleton = skeletonize(image > 127)
        
        # Calculate basic metrics
        stroke_width = np.mean(image > 127) / np.mean(skeleton)
        pixel_density = np.mean(image > 127)
        
        return {
            'stroke_width': stroke_width,
            'pixel_density': pixel_density,
            'skeleton': skeleton
        }
    
    def __call__(self, image_path):
        """Full preprocessing pipeline"""
        image = Image.open(image_path)
        image = self.remove_background(image)
        image = self.correct_slant(image)
        features = self.extract_features(image)
        tensor = self.base_transform(image)
        
        return tensor, features