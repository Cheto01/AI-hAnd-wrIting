# augmentation.py
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
import elasticdeform.torch as elastic

class HandwritingAugmentor:
    def __init__(self, probability=0.5):
        self.probability = probability
        
    def random_distortion(self, image):
        """Apply elastic distortion to simulate natural handwriting variation"""
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        
        # Create displacement fields
        displacement = torch.randn(2, 3, 3) * 5
        return elastic.deform_grid(image, displacement, order=3)
    
    def random_slant(self, image, max_angle=15):
        """Randomly adjust writing slant"""
        angle = random.uniform(-max_angle, max_angle)
        return transforms.functional.affine(image, angle, (0, 0), 1.0, 0)
    
    def random_spacing(self, image):
        """Adjust character and word spacing"""
        width = image.size[0] if isinstance(image, Image.Image) else image.shape[-1]
        stretch_factor = random.uniform(0.9, 1.1)
        new_width = int(width * stretch_factor)
        return transforms.Resize((image.size[1], new_width))(image)
    
    def random_pressure(self, image):
        """Simulate varying writing pressure"""
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.8, 1.2)
        return enhancer.enhance(factor)
    
    def random_noise(self, image):
        """Add random noise to simulate paper texture"""
        noise = torch.randn_like(image) * 0.05
        return torch.clamp(image + noise, 0, 1)
    
    def __call__(self, image):
        """Apply random augmentations with given probability"""
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        augmentations = [
            (self.random_distortion, True),  # Always apply some distortion
            (self.random_slant, random.random() < self.probability),
            (self.random_spacing, random.random() < self.probability),
            (self.random_pressure, random.random() < self.probability),
            (self.random_noise, random.random() < self.probability)
        ]
        
        for aug_func, apply in augmentations:
            if apply:
                image = aug_func(image)
        
        return transforms.ToTensor()(image)