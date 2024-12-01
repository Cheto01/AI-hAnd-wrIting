# style_analysis.py
import numpy as np
from scipy.stats import skew, kurtosis

class HandwritingAnalyzer:
    def __init__(self):
        self.feature_names = [
            'slant_angle', 'baseline_angle', 'size_variation',
            'spacing_consistency', 'pressure_variation', 'connectedness'
        ]
    
    def analyze_style(self, image, skeleton):
        """Analyze handwriting style characteristics"""
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Calculate slant angle
        gradient_y, gradient_x = np.gradient(skeleton)
        slant_angle = np.arctan2(gradient_y, gradient_x).mean()
        
        # Analyze baseline
        rows = np.sum(skeleton, axis=1)
        baseline_angle = np.polyfit(np.arange(len(rows)), rows, 1)[0]
        
        # Size variation
        connected_components = self._get_connected_components(skeleton)
        sizes = [comp.shape[0] * comp.shape[1] for comp in connected_components]
        size_variation = np.std(sizes) if sizes else 0
        
        # Spacing analysis
        spaces = self._analyze_spacing(skeleton)
        spacing_consistency = np.std(spaces) if spaces else 0
        
        # Pressure variation (using intensity values)
        pressure_variation = np.std(image)
        
        # Connectedness
        connectedness = len(connected_components) / np.sum(skeleton)
        
        return {
            'slant_angle': slant_angle,
            'baseline_angle': baseline_angle,
            'size_variation': size_variation,
            'spacing_consistency': spacing_consistency,
            'pressure_variation': pressure_variation,
            'connectedness': connectedness
        }
    
    def _get_connected_components(self, image):
        """Extract connected components from binary image"""
        labeled, num_components = cv2.connectedComponents(image.astype(np.uint8))
        return [image[labeled == i] for i in range(1, num_components)]
    
    def _analyze_spacing(self, skeleton):
        """Analyze character and word spacing"""
        col_proj = np.sum(skeleton, axis=0)
        spaces = []
        space_start = None
        
        for i, proj in enumerate(col_proj):
            if proj == 0 and space_start is None:
                space_start = i
            elif proj > 0 and space_start is not None:
                spaces.append(i - space_start)
                space_start = None
                
        return spaces