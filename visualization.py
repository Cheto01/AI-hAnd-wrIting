# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px

class HandwritingVisualizer:
    def __init__(self):
        self.feature_colors = {
            'slant_angle': '#FF9999',
            'baseline_angle': '#66B2FF',
            'size_variation': '#99FF99',
            'spacing_consistency': '#FFCC99',
            'pressure_variation': '#FF99FF',
            'connectedness': '#99FFFF'
        }
    
    def plot_style_radar(self, style_features):
        """Create a radar chart of style features"""
        fig = go.Figure()
        
        # Prepare the data
        categories = list(style_features.keys())
        values = list(style_features.values())
        values += values[:1]  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color='rgb(67, 67, 67)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Handwriting Style Analysis"
        )
        
        return fig
    
    def visualize_preprocessing(self, original, processed, skeleton):
        """Show preprocessing steps side by side"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        
        axes[1].imshow(processed, cmap='gray')
        axes[1].set_title('Processed')
        
        axes[2].imshow(skeleton, cmap='gray')
        axes[2].set_title('Skeleton')
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_distributions(self, feature_history):
        """Plot historical distribution of style features"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (feature, values) in enumerate(feature_history.items()):
            sns.histplot(values, ax=axes[idx], color=self.feature_colors[feature])
            axes[idx].set_title(feature.replace('_', ' ').title())
        
        plt.tight_layout()
        return fig
    
    def visualize_augmentations(self, original, augmented_samples):
        """Display original and augmented versions"""
        n_samples = len(augmented_samples)
        fig, axes = plt.subplots(1, n_samples + 1, figsize=(3 * (n_samples + 1), 3))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        for idx, aug in enumerate(augmented_samples):
            axes[idx + 1].imshow(aug, cmap='gray')
            axes[idx + 1].set_title(f'Augmentation {idx + 1}')
            axes[idx + 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_style_transfer(self, original, transferred, target_style):
        """Visualize style transfer results"""
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original')
        ax1.axis('off')
        
        # Style transfer result
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(transferred, cmap='gray')
        ax2.set_title('Style Transferred')
        ax2.axis('off')
        
        # Style parameters
        ax3 = fig.add_subplot(gs[2])
        style_items = list(target_style.items())
        y_pos = np.arange(len(style_items))
        
        bars = ax3.barh(y_pos, [v for _, v in style_items])
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([k.replace('_', ' ').title() for k, _ in style_items])
        
        # Color bars according to feature colors
        for bar, (feature, _) in zip(bars, style_items):
            bar.set_color(self.feature_colors[feature])
        
        ax3.set_title('Target Style Parameters')
        
        plt.tight_layout()
        return fig