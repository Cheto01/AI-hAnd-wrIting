# main.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import os

class HandwritingDataset(Dataset):
    def __init__(self, image_paths, preprocessor):
        self.image_paths = image_paths
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_tensor, features = self.preprocessor(img_path)
        return img_tensor, features

class HandwritingSystem:
    def __init__(self, config=None):
        """
        Initialize the handwriting analysis and generation system
        
        Args:
            config (dict, optional): Configuration dictionary for system parameters
        """
        self.config = config or self._get_default_config()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize components
        self._init_components()
        self._init_logging()
        
        # Initialize history tracking
        self.training_history = {
            'losses': [],
            'style_metrics': [],
            'epochs_completed': 0
        }
    
    def _get_default_config(self):
        """Get default configuration settings"""
        return {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'save_interval': 10,
            'augmentation_probability': 0.5,
            'model_save_path': 'models/',
            'log_path': 'logs/',
            'output_path': 'output/',
            'visualization_path': 'visualizations/'
        }
    
    def _init_components(self):
        """Initialize all system components"""
        from preprocessing import PreprocessingPipeline
        from augmentation import HandwritingAugmentor
        from visualization import HandwritingVisualizer
        from style_analysis import HandwritingAnalyzer
        from model import StyleAwareHandwritingModel
        
        self.preprocessor = PreprocessingPipeline()
        self.augmentor = HandwritingAugmentor(
            probability=self.config['augmentation_probability']
        )
        self.visualizer = HandwritingVisualizer()
        self.analyzer = HandwritingAnalyzer()
        self.model = StyleAwareHandwritingModel().to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        self.criterion = nn.MSELoss()
    
    def _init_logging(self):
        """Initialize logging configuration"""
        log_path = Path(self.config['log_path'])
        log_path.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=log_path / f'training_{datetime.now():%Y%m%d_%H%M%S}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def prepare_data(self, image_paths):
        """
        Prepare data for training or inference
        
        Args:
            image_paths (list): List of paths to handwriting images
        
        Returns:
            DataLoader: PyTorch DataLoader for the dataset
        """
        dataset = HandwritingDataset(image_paths, self.preprocessor)
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4
        )
    
    def train(self, image_paths, epochs=None, resume=False):
        """
        Train the handwriting model
        
        Args:
            image_paths (list): List of paths to training images
            epochs (int, optional): Number of epochs to train
            resume (bool): Whether to resume from saved state
        """
        epochs = epochs or self.config['epochs']
        dataloader = self.prepare_data(image_paths)
        
        if resume:
            self._load_checkpoint()
        
        start_epoch = self.training_history['epochs_completed']
        
        for epoch in range(start_epoch, start_epoch + epochs):
            epoch_loss = 0
            epoch_style_metrics = []
            
            for batch_idx, (images, features) in enumerate(dataloader):
                # Move data to device
                images = images.to(self.device)
                
                # Apply augmentation
                augmented_images = torch.stack([
                    self.augmentor(img) for img in images
                ]).to(self.device)
                
                # Extract style features
                style_features = [
                    self.analyzer.analyze_style(img.cpu().numpy(), feat['skeleton'])
                    for img, feat in zip(images, features)
                ]
                
                style_tensors = torch.tensor([
                    [features[f] for f in self.analyzer.feature_names]
                    for features in style_features
                ], dtype=torch.float32).to(self.device)
                
                # Training step
                self.optimizer.zero_grad()
                outputs = self.model(augmented_images, style_tensors)
                loss = self.criterion(outputs, images)
                
                loss.backward()
                self.optimizer.step()
                
                # Record metrics
                epoch_loss += loss.item()
                epoch_style_metrics.extend(style_features)
                
                # Log batch progress
                if batch_idx % 10 == 0:
                    logging.info(
                        f'Epoch: {epoch+1}/{epochs} '
                        f'[{batch_idx*len(images)}/{len(dataloader.dataset)} '
                        f'({100.*batch_idx/len(dataloader):.0f}%)] '
                        f'Loss: {loss.item():.6f}'
                    )
            
            # Update training history
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.training_history['losses'].append(avg_epoch_loss)
            self.training_history['style_metrics'].append(epoch_style_metrics)
            self.training_history['epochs_completed'] = epoch + 1
            
            # Create and save visualizations
            if (epoch + 1) % self.config['save_interval'] == 0:
                self._save_checkpoint(epoch)
                self._create_training_visualizations(epoch)
    
    def generate(self, input_path, style_params=None, output_path=None):
        """
        Generate handwriting with specified style
        
        Args:
            input_path (str): Path to input image
            style_params (dict, optional): Style parameters to apply
            output_path (str, optional): Path to save generated image
        
        Returns:
            PIL.Image: Generated handwriting image
        """
        output_path = output_path or os.path.join(
            self.config['output_path'],
            f'generated_{datetime.now():%Y%m%d_%H%M%S}.png'
        )
        
        with torch.no_grad():
            # Preprocess input
            img_tensor, features = self.preprocessor(input_path)
            
            if style_params is None:
                # Use analyzed style if no parameters provided
                style_features = self.analyzer.analyze_style(
                    img_tensor.numpy(), features['skeleton']
                )
                style_params = torch.tensor([
                    style_features[f] for f in self.analyzer.feature_names
                ], dtype=torch.float32)
            
            # Generate image
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            style_params = style_params.to(self.device)
            
            output = self.model(img_tensor, style_params)
            
            # Save and return result
            output_img = self.visualizer.tensor_to_image(output.squeeze().cpu())
            output_img.save(output_path)
            
            return output_img
    
    def analyze_and_visualize(self, image_path, save_path=None):
        """
        Analyze handwriting and create visualizations
        
        Args:
            image_path (str): Path to image for analysis
            save_path (str, optional): Path to save visualizations
        
        Returns:
            dict: Analysis results and visualization figures
        """
        save_path = save_path or self.config['visualization_path']
        Path(save_path).mkdir(exist_ok=True)
        
        # Perform analysis
        img_tensor, features = self.preprocessor(image_path)
        style_features = self.analyzer.analyze_style(
            img_tensor.numpy(), features['skeleton']
        )
        
        # Create visualizations
        visualizations = {
            'radar_chart': self.visualizer.plot_style_radar(style_features),
            'preprocessing': self.visualizer.visualize_preprocessing(
                Path(image_path).name,
                img_tensor.numpy(),
                features['skeleton']
            ),
            'feature_distributions': self.visualizer.plot_feature_distributions(
                self.training_history['style_metrics']
            ),
            'augmentations': self.visualizer.visualize_augmentations(
                img_tensor.numpy(),
                [self.augmentor(img_tensor).numpy() for _ in range(4)]
            )
        }
        
        # Save visualizations
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for name, fig in visualizations.items():
            fig.savefig(
                os.path.join(save_path, f'{name}_{timestamp}.png')
            )
        
        return {
            'style_features': style_features,
            'visualizations': visualizations
        }
    
    def _save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config['model_save_path'],
            f'checkpoint_epoch_{epoch+1}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path=None):
        """Load training checkpoint"""
        if checkpoint_path is None:
            # Find latest checkpoint
            model_path = Path(self.config['model_save_path'])
            checkpoints = list(model_path.glob('checkpoint_epoch_*.pt'))
            if not checkpoints:
                logging.warning("No checkpoint found, starting from scratch")
                return
            checkpoint_path = max(checkpoints, key=os.path.getctime)
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logging.info(f"Resumed from checkpoint: {checkpoint_path}")
    
    def _create_training_visualizations(self, epoch):
        """Create and save training progress visualizations"""
        viz_path = Path(self.config['visualization_path']) / f'epoch_{epoch+1}'
        viz_path.mkdir(parents=True, exist_ok=True)
        
        # Loss plot
        self.visualizer.plot_training_loss(
            self.training_history['losses']
        ).savefig(viz_path / 'loss_history.png')
        
        # Style metrics distribution
        self.visualizer.plot_feature_distributions(
            self.training_history['style_metrics']
        ).savefig(viz_path / 'style_distributions.png')

if __name__ == '__main__':
    # Example usage
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'save_interval': 10,
        'augmentation_probability': 0.5,
        'model_save_path': 'models/',
        'log_path': 'logs/',
        'output_path': 'output/',
        'visualization_path': 'visualizations/'
    }
    
    system = HandwritingSystem(config)
    
    # Train the system
    image_paths = ['path/to/image1.png', 'path/to/image2.png']  # Replace with actual paths to test on your own data
    system.train(image_paths)
    
    # Generate new handwriting
    system.generate('input.png', output_path='output.png')
    
    # Analyze and visualize results
    analysis_results = system.analyze_and_visualize('sample.png')