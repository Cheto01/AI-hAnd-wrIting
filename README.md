# Neural Handwriting Analysis and Generation System


<img src="/images/inputs.png" alt="inputs sample for style transfer">
<img src="/images/Analysis.png" alt="Analysis sample for style transfer">
<img src="/images/Generate.png" alt="Generating sample for style transfer">


## Overview
This project implements a comprehensive system for analyzing, replicating, and generating handwriting using deep learning. The system can analyze handwriting characteristics, transfer styles between samples, and generate new handwriting with controllable style parameters. 
This is part of a robotic project for a comprehensive handwriting analysis and duplication for a robot writer. More details will be published once the robot is operational.

## Features
- **Advanced Preprocessing Pipeline**
  - Background removal and noise reduction
  - Slant correction and normalization
  - Skeletal feature extraction
  - Automatic stroke analysis

- **Style Analysis**
  - Slant angle detection
  - Baseline consistency measurement
  - Size and spacing variation analysis
  - Pressure and stroke characteristics
  - Connectedness evaluation

- **Data Augmentation**
  - Elastic distortions
  - Random slant adjustments
  - Spacing variations
  - Pressure simulation
  - Natural noise addition

- **Visualization Tools**
  - Style radar charts
  - Feature distribution plots
  - Preprocessing step visualization
  - Augmentation comparisons
  - Training progress tracking

- **Neural Network Architecture**
  - Style-aware generation
  - Content-style separation
  - Custom layers for style injection
  - Controllable style parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/cheto01/AI-hAnd-wrIting.git
cd AI-hAnd-wrIting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## Project Structure
```
AI-hAnd-wrIting/
├── main.py              # Main system implementation
├── preprocessing.py     # Preprocessing pipeline
├── augmentation.py      # Data augmentation methods
├── visualization.py     # Visualization tools
├── style_analysis.py    # Style analysis components
├── model.py            # Neural network architecture
├── requirements.txt     # Project dependencies
├── config/             # Configuration files
├── models/             # Saved model checkpoints
├── logs/               # Training logs
├── output/             # Generated outputs
└── visualizations/     # Saved visualizations
```

## Usage

### Basic Usage
```python
from main import HandwritingSystem

# Initialize the system
system = HandwritingSystem()

# Train the model
system.train(['path/to/sample1.png', 'path/to/sample2.png'])

# Generate new handwriting
system.generate('input.png', output_path='output.png')

# Analyze handwriting
analysis = system.analyze_and_visualize('sample.png')
```

### Configuration
Create a custom configuration file:
```python
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
```

### Training with Custom Style Parameters
```python
# Define custom style parameters
style_params = {
    'slant_angle': 0.5,
    'baseline_angle': 0.3,
    'size_variation': 0.7,
    'spacing_consistency': 0.4,
    'pressure_variation': 0.6,
    'connectedness': 0.5
}

# Generate with custom style
system.generate('input.png', style_params=style_params)
```

## Advanced Features

### Style Transfer
```python
# Analyze source style
source_analysis = system.analyze_and_visualize('source.png')

# Transfer style to new content
system.generate('target.png', style_params=source_analysis['style_features'])
```

### Batch Processing
```python
# Process multiple samples
image_paths = ['sample1.png', 'sample2.png', 'sample3.png']
results = [system.analyze_and_visualize(path) for path in image_paths]
```

### Training Visualization
```python
# View training progress
system.visualizer.plot_training_progress()

# Analyze style distributions
system.visualizer.plot_feature_distributions()
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Papers and research that influenced this work
- Contributors and maintainers
- Open source libraries used in the project

## Contact
Che 
Project Link: https://github.com/cheto01/AI-hAnd-wrIting

## Citation
If you use this project in your research, please cite:
```
@software{handwriting_analysis_2024,
  author = {Cherubin Mugisha},
  title = {Neural Handwriting Analysis and Generation System},
  year = {2024},
  url = {https://github.com/cheto01/AI-hAnd-wrIting}
}
```