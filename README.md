# Cat & Dog Classification

## Project Overview
This project implements a binary image classifier using ResNet architecture to distinguish between cats and dogs. The focus is on demonstrating deep learning fundamentals and architectural decisions rather than achieving state-of-the-art performance.

## 1. Setup & Environment
### 1.1. Conda Environment
```bash
conda create --name CDC python=3.10
conda activate CDC
```

### 1.2. Dependencies
```bash
pip install -r requirements.txt
```

### 1.3. Dataset Structure
Download the dataset from [Kaggle Animal Dataset](https://www.kaggle.com/datasets/arifmia/animal?resource=download)

The dataset should be organized as follows:
```
dataset/
    train/
        cat/
            cat.0.jpg
            cat.1.jpg
            ...
        dog/
            dog.0.jpg
            dog.1.jpg
            ...
    test/
        cat/
            cat.0.jpg
            cat.1.jpg
            ...
        dog/
            dog.0.jpg
            dog.1.jpg
            ...
```
### 1.4. Weights & Biases Integration
in the main.py file, you can find the following code on line 302:
```python
    wandb.login(key="use your key")
```
change the key to your own key to use Weights & Biases for experiment tracking.

## 2. Model Architecture & Design Decisions

### 2.1. ResNet Architecture Choice
We chose ResNet18 as our base architecture for several key reasons:
- **Skip Connections**: ResNet's skip connections help combat the vanishing gradient problem in deep networks by providing direct paths for gradient flow
- **Proven Architecture**: ResNet has shown strong performance on various computer vision tasks while maintaining reasonable computational requirements
- **Feature Hierarchy**: The progressive downsampling in ResNet helps build a robust hierarchy of features, from simple edges to complex object parts

### 2.2. Key Architectural Modifications

#### Input Channel Adaptation
```python
self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, ...)
```
- Modified to accept grayscale input (1 channel) instead of RGB (3 channels)
- Reduces model complexity while preserving important structural features
- Helps prevent overfitting on color information that might not be crucial for cat/dog classification

#### Adaptive Pooling Implementation
```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
```
- Replaces fixed-size pooling with adaptive pooling
- Enables handling of variable input sizes without architecture changes
- Maintains spatial relationship information while reducing feature dimensions

### 2.3. Data Preprocessing & Augmentation Strategy
Our preprocessing pipeline includes carefully chosen augmentations:
- **Grayscale Conversion**: Reduces input complexity while retaining structural information
- **Random Horizontal Flips**: Accounts for different animal orientations
- **Slight Rotations (±10°)**: Makes the model robust to tilted images
- **Color Jittering**: Helps handle varying lighting conditions and image quality
- **Normalization**: Standardizes input distribution to help with training stability

## 3. Training Methodology

### 3.1. Loss Function & Optimization
- **CrossEntropyLoss**: Chosen for numerical stability in binary classification
- **Adam Optimizer**: Adapts learning rates per parameter, crucial for deep architectures
- **ReduceLROnPlateau**: Adjusts learning rate based on validation performance
  - Helps escape local minima
  - Enables fine-tuning in later training stages

### 3.2. Training Monitoring
Training progress can be monitored through Weights & Biases:
[Training Dashboard](https://api.wandb.ai/links/brtlab/nsfkm6do)

Key metrics tracked:
- Training/Validation Loss: Monitors overall model convergence
- Accuracy: Tracks classification performance
- Learning Rate: Shows optimizer's adaptation

## 4. Model Testing & Inference
Use predict.ipynb for model inference. The notebook demonstrates:
- Proper image preprocessing
- Model loading and inference
- Confidence score interpretation

## 5. Future Improvements
Potential enhancements for production deployment:
1. Data augmentation expansion (e.g., cutout, mixup)
2. Model distillation for faster inference
3. Quantization for reduced model size
4. Ensemble methods for improved robustness

## 6. Results Interpretation
The model achieves better-than-random performance while demonstrating key deep learning concepts:
- Effective gradient flow through skip connections
- Successful feature hierarchy learning
- Proper regularization through data augmentation

Training logs and visualizations are available in the Weights & Biases dashboard.