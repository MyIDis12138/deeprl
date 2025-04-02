# src/deeprl/layers/cnn_layers.py
import torch
import torch.nn as nn
import numpy as np

class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network feature extractor for image observations.
    """
    def __init__(self, input_shape):
        """
        Initialize the CNN feature extractor.
        
        Args:
            input_shape (tuple): Shape of input (channels, height, width)
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate feature size after convolutions
        self.feature_size = self._get_conv_output(input_shape)
    
    def _get_conv_output(self, shape):
        """
        Calculate the output size of the CNN feature extractor.
        
        Args:
            shape (tuple): Input shape (channels, height, width)
            
        Returns:
            int: Flattened feature size
        """
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        Forward pass through the CNN feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Extracted features
        """
        return self.features(x)

# src/deeprl/layers/__init__.py
from .cnn_layers import CNNFeatureExtractor

__all__ = ['CNNFeatureExtractor']