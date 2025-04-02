# src/deeprl/models/base_model.py
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path, device):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
            device (torch.device): Device to load the model on
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
