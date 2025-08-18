"""Task-specific heads for multi-task learning."""

from typing import List, Optional

import torch
import torch.nn as nn


class ActivityHead(nn.Module):
    """Classification head for activity recognition."""
    
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: List[int], dropout: float = 0.3):
        """
        Initialize activity classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of activity classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.classifier(x)


class FallHead(nn.Module):
    """Binary classification head for fall detection."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 dropout: float = 0.3):
        """
        Initialize fall detection head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, 1)
        """
        return self.classifier(x)
