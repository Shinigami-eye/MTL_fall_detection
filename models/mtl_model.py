"""Multi-task learning model combining backbone and task heads."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MTLModel(nn.Module):
    """Multi-task learning model for activity recognition and fall detection."""
    
    def __init__(self, backbone: nn.Module, activity_head: nn.Module,
                 fall_head: nn.Module):
        """
        Initialize MTL model.
        
        Args:
            backbone: Feature extraction backbone
            activity_head: Activity classification head
            fall_head: Fall detection head
        """
        super().__init__()
        
        self.backbone = backbone
        self.activity_head = activity_head
        self.fall_head = fall_head
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, length)
            
        Returns:
            Dictionary with 'activity' and 'fall' predictions
        """
        # Extract features
        features = self.backbone(x)
        
        # Task-specific predictions
        activity_logits = self.activity_head(features)
        fall_logits = self.fall_head(features)
        
        return {
            'activity': activity_logits,
            'fall': fall_logits,
            'features': features  # Also return features for analysis
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get parameter count for each component.
        
        Returns:
            Dictionary with parameter counts
        """
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'backbone': count_parameters(self.backbone),
            'activity_head': count_parameters(self.activity_head),
            'fall_head': count_parameters(self.fall_head),
            'total': count_parameters(self)
        }
    
    def print_summary(self):
        """Print model summary including parameter counts."""
        param_counts = self.get_parameter_count()
        
        print("=" * 60)
        print("Model Summary")
        print("=" * 60)
        print(f"Backbone parameters: {param_counts['backbone']:,}")
        print(f"Activity head parameters: {param_counts['activity_head']:,}")
        print(f"Fall head parameters: {param_counts['fall_head']:,}")
        print(f"Total parameters: {param_counts['total']:,}")
        print("=" * 60)