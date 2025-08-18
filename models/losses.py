"""
Loss functions for multi-task learning including focal loss, 
uncertainty weighting, and GradNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in fall detection
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size,)
            targets: Binary targets (batch_size,)
        """
        # Convert to probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-based multi-task loss weighting
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
    """
    
    def __init__(self, num_tasks: int = 2):
        super().__init__()
        # Initialize log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            losses: Dictionary of task losses
        
        Returns:
            weighted_loss: Combined weighted loss
            weights: Current task weights for logging
        """
        weighted_losses = []
        weights = {}
        
        for i, (task_name, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
            weights[task_name] = precision.item()
        
        total_loss = sum(weighted_losses)
        
        return total_loss, weights


class GradNormLoss:
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing
    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (ICML 2018)
    """
    
    def __init__(self, model: nn.Module, alpha: float = 1.5, num_tasks: int = 2):
        """
        Args:
            model: The multi-task model
            alpha: Asymmetry parameter (>1 for focus on harder tasks)
            num_tasks: Number of tasks
        """
        self.model = model
        self.alpha = alpha
        self.num_tasks = num_tasks
        
        # Initialize task weights
        self.weights = nn.Parameter(torch.ones(num_tasks))
        
        # Track initial losses for relative loss computation
        self.initial_losses = None
    
    def compute_grad_norm(self, losses: Dict[str, torch.Tensor], 
                         shared_params: nn.ParameterList) -> torch.Tensor:
        """
        Compute GradNorm loss for weight adjustment
        
        Args:
            losses: Dictionary of task losses
            shared_params: Shared model parameters to compute gradients for
        """
        if self.initial_losses is None:
            self.initial_losses = {k: v.detach() for k, v in losses.items()}
        
        # Compute weighted losses
        weighted_losses = []
        task_names = list(losses.keys())
        
        for i, task_name in enumerate(task_names):
            weighted_losses.append(self.weights[i] * losses[task_name])
        
        # Compute gradients for each task
        grads = []
        for i, loss in enumerate(weighted_losses):
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            
            # Compute gradient norm for shared parameters
            grad_norm = 0
            for param in shared_params:
                if param.grad is not None:
                    grad_norm += param.grad.norm() ** 2
            grad_norm = grad_norm ** 0.5
            grads.append(grad_norm)
        
        # Compute relative losses
        relative_losses = []
        for task_name in task_names:
            rel_loss = losses[task_name] / (self.initial_losses[task_name] + 1e-8)
            relative_losses.append(rel_loss)
        
        # Compute average relative loss
        avg_rel_loss = sum(relative_losses) / len(relative_losses)
        
        # Compute target gradient norms
        target_grads = []
        for i, rel_loss in enumerate(relative_losses):
            target_grad = (rel_loss / avg_rel_loss) ** self.alpha
            target_grad *= grads[0].detach()  # Use first task's gradient as reference
            target_grads.append(target_grad)
        
        # Compute GradNorm loss
        grad_norm_loss = 0
        for i in range(len(grads)):
            grad_norm_loss += torch.abs(grads[i] - target_grads[i])
        
        # Normalize weights
        with torch.no_grad():
            self.weights.data = self.weights.data / self.weights.data.sum() * self.num_tasks
        
        return grad_norm_loss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with configurable weighting strategy
    """
    
    def __init__(
        self,
        num_activities: int = 16,
        weighting_strategy: str = 'static',  # 'static', 'uncertainty', 'gradnorm'
        activity_weight: float = 0.3,
        fall_weight: float = 0.7,
        focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.weighting_strategy = weighting_strategy
        self.activity_weight = activity_weight
        self.fall_weight = fall_weight
        
        # Activity loss (cross-entropy)
        self.activity_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # Fall detection loss
        if focal_loss:
            self.fall_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            pos_weight = torch.tensor([10.0])  # Weight for positive class (falls are rare)
            self.fall_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Uncertainty weighting if selected
        if weighting_strategy == 'uncertainty':
            self.uncertainty_weights = UncertaintyWeightedLoss(num_tasks=2)
        
        self.current_weights = {'activity': activity_weight, 'fall': fall_weight}
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model predictions with 'activity_logits' and 'fall_logits'
            targets: Ground truth with 'activity_label' and 'fall_label'
        
        Returns:
            Dictionary with 'total_loss', individual losses, and weights
        """
        # Compute individual losses
        activity_loss = self.activity_loss_fn(
            predictions['activity_logits'],
            targets['activity_label']
        )
        
        fall_loss = self.fall_loss_fn(
            predictions['fall_logits'],
            targets['fall_label']
        )
        
        losses = {
            'activity_loss': activity_loss,
            'fall_loss': fall_loss
        }
        
        # Apply weighting strategy
        if self.weighting_strategy == 'static':
            total_loss = (self.activity_weight * activity_loss + 
                         self.fall_weight * fall_loss)
            weights = self.current_weights
        
        elif self.weighting_strategy == 'uncertainty':
            total_loss, weights = self.uncertainty_weights(losses)
            self.current_weights = weights
        
        else:  # Default to static
            total_loss = (self.activity_weight * activity_loss + 
                         self.fall_weight * fall_loss)
            weights = self.current_weights
        
        return {
            'total_loss': total_loss,
            'activity_loss': activity_loss.detach(),
            'fall_loss': fall_loss.detach(),
            'weights': weights
        }


def compute_gradient_conflict(model: nn.Module, 
                             loss1: torch.Tensor, 
                             loss2: torch.Tensor) -> float:
    """
    Compute cosine similarity between gradients of two losses
    to detect negative transfer
    
    Args:
        model: The model
        loss1: First task loss
        loss2: Second task loss
    
    Returns:
        Cosine similarity between gradients (-1 to 1, negative indicates conflict)
    """
    # Get gradients for first loss
    model.zero_grad()
    loss1.backward(retain_graph=True)
    grads1 = []
    for param in model.parameters():
        if param.grad is not None:
            grads1.append(param.grad.clone().flatten())
    grads1 = torch.cat(grads1)
    
    # Get gradients for second loss
    model.zero_grad()
    loss2.backward(retain_graph=True)
    grads2 = []
    for param in model.parameters():
        if param.grad is not None:
            grads2.append(param.grad.clone().flatten())
    grads2 = torch.cat(grads2)
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(grads1.unsqueeze(0), grads2.unsqueeze(0))
    
    return cos_sim.item()
