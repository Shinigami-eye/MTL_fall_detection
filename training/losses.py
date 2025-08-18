"""Loss functions for multi-task learning."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTLLoss(nn.Module):
    """Base multi-task loss with fixed weights."""
    
    def __init__(self, task_weights: Dict[str, float],
                 fall_loss_type: str = "bce",
                 focal_gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize MTL loss.
        
        Args:
            task_weights: Fixed weights for each task
            fall_loss_type: Type of loss for fall detection ("bce" or "focal")
            focal_gamma: Gamma parameter for focal loss
            class_weights: Optional class weights for fall detection
        """
        super().__init__()
        self.task_weights = task_weights
        self.fall_loss_type = fall_loss_type
        self.focal_gamma = focal_gamma
        
        # Activity loss
        self.activity_loss = nn.CrossEntropyLoss()
        
        # Fall loss
        if fall_loss_type == "focal":
            self.fall_loss = FocalLoss(gamma=focal_gamma, pos_weight=class_weights)
        else:
            self.fall_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary with individual and total losses
        """
        # Activity loss
        activity_loss = self.activity_loss(predictions['activity'], targets['activity'])
        
        # Fall loss
        fall_loss = self.fall_loss(predictions['fall'].squeeze(), targets['fall'].float())
        
        # Total weighted loss
        total_loss = (self.task_weights['activity'] * activity_loss +
                     self.task_weights['fall'] * fall_loss)
        
        return {
            'total': total_loss,
            'activity': activity_loss,
            'fall': fall_loss
        }


class UncertaintyWeightedLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weights."""
    
    def __init__(self, init_sigma: float = 1.0, learnable: bool = True,
                 fall_loss_type: str = "bce", focal_gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize uncertainty-weighted loss.
        
        Args:
            init_sigma: Initial sigma values
            learnable: Whether sigmas are learnable
            fall_loss_type: Type of loss for fall detection
            focal_gamma: Gamma for focal loss
            class_weights: Optional class weights
        """
        super().__init__()
        
        # Task uncertainties (log variance)
        self.log_vars = nn.Parameter(
            torch.ones(2) * init_sigma,
            requires_grad=learnable
        )
        
        # Task losses
        self.activity_loss = nn.CrossEntropyLoss()
        
        if fall_loss_type == "focal":
            self.fall_loss = FocalLoss(gamma=focal_gamma, pos_weight=class_weights)
        else:
            self.fall_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-weighted loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary with losses and learned weights
        """
        # Individual losses
        activity_loss = self.activity_loss(predictions['activity'], targets['activity'])
        fall_loss = self.fall_loss(predictions['fall'].squeeze(), targets['fall'].float())
        
        # Uncertainty weighting
        precision_activity = torch.exp(-self.log_vars[0])
        precision_fall = torch.exp(-self.log_vars[1])
        
        weighted_activity = precision_activity * activity_loss + self.log_vars[0]
        weighted_fall = precision_fall * fall_loss + self.log_vars[1]
        
        total_loss = weighted_activity + weighted_fall
        
        return {
            'total': total_loss,
            'activity': activity_loss,
            'fall': fall_loss,
            'sigma_activity': torch.exp(self.log_vars[0] / 2),
            'sigma_fall': torch.exp(self.log_vars[1] / 2)
        }


class GradNormLoss(nn.Module):
    """GradNorm for balancing task training rates."""
    
    def __init__(self, model: nn.Module, alpha: float = 1.0,
                 fall_loss_type: str = "bce", focal_gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize GradNorm loss.
        
        Args:
            model: MTL model
            alpha: Asymmetry parameter
            fall_loss_type: Type of loss for fall detection
            focal_gamma: Gamma for focal loss
            class_weights: Optional class weights
        """
        super().__init__()
        
        self.model = model
        self.alpha = alpha
        
        # Task weights (learnable)
        self.task_weights = nn.Parameter(torch.ones(2))
        
        # Task losses
        self.activity_loss = nn.CrossEntropyLoss()
        
        if fall_loss_type == "focal":
            self.fall_loss = FocalLoss(gamma=focal_gamma, pos_weight=class_weights)
        else:
            self.fall_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        
        # Initial losses for relative training rate
        self.initial_losses = None
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute GradNorm loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary with losses
        """
        # Individual losses
        activity_loss = self.activity_loss(predictions['activity'], targets['activity'])
        fall_loss = self.fall_loss(predictions['fall'].squeeze(), targets['fall'].float())
        
        losses = torch.stack([activity_loss, fall_loss])
        
        # Initialize if needed
        if self.initial_losses is None:
            self.initial_losses = losses.detach()
        
        # Normalize weights
        weights = F.softmax(self.task_weights, dim=0) * 2
        
        # Weighted loss
        total_loss = torch.sum(weights * losses)
        
        return {
            'total': total_loss,
            'activity': activity_loss,
            'fall': fall_loss,
            'weight_activity': weights[0],
            'weight_fall': weights[1]
        }
    
    def update_gradnorm_weights(self, optimizer: torch.optim.Optimizer):
        """
        Update task weights using GradNorm algorithm.
        
        Args:
            optimizer: Optimizer for the model
        """
        # Get last shared layer
        shared_layer = None
        for name, param in self.model.named_parameters():
            if 'backbone' in name and param.requires_grad:
                shared_layer = param
                break
        
        if shared_layer is None:
            return
        
        # Compute gradients for each task
        grads = []
        for i in range(2):
            optimizer.zero_grad()
            if i == 0:
                loss = self.activity_loss
            else:
                loss = self.fall_loss
            
            loss.backward(retain_graph=True)
            grads.append(shared_layer.grad.clone())
        
        # Compute gradient norms
        grad_norms = torch.stack([g.norm() for g in grads])
        
        # Compute relative training rates
        relative_rates = (self.losses / self.initial_losses).detach()
        
        # Target gradient norms
        mean_norm = grad_norms.mean()
        target_norms = mean_norm * (relative_rates ** self.alpha)
        
        # GradNorm loss
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_norms))
        
        # Update weights
        optimizer.zero_grad()
        gradnorm_loss.backward()
        optimizer.step()


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        """
        Initialize focal loss.
        
        Args:
            gamma: Focusing parameter
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()