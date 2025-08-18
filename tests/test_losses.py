"""Tests for loss functions."""

import torch
import pytest

from training.losses import MTLLoss, UncertaintyWeightedLoss, FocalLoss


class TestMTLLoss:
    """Test multi-task loss functions."""
    
    def test_mtl_loss_computation(self):
        """Test basic MTL loss computation."""
        loss_fn = MTLLoss(
            task_weights={'activity': 0.5, 'fall': 0.5},
            fall_loss_type='bce'
        )
        
        # Create dummy predictions and targets
        batch_size = 16
        predictions = {
            'activity': torch.randn(batch_size, 13),
            'fall': torch.randn(batch_size, 1)
        }
        
        targets = {
            'activity': torch.randint(0, 13, (batch_size,)),
            'fall': torch.randint(0, 2, (batch_size,))
        }
        
        # Compute loss
        losses = loss_fn(predictions, targets)
        
        assert 'total' in losses
        assert 'activity' in losses
        assert 'fall' in losses
        assert losses['total'].requires_grad
    
    def test_uncertainty_weighted_loss(self):
        """Test uncertainty-weighted loss."""
        loss_fn = UncertaintyWeightedLoss(
            init_sigma=1.0,
            learnable=True
        )
        
        batch_size = 8
        predictions = {
            'activity': torch.randn(batch_size, 13),
            'fall': torch.randn(batch_size, 1)
        }
        
        targets = {
            'activity': torch.randint(0, 13, (batch_size,)),
            'fall': torch.randint(0, 2, (batch_size,))
        }
        
        losses = loss_fn(predictions, targets)
        
        assert 'sigma_activity' in losses
        assert 'sigma_fall' in losses
        assert loss_fn.log_vars.requires_grad
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        loss_fn = FocalLoss(gamma=2.0)
        
        # Create imbalanced targets
        inputs = torch.randn(100)
        targets = torch.zeros(100)
        targets[:5] = 1  # 5% positive
        
        loss = loss_fn(inputs, targets)
        
        assert loss.item() > 0
        assert loss.requires_grad