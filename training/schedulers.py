"""Learning rate scheduler utilities."""

import math
from typing import Optional

import torch
from omegaconf import DictConfig


def create_scheduler(optimizer: torch.optim.Optimizer, config: DictConfig,
                    steps_per_epoch: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Scheduler configuration
        steps_per_epoch: Number of training steps per epoch
        
    Returns:
        Scheduler instance or None
    """
    scheduler_type = config.type.lower()
    
    if scheduler_type == "none":
        return None
    
    elif scheduler_type == "cosine":
        total_steps = steps_per_epoch * config.get('epochs', 100)
        warmup_steps = steps_per_epoch * config.get('warmup_epochs', 5)
        
        return CosineAnnealingWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class CosineAnnealingWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""
    
    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor
                   for base_lr in self.base_lrs]
