"""Optimizer creation utilities."""

from typing import Any, Dict

import torch
from omegaconf import DictConfig


def create_optimizer(model: torch.nn.Module, config: DictConfig) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    opt_type = config.type.lower()
    
    if opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.get('betas', [0.9, 0.999])
        )
    
    elif opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.get('betas', [0.9, 0.999])
        )
    
    elif opt_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=config.weight_decay,
            nesterov=config.get('nesterov', True)
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
