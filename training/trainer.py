"""Main trainer class for multi-task learning."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..eval.adl_metrics import compute_adl_metrics
from ..eval.fall_metrics import compute_fall_metrics

logger = logging.getLogger(__name__)


class MTLTrainer:
    """Trainer for multi-task learning models."""
    
    def __init__(self, model: nn.Module, config: DictConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: MTL model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Create loss function
        self.criterion = self._create_loss()
        
        # Create optimizer
        from .optimizers import create_optimizer
        self.optimizer = create_optimizer(model, config.training.optimizer)
        
        # Create scheduler
        from .schedulers import create_scheduler
        self.scheduler = create_scheduler(
            self.optimizer, config.training.scheduler,
            len(train_loader)
        )
        
        # Mixed precision training
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping
        self.gradient_clip = config.training.gradient_clip
        
        # Checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.best_metric = -float('inf')
        self.epoch = 0
        
        # Tensorboard
        if config.training.use_tensorboard:
            self.writer = SummaryWriter(
                log_dir=self.checkpoint_dir / 'tensorboard'
            )
        else:
            self.writer = None
    
    def _create_loss(self):
        """Create loss function based on configuration."""
        from .losses import MTLLoss, UncertaintyWeightedLoss, GradNormLoss
        
        loss_type = self.config.training.loss_type
        
        # Calculate class weights for fall detection if needed
        class_weights = None
        if self.config.training.use_class_weights:
            # Calculate from training data
            fall_labels = []
            for batch in self.train_loader:
                fall_labels.extend(batch['fall_label'].numpy())
            
            fall_ratio = np.mean(fall_labels)
            class_weights = torch.tensor([(1 - fall_ratio) / fall_ratio])
            logger.info(f"Fall class weight: {class_weights.item():.3f}")
        
        if loss_type == "weighted" or loss_type == "single_task":
            return MTLLoss(
                task_weights=self.config.training.task_weights,
                fall_loss_type=self.config.training.fall_loss_type,
                focal_gamma=self.config.training.focal_gamma,
                class_weights=class_weights
            )
        
        elif loss_type == "uncertainty":
            return UncertaintyWeightedLoss(
                init_sigma=self.config.training.uncertainty.init_sigma,
                learnable=self.config.training.uncertainty.learnable,
                fall_loss_type=self.config.training.fall_loss_type,
                focal_gamma=self.config.training.focal_gamma,
                class_weights=class_weights
            )
        
        elif loss_type == "gradnorm":
            return GradNormLoss(
                model=self.model,
                alpha=self.config.training.gradnorm.alpha,
                fall_loss_type=self.config.training.fall_loss_type,
                focal_gamma=self.config.training.focal_gamma,
                class_weights=class_weights
            )
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0
        activity_loss = 0
        fall_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['data'].to(self.device)
            targets = {
                'activity': batch['activity_label'].to(self.device),
                'fall': batch['fall_label'].to(self.device)
            }
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                predictions = self.model(inputs)
                losses = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(losses['total']).backward()
                
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update GradNorm if needed
            if hasattr(self.criterion, 'update_gradnorm_weights'):
                if batch_idx % self.config.training.gradnorm.update_freq == 0:
                    self.criterion.update_gradnorm_weights(self.optimizer)
            
            # Track losses
            total_loss += losses['total'].item()
            activity_loss += losses['activity'].item()
            fall_loss += losses['fall'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'act': f"{losses['activity'].item():.4f}",
                'fall': f"{losses['fall'].item():.4f}"
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config.training.log_interval == 0:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/total_loss', losses['total'].item(), global_step)
                self.writer.add_scalar('train/activity_loss', losses['activity'].item(), global_step)
                self.writer.add_scalar('train/fall_loss', losses['fall'].item(), global_step)
                
                if 'sigma_activity' in losses:
                    self.writer.add_scalar('train/sigma_activity', losses['sigma_activity'].item(), global_step)
                    self.writer.add_scalar('train/sigma_fall', losses['sigma_fall'].item(), global_step)
        
        return {
            'total_loss': total_loss / num_batches,
            'activity_loss': activity_loss / num_batches,
            'fall_loss': fall_loss / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        all_predictions = {'activity': [], 'fall': []}
        all_targets = {'activity': [], 'fall': []}
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                inputs = batch['data'].to(self.device)
                targets = {
                    'activity': batch['activity_label'].to(self.device),
                    'fall': batch['fall_label'].to(self.device)
                }
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    predictions = self.model(inputs)
                    losses = self.criterion(predictions, targets)
                
                # Track predictions
                all_predictions['activity'].append(
                    torch.argmax(predictions['activity'], dim=1).cpu()
                )
                all_predictions['fall'].append(
                    torch.sigmoid(predictions['fall']).cpu()
                )
                all_targets['activity'].append(targets['activity'].cpu())
                all_targets['fall'].append(targets['fall'].cpu())
                
                total_loss += losses['total'].item()
                num_batches += 1
        
        # Concatenate predictions
        all_predictions['activity'] = torch.cat(all_predictions['activity'])
        all_predictions['fall'] = torch.cat(all_predictions['fall'])
        all_targets['activity'] = torch.cat(all_targets['activity'])
        all_targets['fall'] = torch.cat(all_targets['fall'])
        
        # Compute metrics
        adl_metrics = compute_adl_metrics(
            all_predictions['activity'].numpy(),
            all_targets['activity'].numpy()
        )
        
        fall_metrics = compute_fall_metrics(
            all_predictions['fall'].numpy(),
            all_targets['fall'].numpy()
        )
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_adl_accuracy': adl_metrics['accuracy'],
            'val_adl_macro_f1': adl_metrics['macro_f1'],
            'val_fall_precision': fall_metrics['precision'],
            'val_fall_recall': fall_metrics['recall'],
            'val_fall_f1': fall_metrics['f1'],
            'val_fall_prauc': fall_metrics['pr_auc']
        }
        
        return metrics
    
    def train(self, num_epochs: int):
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config.training.val_interval == 0:
                val_metrics = self.validate()
                
                # Compute composite metric
                composite_metric = (
                    self.config.training.metric_weights['val_fall_prauc'] * val_metrics['val_fall_prauc'] +
                    self.config.training.metric_weights['val_adl_macro_f1'] * val_metrics['val_adl_macro_f1']
                )
                
                # Log metrics
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['total_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"val_adl_f1={val_metrics['val_adl_macro_f1']:.4f}, "
                    f"val_fall_prauc={val_metrics['val_fall_prauc']:.4f}"
                )
                
                # Save best model
                if composite_metric > self.best_metric:
                    self.best_metric = composite_metric
                    self.save_checkpoint('best_model.pt', val_metrics)
                    logger.info(f"New best model! Composite metric: {composite_metric:.4f}")
                
                # Log to tensorboard
                if self.writer:
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Save last model
            if self.config.training.save_last:
                self.save_checkpoint('last_model.pt', train_metrics)
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            metrics: Metrics to save with checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        logger.info(f"Saved checkpoint: {filename}")
