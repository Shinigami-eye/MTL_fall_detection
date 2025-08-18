"""Custom samplers for balanced batch creation."""

import logging
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    """Sampler that ensures balanced batches with fall samples."""
    
    def __init__(self, labels: np.ndarray, batch_size: int,
                 fall_ratio: float = 0.15, seed: int = 42):
        """
        Initialize balanced batch sampler.
        
        Args:
            labels: Array of fall labels (0/1)
            batch_size: Batch size
            fall_ratio: Minimum ratio of fall samples per batch
            seed: Random seed
        """
        self.labels = labels
        self.batch_size = batch_size
        self.fall_ratio = fall_ratio
        self.seed = seed
        
        # Separate fall and non-fall indices
        self.fall_indices = np.where(labels == 1)[0]
        self.non_fall_indices = np.where(labels == 0)[0]
        
        # Calculate samples per batch
        self.n_fall_per_batch = max(1, int(batch_size * fall_ratio))
        self.n_non_fall_per_batch = batch_size - self.n_fall_per_batch
        
        logger.info(
            f"Balanced sampler: {len(self.fall_indices)} fall, "
            f"{len(self.non_fall_indices)} non-fall samples. "
            f"Batch composition: {self.n_fall_per_batch} fall, "
            f"{self.n_non_fall_per_batch} non-fall"
        )
    
    def __iter__(self):
        """Generate indices for balanced batches."""
        rng = np.random.RandomState(self.seed)
        
        # Shuffle indices
        fall_indices = self.fall_indices.copy()
        non_fall_indices = self.non_fall_indices.copy()
        rng.shuffle(fall_indices)
        rng.shuffle(non_fall_indices)
        
        # Create batches
        n_batches = len(self) // self.batch_size
        
        for batch_idx in range(n_batches):
            batch_indices = []
            
            # Add fall samples (with replacement if needed)
            fall_batch_indices = np.random.choice(
                fall_indices,
                size=self.n_fall_per_batch,
                replace=len(fall_indices) < self.n_fall_per_batch
            )
            batch_indices.extend(fall_batch_indices)
            
            # Add non-fall samples
            start_idx = batch_idx * self.n_non_fall_per_batch
            end_idx = start_idx + self.n_non_fall_per_batch
            
            if end_idx > len(non_fall_indices):
                # Wrap around
                remaining = end_idx - len(non_fall_indices)
                batch_indices.extend(non_fall_indices[start_idx:])
                batch_indices.extend(non_fall_indices[:remaining])
            else:
                batch_indices.extend(non_fall_indices[start_idx:end_idx])
            
            # Shuffle batch
            rng.shuffle(batch_indices)
            
            for idx in batch_indices:
                yield int(idx)
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.labels)
