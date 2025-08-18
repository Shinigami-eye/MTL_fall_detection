"""Tests for balanced sampling."""

import numpy as np
import pytest

from training.samplers import BalancedBatchSampler


class TestBalancedBatchSampler:
    """Test balanced batch sampling."""
    
    def test_balanced_batches(self):
        """Test that batches contain fall samples."""
        # Create imbalanced labels
        labels = np.zeros(1000)
        labels[::20] = 1  # 5% fall samples
        
        sampler = BalancedBatchSampler(
            labels=labels,
            batch_size=32,
            fall_ratio=0.25,  # Want 25% fall in each batch
            seed=42
        )
        
        # Check that we have fall samples
        assert len(sampler.fall_indices) == 50
        assert len(sampler.non_fall_indices) == 950
        
        # Get first batch
        batch_indices = list(sampler)[:32]
        batch_labels = labels[batch_indices]
        
        # Check fall ratio in batch
        fall_count = np.sum(batch_labels)
        assert fall_count >= 1  # At least one fall sample
    
    def test_sampler_length(self):
        """Test sampler length."""
        labels = np.random.randint(0, 2, 100)
        
        sampler = BalancedBatchSampler(
            labels=labels,
            batch_size=10,
            fall_ratio=0.2
        )
        
        assert len(sampler) == 100