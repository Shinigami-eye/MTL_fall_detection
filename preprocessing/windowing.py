"""Windowing utilities for time series data."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WindowGenerator:
    """Generate fixed-size windows from time series data."""
    
    def __init__(self, window_size: int, stride: int,
                 sampling_rate: float = 50.0):
        """
        Initialize window generator.
        
        Args:
            window_size: Window size in samples
            stride: Stride between windows in samples
            sampling_rate: Sampling rate in Hz
        """
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        
        # Calculate overlap
        self.overlap = window_size - stride
        self.overlap_ratio = self.overlap / window_size if window_size > 0 else 0
        
        logger.info(
            f"Window generator initialized: size={window_size} samples "
            f"({window_size/sampling_rate:.2f}s), stride={stride} samples, "
            f"overlap={self.overlap_ratio:.1%}"
        )
    
    def generate_windows(self, data: np.ndarray,
                        labels: Optional[np.ndarray] = None,
                        metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Generate windows from continuous data.
        
        Args:
            data: Input data of shape (n_samples, n_channels)
            labels: Optional labels for each sample
            metadata: Optional metadata to attach to each window
            
        Returns:
            List of window dictionaries
        """
        n_samples = data.shape[0]
        
        if n_samples < self.window_size:
            logger.warning(
                f"Data length ({n_samples}) shorter than window size "
                f"({self.window_size}), skipping"
            )
            return []
        
        windows = []
        
        # Calculate number of windows
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            # Extract window data
            window_data = data[start_idx:end_idx]
            
            # Create window dictionary
            window_dict = {
                'data': window_data,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'window_idx': i
            }
            
            # Add metadata if provided
            if metadata:
                window_dict.update(metadata)
            
            # Assign label if provided
            if labels is not None:
                window_labels = labels[start_idx:end_idx]
                # Use majority voting for window label
                window_dict['label'] = self._get_window_label(window_labels)
                window_dict['label_distribution'] = self._get_label_distribution(window_labels)
            
            windows.append(window_dict)
        
        return windows
    
    def _get_window_label(self, labels: np.ndarray) -> int:
        """
        Get single label for window using majority voting.
        
        Args:
            labels: Array of labels within window
            
        Returns:
            Most common label
        """
        unique, counts = np.unique(labels, return_counts=True)
        return int(unique[np.argmax(counts)])
    
    def _get_label_distribution(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Get distribution of labels within window.
        
        Args:
            labels: Array of labels within window
            
        Returns:
            Dictionary mapping label to proportion
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        return {int(label): count/total for label, count in zip(unique, counts)}
    
    def generate_windows_from_trials(self, trials: List[Dict],
                                    add_magnitude: bool = False) -> List[Dict]:
        """
        Generate windows from multiple trials.
        
        Args:
            trials: List of trial dictionaries with 'data' and 'metadata'
            add_magnitude: Whether to add magnitude channel
            
        Returns:
            List of all windows
        """
        all_windows = []
        
        for trial in trials:
            data = trial['data']
            
            # Add magnitude channel if requested
            if add_magnitude and data.shape[1] >= 3:
                magnitude = self._compute_magnitude(data[:, :3])
                data = np.column_stack([data, magnitude])
            
            # Generate windows for this trial
            windows = self.generate_windows(
                data,
                labels=trial.get('labels'),
                metadata=trial.get('metadata')
            )
            
            all_windows.extend(windows)
        
        logger.info(f"Generated {len(all_windows)} windows from {len(trials)} trials")
        return all_windows
    
    def _compute_magnitude(self, data: np.ndarray) -> np.ndarray:
        """
        Compute magnitude of 3D vectors.
        
        Args:
            data: Array of shape (n_samples, 3)
            
        Returns:
            Magnitude array of shape (n_samples,)
        """
        return np.sqrt(np.sum(data ** 2, axis=1))
    
    def validate_window_labels(self, windows: List[Dict],
                               conflict_threshold: float = 0.3) -> Tuple[bool, List[int]]:
        """
        Validate window labels for conflicts.
        
        Args:
            windows: List of window dictionaries
            conflict_threshold: Threshold for label conflict detection
            
        Returns:
            Tuple of (has_conflicts, list of conflicted window indices)
        """
        conflicted_windows = []
        
        for i, window in enumerate(windows):
            if 'label_distribution' not in window:
                continue
            
            # Check if majority label is strong enough
            max_prop = max(window['label_distribution'].values())
            if max_prop < (1 - conflict_threshold):
                conflicted_windows.append(i)
                logger.warning(
                    f"Window {i} has label conflict: {window['label_distribution']}"
                )
        
        return len(conflicted_windows) > 0, conflicted_windows
