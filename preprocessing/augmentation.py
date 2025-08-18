"""Data augmentation for sensor signals."""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DataAugmentation:
    """Augmentation techniques for IMU sensor data."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize data augmentation.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
    
    def add_noise(self, data: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to signal.
        
        Args:
            data: Input signal
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy signal
        """
        noise = self.rng.normal(0, noise_std, data.shape)
        return data + noise
    
    def scale(self, data: np.ndarray,
              scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Random scaling of signal amplitude.
        
        Args:
            data: Input signal
            scale_range: Range of scaling factors
            
        Returns:
            Scaled signal
        """
        scale_factor = self.rng.uniform(scale_range[0], scale_range[1])
        return data * scale_factor
    
    def time_warp(self, data: np.ndarray,
                  warp_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Apply time warping to signal.
        
        Args:
            data: Input signal (n_samples, n_channels)
            warp_range: Range of warping factors
            
        Returns:
            Time-warped signal
        """
        n_samples = data.shape[0]
        warp_factor = self.rng.uniform(warp_range[0], warp_range[1])
        
        # Create warped time indices
        original_indices = np.arange(n_samples)
        warped_length = int(n_samples * warp_factor)
        warped_indices = np.linspace(0, n_samples - 1, warped_length)
        
        # Interpolate for each channel
        warped_data = np.zeros((warped_length, data.shape[1]))
        for i in range(data.shape[1]):
            warped_data[:, i] = np.interp(warped_indices, original_indices, data[:, i])
        
        # Resample back to original length
        final_indices = np.linspace(0, warped_length - 1, n_samples)
        resampled_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            resampled_data[:, i] = np.interp(final_indices,
                                            np.arange(warped_length),
                                            warped_data[:, i])
        
        return resampled_data
    
    def rotation(self, data: np.ndarray,
                 max_angle: float = 10.0) -> np.ndarray:
        """
        Apply random rotation to 3D sensor data.
        
        Args:
            data: Input signal (n_samples, 3) or (n_samples, 6) for acc+gyro
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Rotated signal
        """
        if data.shape[1] not in [3, 6]:
            logger.warning("Rotation only applies to 3D or 6D data")
            return data
        
        # Generate random rotation angles
        angles = self.rng.uniform(-max_angle, max_angle, 3) * np.pi / 180
        
        # Create rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(angles[0]), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        
        # Apply rotation
        if data.shape[1] == 3:
            return data @ R.T
        else:
            # Apply to both accelerometer and gyroscope
            rotated = np.zeros_like(data)
            rotated[:, :3] = data[:, :3] @ R.T
            rotated[:, 3:6] = data[:, 3:6] @ R.T
            return rotated
    
    def permutation(self, data: np.ndarray,
                   n_segments: int = 4) -> np.ndarray:
        """
        Random permutation of signal segments.
        
        Args:
            data: Input signal
            n_segments: Number of segments to split into
            
        Returns:
            Permuted signal
        """
        n_samples = data.shape[0]
        segment_length = n_samples // n_segments
        
        # Create segments
        segments = []
        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length if i < n_segments - 1 else n_samples
            segments.append(data[start:end])
        
        # Randomly permute segments
        permuted_indices = self.rng.permutation(n_segments)
        permuted_data = np.vstack([segments[i] for i in permuted_indices])
        
        return permuted_data