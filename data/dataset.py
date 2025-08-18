"""
IMU Dataset loader for multi-task learning (activity recognition + fall detection)
Handles window segmentation, dual labeling, and balanced sampling
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Dict, Optional, List
import scipy.signal
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data preprocessing"""
    window_size: float = 2.56  # seconds
    sampling_rate: int = 100  # Hz
    overlap: float = 0.5  # 50% overlap
    lowpass_cutoff: float = 20.0  # Hz for noise filtering
    
    @property
    def window_samples(self) -> int:
        return int(self.window_size * self.sampling_rate)
    
    @property
    def stride_samples(self) -> int:
        return int(self.window_samples * (1 - self.overlap))


class IMUDataset(Dataset):
    """
    Multi-task IMU dataset for activity recognition and fall detection
    
    Args:
        data_path: Path to preprocessed data file
        split: One of 'train', 'val', 'test'
        config: Data configuration object
        augment: Whether to apply data augmentation
    """
    
    # Activity class mapping
    ACTIVITY_CLASSES = [
        'walking', 'standing', 'sitting', 'lying', 'running',
        'jumping', 'sitting_down', 'standing_up', 'picking_up',
        'bending', 'stairs_up', 'stairs_down', 'transition', 
        'fall_forward', 'fall_backward', 'fall_lateral'
    ]
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        config: Optional[DataConfig] = None,
        augment: bool = False
    ):
        self.data_path = data_path
        self.split = split
        self.config = config or DataConfig()
        self.augment = augment
        
        # Load and preprocess data
        self.windows, self.activity_labels, self.fall_labels = self._load_data()
        
        # Compute class weights for balanced sampling
        self.fall_weights = self._compute_class_weights(self.fall_labels)
        self.activity_weights = self._compute_class_weights(self.activity_labels)
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and segment IMU data into windows
        Returns: (windows, activity_labels, fall_labels)
        """
        # TODO: Implement actual data loading from SisFall/MobiAct datasets
        # For now, generate synthetic data for demonstration
        
        n_samples = 1000 if self.split == 'train' else 200
        n_channels = 6  # 3-axis acc + 3-axis gyro
        
        # Generate synthetic IMU windows
        np.random.seed(42)
        windows = np.random.randn(n_samples, n_channels, self.config.window_samples).astype(np.float32)
        
        # Generate activity labels (multi-class)
        activity_labels = np.random.randint(0, len(self.ACTIVITY_CLASSES), n_samples)
        
        # Generate fall labels (binary) - make falls rare
        fall_probs = np.where(activity_labels >= 13, 0.9, 0.05)  # Higher prob for fall activities
        fall_labels = (np.random.random(n_samples) < fall_probs).astype(np.int64)
        
        # Apply preprocessing
        windows = self._preprocess_windows(windows)
        
        return windows, activity_labels, fall_labels
    
    def _preprocess_windows(self, windows: np.ndarray) -> np.ndarray:
        """Apply filtering and normalization to windows"""
        # Design Butterworth lowpass filter
        sos = scipy.signal.butter(
            4, self.config.lowpass_cutoff,
            btype='low', fs=self.config.sampling_rate,
            output='sos'
        )
        
        # Apply filter to each channel
        filtered = np.zeros_like(windows)
        for i in range(windows.shape[0]):
            for ch in range(windows.shape[1]):
                filtered[i, ch] = scipy.signal.sosfiltfilt(sos, windows[i, ch])
        
        # Add magnitude channel (often useful for fall detection)
        acc_magnitude = np.sqrt(np.sum(filtered[:, :3, :]**2, axis=1, keepdims=True))
        gyro_magnitude = np.sqrt(np.sum(filtered[:, 3:6, :]**2, axis=1, keepdims=True))
        
        # Concatenate original channels with magnitudes
        windows_with_mag = np.concatenate([filtered, acc_magnitude, gyro_magnitude], axis=1)
        
        # Z-score normalization per channel
        mean = windows_with_mag.mean(axis=(0, 2), keepdims=True)
        std = windows_with_mag.std(axis=(0, 2), keepdims=True) + 1e-6
        normalized = (windows_with_mag - mean) / std
        
        return normalized
    
    def _compute_class_weights(self, labels: np.ndarray) -> np.ndarray:
        """Compute inverse frequency weights for balanced sampling"""
        unique, counts = np.unique(labels, return_counts=True)
        weights = len(labels) / (len(unique) * counts)
        weight_map = dict(zip(unique, weights))
        return np.array([weight_map[label] for label in labels])
    
    def _augment(self, window: np.ndarray) -> np.ndarray:
        """Apply data augmentation to a window"""
        if not self.augment or self.split != 'train':
            return window
        
        # Time warping (speed up/slow down)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            new_length = int(window.shape[1] * scale)
            augmented = np.zeros((window.shape[0], self.config.window_samples))
            for ch in range(window.shape[0]):
                resampled = np.interp(
                    np.linspace(0, new_length-1, self.config.window_samples),
                    np.arange(new_length),
                    np.interp(np.linspace(0, window.shape[1]-1, new_length),
                             np.arange(window.shape[1]), window[ch])
                )
                augmented[ch] = resampled
            window = augmented
        
        # Rotation augmentation (simulate sensor orientation changes)
        if np.random.random() < 0.3:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            # Apply rotation to acc X-Y plane
            window[:2] = rotation_matrix @ window[:2]
        
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.randn(*window.shape) * 0.1
            window = window + noise
        
        return window
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx].copy()
        window = self._augment(window)
        
        return {
            'input': torch.from_numpy(window).float(),
            'activity_label': torch.tensor(self.activity_labels[idx], dtype=torch.long),
            'fall_label': torch.tensor(self.fall_labels[idx], dtype=torch.float32),
            'sample_weight': torch.tensor(self.fall_weights[idx], dtype=torch.float32)
        }
    
    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """Get a sampler that balances fall/non-fall samples"""
        return WeightedRandomSampler(
            weights=self.fall_weights,
            num_samples=len(self.fall_weights),
            replacement=True
        )


def create_dataloaders(
    data_path: str,
    batch_size: int = 64,
    config: Optional[DataConfig] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    config = config or DataConfig()
    
    # Create datasets
    train_dataset = IMUDataset(data_path, 'train', config, augment=True)
    val_dataset = IMUDataset(data_path, 'val', config, augment=False)
    test_dataset = IMUDataset(data_path, 'test', config, augment=False)
    
    # Create balanced sampler for training
    train_sampler = train_dataset.get_balanced_sampler()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
