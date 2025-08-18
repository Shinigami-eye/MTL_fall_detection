"""Signal filtering utilities."""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class SignalFilter:
    """Apply various filters to sensor signals."""
    
    def __init__(self, sampling_rate: float = 50.0):
        """
        Initialize signal filter.
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
    
    def lowpass(self, data: np.ndarray, cutoff: float,
                order: int = 4) -> np.ndarray:
        """
        Apply lowpass filter.
        
        Args:
            data: Input signal
            cutoff: Cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered signal
        """
        sos = signal.butter(order, cutoff / self.nyquist,
                           btype='low', output='sos')
        return signal.sosfiltfilt(sos, data, axis=0)
    
    def highpass(self, data: np.ndarray, cutoff: float,
                 order: int = 4) -> np.ndarray:
        """
        Apply highpass filter.
        
        Args:
            data: Input signal
            cutoff: Cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered signal
        """
        sos = signal.butter(order, cutoff / self.nyquist,
                           btype='high', output='sos')
        return signal.sosfiltfilt(sos, data, axis=0)
    
    def bandpass(self, data: np.ndarray, low_cutoff: float,
                 high_cutoff: float, order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter.
        
        Args:
            data: Input signal
            low_cutoff: Low cutoff frequency in Hz
            high_cutoff: High cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered signal
        """
        sos = signal.butter(order, [low_cutoff / self.nyquist,
                                   high_cutoff / self.nyquist],
                           btype='band', output='sos')
        return signal.sosfiltfilt(sos, data, axis=0)
    
    def median_filter(self, data: np.ndarray,
                      kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter for spike removal.
        
        Args:
            data: Input signal
            kernel_size: Size of median filter kernel
            
        Returns:
            Filtered signal
        """
        return signal.medfilt(data, kernel_size=kernel_size)
    
    def remove_gravity(self, accelerometer_data: np.ndarray,
                      cutoff: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate gravity and linear acceleration components.
        
        Args:
            accelerometer_data: Raw accelerometer data (n_samples, 3)
            cutoff: Cutoff frequency for gravity component
            
        Returns:
            Tuple of (linear_acceleration, gravity)
        """
        # Gravity is the low-frequency component
        gravity = self.lowpass(accelerometer_data, cutoff)
        
        # Linear acceleration is the difference
        linear_acc = accelerometer_data - gravity
        
        return linear_acc, gravity
